"""
Triton kernel (and pure-PyTorch fallback) for asymmetric attention score
computation from packed compressed KV representations.

The asymmetric estimator computes:
    <q, k> ≈ <q, k_mse> + ||r_k|| * coeff * <S@q, sign(S@r_k)>

where k_mse is reconstructed from packed MSE centroid indices and the QJL signs
are stored as 1-bit packed values — so no full fp16 key tensor is ever materialised.

Bit layout follows the same little-endian convention as bit_ops.py:
    Index i starts at bit position  i * b
    It occupies bytes  (i*b)//8  and (if it straddles a boundary) (i*b)//8 + 1
"""

import warnings

import torch

from .bit_ops import unpack_bits

# ---------------------------------------------------------------------------
# Probe Triton availability (import + minimal GPU driver check)
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
_triton_unavailable_reason: str = ""

try:
    import triton
    import triton.language as tl

    # Triton imported, but the GPU driver may still be missing. Run a cheap
    # probe so we know whether the runtime is actually usable.
    try:
        from triton.runtime import driver as _triton_driver
        _ = _triton_driver.active   # triggers driver init / compile
        _TRITON_AVAILABLE = True
    except Exception as _probe_err:
        _triton_unavailable_reason = (
            f"Triton imported but GPU driver unavailable: {_probe_err}"
        )
        warnings.warn(
            f"asymmetric_attention_scores will use pure-PyTorch fallback. "
            f"({_triton_unavailable_reason})",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError as _import_err:
    _triton_unavailable_reason = str(_import_err)
    warnings.warn(
        f"Triton is not available ({_triton_unavailable_reason}). "
        "asymmetric_attention_scores will use pure-PyTorch fallback "
        "(no GPU acceleration).",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Triton kernel (only defined when Triton + GPU are available)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_S": bs}, num_warps=nw)
            for bs in [16, 32, 64]
            for nw in [2, 4, 8]
        ],
        key=["S_k", "d", "m", "b"],
    )
    @triton.jit
    def asymmetric_attention_kernel(
        # Inputs
        query_ptr,            # float16[B, H, S_q, d]
        packed_key_indices_ptr,  # uint8[B, H, S_k, ceil(d*(b-1)/8)]
        packed_qjl_signs_ptr,    # uint8[B, H, S_k, ceil(m/8)]
        residual_norms_ptr,      # float16[B, H, S_k]
        centroids_ptr,           # float16[H, 2^(b-1)]
        qjl_matrix_ptr,          # float16[m, d]
        vec_norms_ptr,           # float16[B, H, S_k]  — per-key vector norms
        # Output
        out_ptr,                  # float32[B, H, S_q, S_k]
        # Dimensions
        B,
        H,
        S_q,
        S_k,
        d: tl.constexpr,
        m: tl.constexpr,
        b: tl.constexpr,
        b_eff: tl.constexpr,
        # Strides for query [B, H, S_q, d]
        stride_qb,
        stride_qh,
        stride_qsq,
        stride_qd,
        # Strides for packed_key_indices [B, H, S_k, key_packed_cols]
        stride_kb,
        stride_kh,
        stride_ksk,
        stride_kc,
        # Strides for packed_qjl_signs [B, H, S_k, sign_packed_cols]
        stride_sb,
        stride_sh,
        stride_ssk,
        stride_sc,
        # Strides for residual_norms [B, H, S_k]
        stride_rnb,
        stride_rnh,
        stride_rnsk,
        # Stride for centroids [H, 2^(b-1)]
        stride_ch,
        stride_cc,
        # Strides for qjl_matrix [m, d]
        stride_qjm,
        stride_qjd,
        # Strides for vec_norms [B, H, S_k]
        stride_vnb,
        stride_vnh,
        stride_vnsk,
        # Strides for output [B, H, S_q, S_k]
        stride_ob,
        stride_oh,
        stride_osq,
        stride_osk,
        # Scalar
        coeff,
        # Autotuned
        BLOCK_S: tl.constexpr,
    ):
        """One program per (batch * H). Iterates over all (sq, sk_tile) pairs."""
        bh_idx = tl.program_id(0)
        batch_idx = bh_idx // H
        head_idx = bh_idx % H

        n_centroids = 1 << b_eff  # 2^(b-1) centroids per head
        key_mask = n_centroids - 1  # bitmask for b_eff bits
        sign_mask = (1 << 1) - 1  # 1-bit signs: mask is (1 << 1) - 1 = 1

        # Iterate over all query positions
        for sq in range(S_q):

            # --- Load query vector q[batch, head, sq, :] (d elements) ---
            # We load all d elements into registers.
            # Triton requires a power-of-2 or static block; we iterate element-by-element
            # since d is constexpr.
            q_base = (
                query_ptr
                + batch_idx * stride_qb
                + head_idx * stride_qh
                + sq * stride_qsq
            )

            # Compute S@q (project query through QJL matrix) — m elements
            # q_proj[j] = sum_i qjl_matrix[j, i] * q[i]
            # Precompute once per query position; reused across all key positions.
            q_proj = [tl.zeros([1], dtype=tl.float32)[0] for _ in tl.static_range(0, m)]
            for j in tl.static_range(0, m):
                acc = tl.zeros([1], dtype=tl.float32)[0]
                for i in tl.static_range(0, d):
                    q_val = tl.load(q_base + i * stride_qd).to(tl.float32)
                    s_val = tl.load(
                        qjl_matrix_ptr + j * stride_qjm + i * stride_qjd
                    ).to(tl.float32)
                    acc += q_val * s_val
                q_proj[j] = acc

            # Iterate over key tiles
            for sk_start in range(0, S_k, BLOCK_S):

                # Process each key in the tile
                for sk_off in tl.static_range(0, BLOCK_S):
                    sk = sk_start + sk_off
                    if sk < S_k:

                        # ----- Term 1: q · k_mse -----
                        # Unpack b_eff-bit indices for this key, look up centroids,
                        # compute dot product with query.

                        key_base = (
                            packed_key_indices_ptr
                            + batch_idx * stride_kb
                            + head_idx * stride_kh
                            + sk * stride_ksk
                        )
                        centroid_base = (
                            centroids_ptr
                            + head_idx * stride_ch
                        )

                        dot_mse = tl.zeros([1], dtype=tl.float32)[0]

                        for i in tl.static_range(0, d):
                            # Unpack index i with b_eff bits
                            bit_pos = i * b_eff
                            byte_idx = bit_pos // 8
                            bit_off = bit_pos % 8

                            lo_byte = tl.load(key_base + byte_idx * stride_kc).to(tl.int32)
                            idx = (lo_byte >> bit_off) & key_mask

                            bits_in_lo = 8 - bit_off
                            if b_eff > bits_in_lo:
                                hi_byte = tl.load(key_base + (byte_idx + 1) * stride_kc).to(tl.int32)
                                idx = idx | (
                                    (hi_byte & ((1 << (b_eff - bits_in_lo)) - 1)) << bits_in_lo
                                )

                            # Centroid lookup
                            c_val = tl.load(centroid_base + idx * stride_cc).to(tl.float32)

                            # Query element
                            q_val = tl.load(q_base + i * stride_qd).to(tl.float32)

                            dot_mse += q_val * c_val

                        # ----- Term 2: QJL correction -----
                        # Unpack 1-bit signs for this key (m elements), compute
                        # (S@q) · signs

                        sign_base = (
                            packed_qjl_signs_ptr
                            + batch_idx * stride_sb
                            + head_idx * stride_sh
                            + sk * stride_ssk
                        )

                        dot_qjl = tl.zeros([1], dtype=tl.float32)[0]

                        for j in tl.static_range(0, m):
                            # Unpack 1-bit sign j
                            bit_pos_s = j  # b=1, so bit_pos = j * 1 = j
                            byte_idx_s = bit_pos_s // 8
                            bit_off_s = bit_pos_s % 8

                            sign_byte = tl.load(sign_base + byte_idx_s * stride_sc).to(tl.int32)
                            raw_bit = (sign_byte >> bit_off_s) & sign_mask
                            # Convert {0, 1} -> {-1, +1}: sign = 2*bit - 1
                            sign_val = (raw_bit * 2 - 1).to(tl.float32)

                            # Use precomputed q_proj[j] (computed once per sq, outside key loop)
                            dot_qjl += q_proj[j] * sign_val

                        # ----- Residual norm -----
                        r_norm = tl.load(
                            residual_norms_ptr
                            + batch_idx * stride_rnb
                            + head_idx * stride_rnh
                            + sk * stride_rnsk
                        ).to(tl.float32)

                        # ----- Vector norm (per-key scale) -----
                        # Centroids are in the normalized+rotated space; dot_mse must be
                        # scaled by the original vector norm to recover the true inner product.
                        v_norm = tl.load(
                            vec_norms_ptr
                            + batch_idx * stride_vnb
                            + head_idx * stride_vnh
                            + sk * stride_vnsk
                        ).to(tl.float32)

                        # ----- Combine and store -----
                        score = dot_mse * v_norm + coeff * r_norm * dot_qjl

                        out_offset = (
                            batch_idx * stride_ob
                            + head_idx * stride_oh
                            + sq * stride_osq
                            + sk * stride_osk
                        )
                        tl.store(out_ptr + out_offset, score)

    def asymmetric_attention_scores(
        query: torch.Tensor,
        packed_key_indices: torch.Tensor,
        packed_qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
        centroids: torch.Tensor,
        qjl_matrix: torch.Tensor,
        vec_norms: torch.Tensor,
        b: int,
        coeff: float,
    ) -> torch.Tensor:
        """Compute asymmetric attention scores using the Triton kernel.

        Parameters
        ----------
        query : torch.Tensor
            float16[B, H, S_q, d] — queries pre-rotated into centroid space (q @ Pi.T)
        packed_key_indices : torch.Tensor
            uint8[B, H, S_k, ceil(d*(b-1)/8)]  — MSE centroid indices packed with b-1 bits
        packed_qjl_signs : torch.Tensor
            uint8[B, H, S_k, ceil(m/8)]  — QJL signs packed with 1 bit each
        residual_norms : torch.Tensor
            float16[B, H, S_k]
        centroids : torch.Tensor
            float16[H, 2^(b-1)]
        qjl_matrix : torch.Tensor
            float16[m, d] — effective QJL matrix (S @ Pi.T) for pre-rotated queries
        vec_norms : torch.Tensor
            float16[B, H, S_k] — per-key original vector norms
        b : int
            Total bits (MSE uses b-1 bits; 1 bit reserved for sign encoding).
        coeff : float
            Correction scale factor (typically sqrt(pi/2) / m).

        Returns
        -------
        torch.Tensor
            float32[B, H, S_q, S_k]
        """
        B, H, S_q, d = query.shape
        S_k = packed_key_indices.shape[2]
        m = qjl_matrix.shape[0]

        # Ensure contiguous and correct dtypes
        query = query.contiguous()
        if query.dtype != torch.float16:
            query = query.to(torch.float16)
        packed_key_indices = packed_key_indices.contiguous()
        packed_qjl_signs = packed_qjl_signs.contiguous()
        residual_norms = residual_norms.contiguous()
        if residual_norms.dtype != torch.float16:
            residual_norms = residual_norms.to(torch.float16)
        centroids = centroids.contiguous()
        if centroids.dtype != torch.float16:
            centroids = centroids.to(torch.float16)
        qjl_matrix = qjl_matrix.contiguous()
        if qjl_matrix.dtype != torch.float16:
            qjl_matrix = qjl_matrix.to(torch.float16)
        vec_norms = vec_norms.contiguous()
        if vec_norms.dtype != torch.float16:
            vec_norms = vec_norms.to(torch.float16)

        out = torch.empty(B, H, S_q, S_k, dtype=torch.float32, device=query.device)

        b_eff = b - 1

        grid = (B * H,)
        asymmetric_attention_kernel[grid](
            query,
            packed_key_indices,
            packed_qjl_signs,
            residual_norms,
            centroids,
            qjl_matrix,
            vec_norms,
            out,
            B, H, S_q, S_k, d, m, b, b_eff,
            # query strides
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            # packed_key_indices strides
            packed_key_indices.stride(0), packed_key_indices.stride(1),
            packed_key_indices.stride(2), packed_key_indices.stride(3),
            # packed_qjl_signs strides
            packed_qjl_signs.stride(0), packed_qjl_signs.stride(1),
            packed_qjl_signs.stride(2), packed_qjl_signs.stride(3),
            # residual_norms strides
            residual_norms.stride(0), residual_norms.stride(1), residual_norms.stride(2),
            # centroids strides
            centroids.stride(0), centroids.stride(1),
            # qjl_matrix strides
            qjl_matrix.stride(0), qjl_matrix.stride(1),
            # vec_norms strides
            vec_norms.stride(0), vec_norms.stride(1), vec_norms.stride(2),
            # output strides
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            # scalar
            coeff,
        )
        return out

else:
    # -----------------------------------------------------------------------
    # Pure-PyTorch fallback — produces identical results to the Triton kernel
    # but runs without GPU/Triton (useful for CPU testing and CI).
    # -----------------------------------------------------------------------

    def asymmetric_attention_scores(
        query: torch.Tensor,
        packed_key_indices: torch.Tensor,
        packed_qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
        centroids: torch.Tensor,
        qjl_matrix: torch.Tensor,
        vec_norms: torch.Tensor,
        b: int,
        coeff: float,
    ) -> torch.Tensor:
        """Compute asymmetric attention scores (pure-PyTorch fallback).

        Parameters
        ----------
        query : torch.Tensor
            float16[B, H, S_q, d] — queries pre-rotated into centroid space (q @ Pi.T)
        packed_key_indices : torch.Tensor
            uint8[B, H, S_k, ceil(d*(b-1)/8)]
        packed_qjl_signs : torch.Tensor
            uint8[B, H, S_k, ceil(m/8)]
        residual_norms : torch.Tensor
            float16[B, H, S_k]
        centroids : torch.Tensor
            float16[H, 2^(b-1)]
        qjl_matrix : torch.Tensor
            float16[m, d] — effective QJL matrix (S @ Pi.T) for pre-rotated queries
        vec_norms : torch.Tensor
            float16[B, H, S_k] — per-key original vector norms
        b : int
            Total bits (MSE uses b-1 bits).
        coeff : float
            Correction scale factor.

        Returns
        -------
        torch.Tensor
            float32[B, H, S_q, S_k]
        """
        B, H, S_q, d = query.shape
        S_k = packed_key_indices.shape[2]
        m = qjl_matrix.shape[0]
        b_eff = b - 1  # bits used for MSE centroid indices

        # --- Term 1: q_rot · centroids * vec_norm ---
        # query is pre-rotated (q @ Pi.T), centroids are in the same rotated space.
        # Unpack MSE indices: uint8[B, H, S_k, ceil(d*b_eff/8)] -> int32[B*H*S_k, d]
        flat_packed_keys = packed_key_indices.reshape(B * H * S_k, -1)
        key_indices = unpack_bits(flat_packed_keys, b_eff, d)  # int32[B*H*S_k, d]
        key_indices = key_indices.reshape(B, H, S_k, d)        # int32[B, H, S_k, d]

        # Centroid lookup: centroids is float16[H, 2^(b-1)]
        h_idx = torch.arange(H, device=key_indices.device).view(1, H, 1, 1)
        k_rot = centroids[h_idx, key_indices]  # float16[B, H, S_k, d] — in rotated space

        # Term 1: (q_rot · k_rot) * vec_norm
        # vec_norms: [B, H, S_k] -> [B, H, 1, S_k] for broadcasting
        term1 = torch.matmul(
            query.float(),                   # [B, H, S_q, d]
            k_rot.float().transpose(-2, -1), # [B, H, d, S_k]
        ) * vec_norms.float().unsqueeze(-2)  # float32[B, H, S_q, S_k]

        # --- Term 2: QJL correction ---
        # Unpack 1-bit signs: uint8[B, H, S_k, ceil(m/8)] -> int32[B*H*S_k, m]
        flat_packed_signs = packed_qjl_signs.reshape(B * H * S_k, -1)
        raw_signs = unpack_bits(flat_packed_signs, 1, m)  # int32[B*H*S_k, m], values in {0, 1}
        raw_signs = raw_signs.reshape(B, H, S_k, m)       # int32[B, H, S_k, m]

        # Convert {0, 1} -> {-1, +1}: sign = 2*bit - 1
        signs = (raw_signs.float() * 2.0 - 1.0)  # float32[B, H, S_k, m]

        # Project pre-rotated queries through effective QJL matrix.
        # qjl_matrix = S_eff = S @ Pi.T, so q_rot @ S_eff.T = q_orig @ S.T
        q_proj = torch.matmul(query.float(), qjl_matrix.float().T)  # [B, H, S_q, m]

        # QJL inner products
        term2 = torch.matmul(q_proj, signs.transpose(-2, -1))  # float32[B, H, S_q, S_k]

        # Scale term2 by coeff * residual_norms
        scaled_term2 = coeff * residual_norms.float().unsqueeze(-2) * term2

        return term1 + scaled_term2


__all__ = ["asymmetric_attention_scores"]
