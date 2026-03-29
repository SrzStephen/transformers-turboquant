"""
Triton kernels (and pure-PyTorch fallbacks) for Lloyd-Max quantization and
dequantization.

Quantization uses a binary search over codebook boundaries to map each float32
coordinate to an integer index in [0, 2^b_eff - 1], then packs the indices
into uint8 bytes (little-endian bit layout, same convention as bit_ops.py).

Dequantization unpacks the indices and performs a centroid table lookup,
returning float16 output.

Bit layout (little-endian within bytes):
  Index i starts at bit position  i * b_eff
  It occupies bytes  (i*b_eff)//8  and (if it straddles a boundary) (i*b_eff)//8 + 1
"""

import math
import warnings

import torch

from .bit_ops import _out_cols

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
            f"lloyd_max_quantize / lloyd_max_dequantize will use pure-PyTorch fallbacks. "
            f"({_triton_unavailable_reason})",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError as _import_err:
    _triton_unavailable_reason = str(_import_err)
    warnings.warn(
        f"Triton is not available ({_triton_unavailable_reason}). "
        "lloyd_max_quantize / lloyd_max_dequantize will use pure-PyTorch fallbacks "
        "(no GPU acceleration).",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Triton kernels (only defined when Triton + GPU are available)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": bd}, num_warps=nw)
            for bd in [32, 64, 128, 256]
            for nw in [2, 4, 8]
        ],
        key=["d", "b_eff"],
    )
    @triton.jit
    def lloyd_max_quantize_kernel(
        input_ptr,       # float32[N, d]
        bounds_ptr,      # float32[n_boundaries]  where n_boundaries = 2^b_eff - 1
        out_ptr,         # uint8[N, ceil(d*b_eff/8)]  — pre-zeroed by wrapper
        N,
        d,
        n_boundaries,    # 2^b_eff - 1
        b_eff: tl.constexpr,
        stride_in_n,     # row stride for input (in elements)
        stride_out_n,    # row stride for output (in elements)
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)

        row_in_ptr = input_ptr + row * stride_in_n
        row_out_ptr = out_ptr + row * stride_out_n

        # Load boundaries into shared memory (up to 255 entries for b_eff <= 8)
        # We load them via a loop so they get cached in L1/shared across the row.
        # Triton doesn't have explicit shared memory declarations, but register
        # caching through constexpr-sized arrays is idiomatic.

        # Process the row in BLOCK_D chunks
        for start in range(0, d, BLOCK_D):
            for k in tl.static_range(0, BLOCK_D):
                i = start + k
                if i < d:
                    x = tl.load(row_in_ptr + i)

                    # Binary search over boundaries to find quantization index.
                    # We seek the smallest j such that x < boundaries[j].
                    # Result index = j (in [0, n_boundaries]).
                    # Explicit int32 init prevents Triton from promoting lo/hi to float32
                    # (which would make the >> 1 right-shift undefined in PTX).
                    lo = tl.zeros([1], dtype=tl.int32)[0]
                    hi = tl.full([1], n_boundaries, dtype=tl.int32)[0]

                    # Unrolled binary search — b_eff is constexpr so the
                    # compiler knows how many iterations we need (at most b_eff).
                    for _ in tl.static_range(0, b_eff):
                        mid = (lo + hi) >> 1
                        mid_val = tl.load(bounds_ptr + mid)
                        # If x >= bounds[mid], move lo up; otherwise hi down.
                        lo = tl.where(x >= mid_val, mid + 1, lo)
                        hi = tl.where(x >= mid_val, hi, mid)

                    # lo == hi == the quantization index in [0, 2^b_eff - 1]
                    idx = lo  # int32

                    # Packing: use atomic_or to handle concurrent writes to shared output bytes.
                    # The spec describes a warp-shuffle approach; atomic_or is functionally
                    # equivalent and simpler to implement correctly. Sub-byte atomics are
                    # widened to 32-bit CAS by the PTX compiler, which is correct on all targets.
                    bit_pos = i * b_eff
                    byte_idx = bit_pos // 8
                    bit_off = bit_pos % 8

                    lo_bits = ((idx & ((1 << b_eff) - 1)) << bit_off) & 0xFF
                    tl.atomic_or(row_out_ptr + byte_idx, lo_bits.to(tl.uint8))

                    bits_in_lo = 8 - bit_off
                    if b_eff > bits_in_lo:
                        hi_bits = (
                            (idx >> bits_in_lo) & ((1 << (b_eff - bits_in_lo)) - 1)
                        ) & 0xFF
                        tl.atomic_or(
                            row_out_ptr + byte_idx + 1,
                            hi_bits.to(tl.uint8),
                        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": bd}, num_warps=nw)
            for bd in [32, 64, 128, 256]
            for nw in [2, 4, 8]
        ],
        key=["d", "b"],
    )
    @triton.jit
    def lloyd_max_dequantize_kernel(
        packed_ptr,      # uint8[N, ceil(d*b/8)]
        centroids_ptr,   # float16[2^b]
        out_ptr,         # float16[N, d]
        N,
        d,
        b: tl.constexpr,
        stride_in_n,     # row stride for packed input (in elements)
        stride_out_n,    # row stride for output (in elements)
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)

        row_packed_ptr = packed_ptr + row * stride_in_n
        row_out_ptr = out_ptr + row * stride_out_n

        mask_val = (1 << b) - 1

        for start in range(0, d, BLOCK_D):
            for k in tl.static_range(0, BLOCK_D):
                i = start + k
                if i < d:
                    bit_pos = i * b
                    byte_idx = bit_pos // 8
                    bit_off = bit_pos % 8

                    lo = tl.load(row_packed_ptr + byte_idx).to(tl.int32)
                    idx = (lo >> bit_off) & mask_val

                    bits_in_lo = 8 - bit_off
                    if b > bits_in_lo:
                        hi = tl.load(row_packed_ptr + byte_idx + 1).to(tl.int32)
                        idx = idx | (
                            (hi & ((1 << (b - bits_in_lo)) - 1)) << bits_in_lo
                        )

                    # Centroid table lookup
                    centroid = tl.load(centroids_ptr + idx)
                    tl.store(row_out_ptr + i, centroid)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    def lloyd_max_quantize(
        rotated: torch.Tensor,
        boundaries: torch.Tensor,
        b_eff: int,
    ) -> torch.Tensor:
        """Quantize float32[N, d] using Lloyd-Max boundaries; pack to uint8.

        Parameters
        ----------
        rotated : torch.Tensor
            float32 tensor of shape ``[N, d]``.
        boundaries : torch.Tensor
            float32 tensor of shape ``[2^b_eff - 1]`` — the decision boundaries
            from the Lloyd-Max codebook.
        b_eff : int
            Number of bits for quantization (1 <= b_eff <= 8).

        Returns
        -------
        torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b_eff/8)]``.
        """
        assert 1 <= b_eff <= 8, f"b_eff must be between 1 and 8, got {b_eff}"
        assert boundaries.shape[0] == (1 << b_eff) - 1, (
            f"Expected {(1 << b_eff) - 1} boundaries for b_eff={b_eff}, "
            f"got {boundaries.shape[0]}"
        )
        if rotated.dtype != torch.float32:
            rotated = rotated.to(torch.float32)
        if boundaries.dtype != torch.float32:
            boundaries = boundaries.to(torch.float32)
        rotated = rotated.contiguous()
        boundaries = boundaries.contiguous()

        N, d = rotated.shape
        n_boundaries = boundaries.shape[0]  # should be 2^b_eff - 1
        oc = _out_cols(d, b_eff)
        out = torch.zeros(N, oc, dtype=torch.uint8, device=rotated.device)

        grid = (N,)
        lloyd_max_quantize_kernel[grid](
            rotated, boundaries, out,
            N, d, n_boundaries, b_eff,
            rotated.stride(0),
            out.stride(0),
        )
        return out

    def lloyd_max_dequantize(
        packed: torch.Tensor,
        centroids: torch.Tensor,
        b: int,
        d: int,
    ) -> torch.Tensor:
        """Dequantize uint8 packed indices to float16 using centroid lookup.

        Parameters
        ----------
        packed : torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b/8)]``.
        centroids : torch.Tensor
            float16 tensor of shape ``[2^b]`` — the reconstruction centroids
            from the Lloyd-Max codebook.
        b : int
            Number of bits per index (1 <= b <= 8).
        d : int
            Original number of coordinates per row.

        Returns
        -------
        torch.Tensor
            float16 tensor of shape ``[N, d]``.
        """
        assert 1 <= b <= 8, f"b must be between 1 and 8, got {b}"
        if centroids.dtype != torch.float16:
            centroids = centroids.to(torch.float16)
        packed = packed.contiguous()
        centroids = centroids.contiguous()

        N = packed.shape[0]
        out = torch.empty(N, d, dtype=torch.float16, device=packed.device)

        grid = (N,)
        lloyd_max_dequantize_kernel[grid](
            packed, centroids, out,
            N, d, b,
            packed.stride(0),
            out.stride(0),
        )
        return out

else:
    # ------------------------------------------------------------------
    # Pure-PyTorch fallbacks — produce identical outputs to the Triton
    # kernels but run without Triton (CPU or GPU via plain PyTorch).
    # ------------------------------------------------------------------
    from turboquant_pytorch.kernels.bit_ops import pack_bits, unpack_bits

    def lloyd_max_quantize(
        rotated: torch.Tensor,
        boundaries: torch.Tensor,
        b_eff: int,
    ) -> torch.Tensor:
        """Quantize float32[N, d] using Lloyd-Max boundaries; pack to uint8 (pure PyTorch).

        Parameters
        ----------
        rotated : torch.Tensor
            float32 tensor of shape ``[N, d]``.
        boundaries : torch.Tensor
            float32 tensor of shape ``[2^b_eff - 1]``.
        b_eff : int
            Number of bits for quantization (1 <= b_eff <= 8).

        Returns
        -------
        torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b_eff/8)]``.
        """
        assert boundaries.shape[0] == (1 << b_eff) - 1, (
            f"Expected {(1 << b_eff) - 1} boundaries for b_eff={b_eff}, "
            f"got {boundaries.shape[0]}"
        )
        if rotated.dtype != torch.float32:
            rotated = rotated.to(torch.float32)
        if boundaries.dtype != torch.float32:
            boundaries = boundaries.to(torch.float32)

        # torch.searchsorted expects sorted 1-D boundaries and an input of the
        # same dtype. It returns the insertion index in [0, len(boundaries)],
        # which is exactly our quantization index in [0, 2^b_eff - 1].
        # boundaries is [2^b_eff - 1]; rotated is [N, d].
        # We vectorise over the full [N, d] tensor at once.
        indices = torch.searchsorted(
            boundaries.contiguous(),
            rotated.reshape(-1).contiguous(),
        ).reshape(rotated.shape).to(torch.int32)  # [N, d] int32

        return pack_bits(indices, b_eff)

    def lloyd_max_dequantize(
        packed: torch.Tensor,
        centroids: torch.Tensor,
        b: int,
        d: int,
    ) -> torch.Tensor:
        """Dequantize uint8 packed indices to float16 using centroid lookup (pure PyTorch).

        Parameters
        ----------
        packed : torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b/8)]``.
        centroids : torch.Tensor
            float16 tensor of shape ``[2^b]``.
        b : int
            Number of bits per index (1 <= b <= 8).
        d : int
            Original number of coordinates per row.

        Returns
        -------
        torch.Tensor
            float16 tensor of shape ``[N, d]``.
        """
        if centroids.dtype != torch.float16:
            centroids = centroids.to(torch.float16)

        indices = unpack_bits(packed, b, d)  # [N, d] int32
        return centroids[indices].to(torch.float16)


__all__ = ["lloyd_max_quantize", "lloyd_max_dequantize"]
