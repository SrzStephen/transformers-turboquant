"""
Triton kernels (and pure-PyTorch fallbacks) for bit-packing and unpacking
integer indices.

Bit layout (little-endian within bytes):
  Index i starts at bit position  i * b
  It occupies bytes  (i*b)//8  and (if it straddles a boundary) (i*b)//8 + 1
"""

import math
import warnings

import torch

# ---------------------------------------------------------------------------
# Probe Triton availability (import + minimal GPU driver check)
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
_triton_unavailable_reason: str = ""

try:
    import triton
    import triton.language as tl

    # Triton imported, but the GPU driver may still be missing.  Run a cheap
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
            f"pack_bits / unpack_bits will use pure-PyTorch fallbacks. "
            f"({_triton_unavailable_reason})",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError as _import_err:
    _triton_unavailable_reason = str(_import_err)
    warnings.warn(
        f"Triton is not available ({_triton_unavailable_reason}). "
        "pack_bits / unpack_bits will use pure-PyTorch fallbacks (no GPU acceleration).",
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
        key=["d", "b"],
    )
    @triton.jit
    def pack_bits_kernel(
        indices_ptr,   # int32[N, d]
        out_ptr,       # uint8[N, out_cols]
        N,
        d,
        b: tl.constexpr,
        stride_n,      # stride along N dim for indices (in elements)
        stride_out_n,  # stride along N dim for output (in elements)
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)

        # Base pointers for this row
        row_idx_ptr = indices_ptr + row * stride_n
        row_out_ptr = out_ptr + row * stride_out_n

        # Output is pre-zeroed by the Python wrapper (torch.zeros), so no
        # zero-init loop is needed here.

        # Pack each index in the row
        for start in range(0, d, BLOCK_D):
            for k in tl.static_range(0, BLOCK_D):
                i = start + k
                if i < d:
                    val = tl.load(row_idx_ptr + i).to(tl.int32)
                    bit_pos = i * b
                    byte_idx = bit_pos // 8
                    bit_off = bit_pos % 8

                    # Low byte contribution — atomic OR to avoid data races
                    # when multiple threads write to the same output byte.
                    lo_bits = ((val & ((1 << b) - 1)) << bit_off) & 0xFF
                    tl.atomic_or(row_out_ptr + byte_idx, lo_bits.to(tl.uint8))

                    # High byte contribution (when index straddles a byte boundary)
                    bits_in_lo = 8 - bit_off
                    if b > bits_in_lo:
                        next_byte = byte_idx + 1
                        hi_bits = (
                            (val >> bits_in_lo) & ((1 << (b - bits_in_lo)) - 1)
                        ) & 0xFF
                        tl.atomic_or(
                            row_out_ptr + next_byte,
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
    def unpack_bits_kernel(
        packed_ptr,    # uint8[N, out_cols]
        out_ptr,       # int32[N, d]
        N,
        d,
        b: tl.constexpr,
        stride_in_n,
        stride_out_n,
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
                    val = (lo >> bit_off) & mask_val

                    bits_in_lo = 8 - bit_off
                    if b > bits_in_lo:
                        hi = tl.load(row_packed_ptr + byte_idx + 1).to(tl.int32)
                        val = val | (
                            (hi & ((1 << (b - bits_in_lo)) - 1)) << bits_in_lo
                        )

                    tl.store(row_out_ptr + i, val.to(tl.int32))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _out_cols(d: int, b: int) -> int:
    """Number of uint8 bytes needed to store d indices of b bits each."""
    return math.ceil(d * b / 8)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    def pack_bits(indices: torch.Tensor, b: int) -> torch.Tensor:
        """Pack int32[N, d] indices into uint8[N, ceil(d*b/8)] using Triton.

        Parameters
        ----------
        indices : torch.Tensor
            Integer tensor of shape ``[N, d]`` with dtype ``torch.int32``.
        b : int
            Number of bits per index (1 <= b <= 8).

        Returns
        -------
        torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b/8)]``.
        """
        assert 1 <= b <= 8, f"b must be between 1 and 8, got {b}"
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)
        indices = indices.contiguous()

        N, d = indices.shape
        oc = _out_cols(d, b)
        out = torch.zeros(N, oc, dtype=torch.uint8, device=indices.device)

        grid = (N,)
        pack_bits_kernel[grid](
            indices, out,
            N, d, b,
            indices.stride(0),
            out.stride(0),
        )
        return out

    def unpack_bits(packed: torch.Tensor, b: int, d: int) -> torch.Tensor:
        """Unpack uint8[N, ceil(d*b/8)] into int32[N, d] using Triton.

        Parameters
        ----------
        packed : torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b/8)]``.
        b : int
            Number of bits per index.
        d : int
            Original number of indices per row.

        Returns
        -------
        torch.Tensor
            int32 tensor of shape ``[N, d]``.
        """
        assert 1 <= b <= 8, f"b must be between 1 and 8, got {b}"
        packed = packed.contiguous()
        N = packed.shape[0]
        out = torch.zeros(N, d, dtype=torch.int32, device=packed.device)

        grid = (N,)
        unpack_bits_kernel[grid](
            packed, out,
            N, d, b,
            packed.stride(0),
            out.stride(0),
        )
        return out

else:
    # ------------------------------------------------------------------
    # Pure-PyTorch fallbacks — produce identical outputs to the Triton
    # kernels but run on CPU (or GPU via PyTorch, without Triton speed).
    # ------------------------------------------------------------------

    def pack_bits(indices: torch.Tensor, b: int) -> torch.Tensor:
        """Pack int32[N, d] indices into uint8[N, ceil(d*b/8)] (pure PyTorch).

        Parameters
        ----------
        indices : torch.Tensor
            Integer tensor of shape ``[N, d]`` with dtype ``torch.int32``.
        b : int
            Number of bits per index (1 <= b <= 8).

        Returns
        -------
        torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b/8)]``.
        """
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        N, d = indices.shape
        oc = _out_cols(d, b)

        # Work in int32 to allow shifting without overflow issues
        out = torch.zeros(N, oc, dtype=torch.int32, device=indices.device)

        mask = (1 << b) - 1

        for i in range(d):
            bit_pos = i * b
            byte_idx = bit_pos // 8
            bit_off = bit_pos % 8

            val = indices[:, i] & mask  # (N,) int32

            # Low byte
            out[:, byte_idx] = out[:, byte_idx] | (val << bit_off)

            # High byte (straddle case)
            bits_in_lo = 8 - bit_off
            if b > bits_in_lo:
                hi_bits = (val >> bits_in_lo) & ((1 << (b - bits_in_lo)) - 1)
                out[:, byte_idx + 1] = out[:, byte_idx + 1] | hi_bits

        # Mask each byte to 8 bits and cast to uint8
        return (out & 0xFF).to(torch.uint8)

    def unpack_bits(packed: torch.Tensor, b: int, d: int) -> torch.Tensor:
        """Unpack uint8[N, ceil(d*b/8)] into int32[N, d] (pure PyTorch).

        Parameters
        ----------
        packed : torch.Tensor
            uint8 tensor of shape ``[N, ceil(d*b/8)]``.
        b : int
            Number of bits per index.
        d : int
            Original number of indices per row.

        Returns
        -------
        torch.Tensor
            int32 tensor of shape ``[N, d]``.
        """
        N = packed.shape[0]
        # Cast to int32 for arithmetic
        p = packed.to(torch.int32)

        out = torch.zeros(N, d, dtype=torch.int32, device=packed.device)
        mask = (1 << b) - 1

        for i in range(d):
            bit_pos = i * b
            byte_idx = bit_pos // 8
            bit_off = bit_pos % 8

            val = (p[:, byte_idx] >> bit_off) & mask

            bits_in_lo = 8 - bit_off
            if b > bits_in_lo:
                hi_mask = (1 << (b - bits_in_lo)) - 1
                val = val | ((p[:, byte_idx + 1] & hi_mask) << bits_in_lo)

            out[:, i] = val

        return out


__all__ = ["pack_bits", "unpack_bits"]
