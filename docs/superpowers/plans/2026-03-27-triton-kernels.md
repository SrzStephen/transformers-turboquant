# Triton GPU Kernels for TurboQuant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pure-PyTorch internals of `TurboQuantCompressorV2` and `TurboQuantCompressorMSE` with Triton GPU kernels that bit-pack indices to reduce memory bandwidth, while preserving the exact public API.

**Architecture:** Five Triton kernels in a new `kernels/` subpackage. Rotation stays as `torch.matmul` (cuBLAS). A `use_triton=True` constructor flag enables PyTorch fallback. Storage changes from fp16 dequantised tensors to bit-packed integer indices — keys go from ~388 bytes/vector to ~68 bytes at b=3, d=128.

**Tech Stack:** Python 3.13, PyTorch ≥ 2.11, Triton ≥ 3.0, pytest, rich (transitive dep via typer)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/turboquant_pytorch/kernels/__init__.py` | Create | Re-exports all five kernel functions |
| `src/turboquant_pytorch/kernels/bit_ops.py` | Create | `pack_bits` and `unpack_bits` Triton kernels |
| `src/turboquant_pytorch/kernels/quantize.py` | Create | `lloyd_max_quantize` (fused quantize+pack) and `lloyd_max_dequantize` |
| `src/turboquant_pytorch/kernels/attention.py` | Create | `asymmetric_attention_scores` — fused centroid-lookup + dot + QJL |
| `src/turboquant_pytorch/compressors.py` | Modify | Add `use_triton` flag; swap compress/decompress/attention internals |
| `pyproject.toml` | Modify | Add `triton>=3.0.0`; add slow marker; exclude slow from default run |
| `justfile` | Modify | Add `test-slow` target |
| `tests/test_kernels.py` | Create | Unit: round-trip, quantize equivalence, attention equivalence |
| `tests/test_compressors.py` | Create | Integration: Triton vs PyTorch backend agreement |
| `tests/test_memory.py` | Create | Regression: Triton storage strictly smaller than fp16 baseline |
| `tests/test_slow.py` | Create | End-to-end: load Qwen, compare three modes, print rich table |

---

## Background: Bit-packing layout

Indices are packed in little-endian bit order across bytes. For `b` bits per index and dimension `d`:

- Index `i` occupies global bits `[i*b, i*b + b)`.
- Global bit `g` lives in byte `g // 8` at bit position `g % 8`.
- Output buffer is `uint8[N, ceil(d*b/8)]`.

**Pack kernel strategy (one thread per output byte):** For output byte `B`, iterate over bit positions 0..7. For bit position `p`, the global bit is `8*B + p`, which comes from index `(8*B+p)//b` at intra-index bit `(8*B+p)%b`. Extract that bit, place it at position `p` in the output byte. No sub-byte atomics — each thread owns exactly one output byte.

**Unpack kernel strategy (one thread per output index):** For output index `i`, iterate over bit positions `0..b-1`. Bit `j` of index `i` is at global bit `i*b+j`, which lives in byte `(i*b+j)//8` at bit position `(i*b+j)%8`. Load byte, shift, mask, OR into result.

---

## Task 1: Dependencies and test infrastructure

**Files:**
- Modify: `pyproject.toml`
- Modify: `justfile`

- [ ] **Step 1: Update pyproject.toml**

Add `triton>=3.0.0` to `dependencies`. Add slow marker and exclude slow tests from default `addopts`:

```toml
[project]
dependencies = [
    "accelerate>=1.13.0",
    "bitsandbytes>=0.49.2",
    "scipy>=1.17.1",
    "torch>=2.11.0",
    "transformers>=5.4.0",
    "triton>=3.0.0",
    "typer>=0.12",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-n auto --cov=transformers_turboquant --cov-report=term-missing --cov-report=xml -m 'not slow'"
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (run with just test-slow)",
]
```

- [ ] **Step 2: Add test-slow to justfile**

Append to `justfile`:

```
test-slow:
    uv run pytest -m slow -s -v
```

- [ ] **Step 3: Sync deps**

```bash
uv sync
```

Expected: resolves triton without errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml justfile uv.lock
git commit -m "chore: add triton>=3.0.0 dependency and test-slow target"
```

---

## Task 2: `pack_bits` and `unpack_bits` Triton kernels

**Files:**
- Create: `src/turboquant_pytorch/kernels/__init__.py`
- Create: `src/turboquant_pytorch/kernels/bit_ops.py`
- Create: `tests/test_kernels.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_kernels.py`:

```python
import math
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_pack_unpack_roundtrip(b, d):
    from turboquant_pytorch.kernels.bit_ops import pack_bits, unpack_bits

    N = 16
    max_val = (1 << b) - 1
    indices = torch.randint(0, max_val + 1, (N, d), dtype=torch.int32, device="cuda")
    packed = pack_bits(indices, b)
    recovered = unpack_bits(packed, b, d)
    assert recovered.shape == (N, d), f"Shape mismatch: {recovered.shape}"
    assert (recovered == indices).all(), f"Round-trip failed for b={b}, d={d}"


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
def test_pack_output_shape_and_dtype(b):
    from turboquant_pytorch.kernels.bit_ops import pack_bits

    N, d = 8, 128
    indices = torch.zeros(N, d, dtype=torch.int32, device="cuda")
    packed = pack_bits(indices, b)
    expected_bytes = math.ceil(d * b / 8)
    assert packed.shape == (N, expected_bytes), \
        f"Expected ({N}, {expected_bytes}), got {packed.shape}"
    assert packed.dtype == torch.uint8


@cuda_only
def test_pack_known_values():
    """b=4: two 4-bit indices pack into one byte. Index 0xA (1010) + 0x5 (0101) = byte 0x5A."""
    from turboquant_pytorch.kernels.bit_ops import pack_bits, unpack_bits

    b = 4
    indices = torch.tensor([[0xA, 0x5]], dtype=torch.int32, device="cuda")
    packed = pack_bits(indices, b)
    assert packed.shape == (1, 1)
    # Little-endian: index 0 occupies bits 0-3, index 1 occupies bits 4-7
    # byte = (0xA & 0xF) | ((0x5 & 0xF) << 4) = 0x0A | 0x50 = 0x5A
    assert packed[0, 0].item() == 0x5A, \
        f"Expected 0x5A, got 0x{packed[0, 0].item():02X}"
    recovered = unpack_bits(packed, b, 2)
    assert (recovered == indices).all()
```

- [ ] **Step 2: Run to confirm ImportError failure**

```bash
uv run pytest tests/test_kernels.py -v -m "" 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'turboquant_pytorch.kernels'`

- [ ] **Step 3: Create the kernels package**

Create `src/turboquant_pytorch/kernels/__init__.py`:

```python
from .bit_ops import pack_bits, unpack_bits
from .quantize import lloyd_max_quantize, lloyd_max_dequantize
from .attention import asymmetric_attention_scores

__all__ = [
    "pack_bits",
    "unpack_bits",
    "lloyd_max_quantize",
    "lloyd_max_dequantize",
    "asymmetric_attention_scores",
]
```

- [ ] **Step 4: Implement bit_ops.py**

Create `src/turboquant_pytorch/kernels/bit_ops.py`:

```python
"""
Bit-packing / unpacking Triton kernels.

  pack_bits:   int32[N, d]              -> uint8[N, ceil(d*b/8)]
  unpack_bits: uint8[N, ceil(d*b/8)]   -> int32[N, d]

Indices are packed in little-endian bit order: index 0 occupies the
lowest-order bits of byte 0, index 1 follows immediately, etc.

Pack strategy  — one SIMD lane per output byte:
  For each bit position p in [0,8):
    global_bit  = byte_idx * 8 + p
    index_pos   = global_bit // b
    bit_in_idx  = global_bit % b
    result |= ((index[index_pos] >> bit_in_idx) & 1) << p

Unpack strategy — one SIMD lane per output index:
  For each bit position j in [0,b):
    global_bit  = idx * b + j
    byte_idx    = global_bit // 8
    bit_in_byte = global_bit % 8
    result |= ((packed[byte_idx] >> bit_in_byte) & 1) << j
"""

import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_BYTES": 32}, num_warps=4),
        triton.Config({"BLOCK_BYTES": 64}, num_warps=4),
        triton.Config({"BLOCK_BYTES": 128}, num_warps=8),
    ],
    key=["n_bytes", "b"],
)
@triton.jit
def _pack_bits_kernel(
    indices_ptr,   # int32[N, d]
    packed_ptr,    # uint8[N, n_bytes]
    N, d, n_bytes,
    b: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    byte_block_start = tl.program_id(1) * BLOCK_BYTES

    byte_offsets = byte_block_start + tl.arange(0, BLOCK_BYTES)
    mask = byte_offsets < n_bytes

    result = tl.zeros([BLOCK_BYTES], dtype=tl.uint8)

    # Iterate over all 8 bit positions within each output byte.
    # b is tl.constexpr so tl.static_range is available.
    for bit_pos in tl.static_range(8):
        # Global bit index for each lane's byte at this bit position
        global_bit = byte_offsets * 8 + bit_pos  # [BLOCK_BYTES]
        # Which input index owns this global bit?
        idx_pos = global_bit // b               # [BLOCK_BYTES]
        # Which bit within that input index?
        bit_in_idx = global_bit % b             # [BLOCK_BYTES]

        valid = mask & (idx_pos < d)

        # Gather: load input index value for each lane (scatter read)
        idx_val = tl.load(
            indices_ptr + row * d + idx_pos,
            mask=valid,
            other=0,
        )  # [BLOCK_BYTES] int32

        # Extract the target bit and place it at bit_pos in the output byte
        bit_val = ((idx_val >> bit_in_idx) & 1).to(tl.uint8)
        result = result | (bit_val << tl.constexpr(bit_pos))

    tl.store(packed_ptr + row * n_bytes + byte_offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["d", "b"],
)
@triton.jit
def _unpack_bits_kernel(
    packed_ptr,    # uint8[N, n_bytes]
    indices_ptr,   # int32[N, d]
    N, d, n_bytes,
    b: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_block_start = tl.program_id(1) * BLOCK_D

    d_offsets = d_block_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < d

    result = tl.zeros([BLOCK_D], dtype=tl.int32)

    # Iterate over each bit within the b-bit index.
    for bit_j in tl.static_range(b):
        # Global bit position for bit j of each index in this block
        global_bit = d_offsets * b + bit_j        # [BLOCK_D]
        byte_idx    = global_bit // 8             # [BLOCK_D]
        bit_in_byte = global_bit % 8              # [BLOCK_D]

        # Gather: read the byte containing this bit (may hit same byte multiple times
        # across lanes when b > 1 — that's fine, reads are idempotent)
        byte_val = tl.load(
            packed_ptr + row * n_bytes + byte_idx,
            mask=mask,
            other=0,
        ).to(tl.int32)  # [BLOCK_D]

        bit_val = (byte_val >> bit_in_byte) & 1
        result = result | (bit_val << tl.constexpr(bit_j))

    tl.store(indices_ptr + row * d + d_offsets, result, mask=mask)


def pack_bits(indices: torch.Tensor, b: int) -> torch.Tensor:
    """
    Pack int32[N, d] indices (values in [0, 2^b)) into uint8[N, ceil(d*b/8)].

    Args:
        indices: int32 tensor on CUDA, shape [N, d]
        b:       bits per index, 1 <= b <= 8

    Returns:
        uint8 tensor, shape [N, ceil(d*b/8)]
    """
    assert indices.is_cuda, "indices must be on CUDA"
    assert indices.dtype == torch.int32, f"expected int32, got {indices.dtype}"
    assert 1 <= b <= 8, f"b must be in [1, 8], got {b}"
    N, d = indices.shape
    n_bytes = math.ceil(d * b / 8)
    packed = torch.zeros(N, n_bytes, dtype=torch.uint8, device=indices.device)
    grid = lambda meta: (N, triton.cdiv(n_bytes, meta["BLOCK_BYTES"]))
    _pack_bits_kernel[grid](indices, packed, N, d, n_bytes, b)
    return packed


def unpack_bits(packed: torch.Tensor, b: int, d: int) -> torch.Tensor:
    """
    Unpack uint8[N, ceil(d*b/8)] into int32[N, d].

    Args:
        packed: uint8 tensor on CUDA, shape [N, ceil(d*b/8)]
        b:      bits per index used during packing
        d:      original number of indices per row

    Returns:
        int32 tensor, shape [N, d]
    """
    assert packed.is_cuda, "packed must be on CUDA"
    assert packed.dtype == torch.uint8, f"expected uint8, got {packed.dtype}"
    N, n_bytes = packed.shape
    indices = torch.zeros(N, d, dtype=torch.int32, device=packed.device)
    grid = lambda meta: (N, triton.cdiv(d, meta["BLOCK_D"]))
    _unpack_bits_kernel[grid](packed, indices, N, d, n_bytes, b)
    return indices
```

- [ ] **Step 5: Run tests — expect PASS (or skip if no CUDA)**

```bash
uv run pytest tests/test_kernels.py -v -m "" -k "pack or unpack"
```

Expected: all `test_pack_*` tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant_pytorch/kernels/ tests/test_kernels.py
git commit -m "feat: add pack_bits/unpack_bits Triton kernels with round-trip tests"
```

---

## Task 3: `lloyd_max_quantize` and `lloyd_max_dequantize` Triton kernels

**Files:**
- Create: `src/turboquant_pytorch/kernels/quantize.py`
- Modify: `tests/test_kernels.py`

- [ ] **Step 1: Append failing quantize tests to test_kernels.py**

```python
@cuda_only
@pytest.mark.parametrize("b_eff", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128])
def test_lloyd_max_quantize_matches_searchsorted(b_eff, d):
    """Triton quantizer must match torch.searchsorted on same input."""
    from turboquant_pytorch.kernels.quantize import lloyd_max_quantize
    from turboquant_pytorch.kernels.bit_ops import unpack_bits
    from turboquant_pytorch.lloyd_max import solve_lloyd_max

    N = 32
    torch.manual_seed(0)
    centroids, boundaries = solve_lloyd_max(d, b_eff)
    boundaries_gpu = boundaries.cuda()

    # Input: values drawn from the coordinate distribution (~N(0, 1/d))
    x = torch.randn(N, d, device="cuda") / math.sqrt(d)

    # Reference: torch.searchsorted counts how many boundaries x exceeds
    ref_indices = torch.searchsorted(
        boundaries_gpu.contiguous(), x.contiguous()
    ).to(torch.int32)  # [N, d], values in [0, 2^b_eff)

    # Triton: quantize returns packed uint8; unpack to compare
    packed = lloyd_max_quantize(x.float(), boundaries_gpu, b_eff)
    triton_indices = unpack_bits(packed, b_eff, d)

    assert triton_indices.shape == ref_indices.shape
    mismatches = (triton_indices != ref_indices).sum().item()
    assert mismatches == 0, \
        f"{mismatches} mismatches for b_eff={b_eff}, d={d}"


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128])
def test_lloyd_max_dequantize_returns_valid_centroids(b, d):
    """Every dequantized value must equal one of the codebook centroids."""
    from turboquant_pytorch.kernels.quantize import lloyd_max_quantize, lloyd_max_dequantize
    from turboquant_pytorch.lloyd_max import solve_lloyd_max

    N = 32
    torch.manual_seed(1)
    centroids, boundaries = solve_lloyd_max(d, b)
    boundaries_gpu = boundaries.cuda()
    centroids_gpu = centroids.cuda().half()

    x = torch.randn(N, d, device="cuda") / math.sqrt(d)
    packed = lloyd_max_quantize(x.float(), boundaries_gpu, b)
    reconstructed = lloyd_max_dequantize(packed, centroids_gpu, b, d)

    assert reconstructed.shape == (N, d)
    assert reconstructed.dtype == torch.float16

    # Each reconstructed value must exactly equal a centroid
    centroids_set = centroids.tolist()
    rec_flat = reconstructed.cpu().float().reshape(-1).tolist()
    for val in rec_flat:
        assert any(abs(val - c) < 1e-3 for c in centroids_set), \
            f"Dequantized value {val:.6f} not found in centroids"


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128])
def test_quantize_dequantize_roundtrip_mse(b, d):
    """Quantize then dequantize MSE should be close to the PyTorch reference."""
    from turboquant_pytorch.kernels.quantize import lloyd_max_quantize, lloyd_max_dequantize
    from turboquant_pytorch.lloyd_max import solve_lloyd_max

    N = 64
    torch.manual_seed(2)
    centroids, boundaries = solve_lloyd_max(d, b)
    boundaries_gpu = boundaries.cuda()
    centroids_gpu = centroids.cuda()

    x = torch.randn(N, d, device="cuda") / math.sqrt(d)

    # PyTorch reference
    diffs = x.unsqueeze(-1) - centroids_gpu
    ref_indices = diffs.abs().argmin(dim=-1)
    ref_recon = centroids_gpu[ref_indices]

    # Triton
    packed = lloyd_max_quantize(x.float(), boundaries_gpu, b)
    triton_recon = lloyd_max_dequantize(packed, centroids_gpu.half(), b, d).float()

    torch.testing.assert_close(triton_recon, ref_recon, atol=1e-3, rtol=0,
        msg=f"Quantize-dequantize MSE mismatch for b={b}, d={d}")
```

- [ ] **Step 2: Run to confirm ImportError**

```bash
uv run pytest tests/test_kernels.py -v -m "" -k "quantize or dequantize" 2>&1 | head -20
```

- [ ] **Step 3: Implement quantize.py**

Create `src/turboquant_pytorch/kernels/quantize.py`:

```python
"""
Lloyd-Max quantize / dequantize Triton kernels.

  lloyd_max_quantize:   float32[N, d], boundaries float32[2^b_eff-1]
                        -> uint8[N, ceil(d*b_eff/8)]   (fused quantize + pack)

  lloyd_max_dequantize: uint8[N, ceil(d*b/8)], centroids float16[2^b]
                        -> float16[N, d]

The quantize kernel uses a linear scan over the sorted boundaries to count
how many boundaries each input value exceeds — equivalent to searchsorted.
Since b_eff is tl.constexpr, the number of boundaries (2^b_eff - 1) is
also constexpr, enabling tl.static_range for full loop unrolling.
"""

import math
import torch
import triton
import triton.language as tl
from .bit_ops import pack_bits, unpack_bits


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["d", "b_eff"],
)
@triton.jit
def _lloyd_max_quantize_kernel(
    x_ptr,          # float32[N, d]
    bounds_ptr,     # float32[2^b_eff - 1]  sorted boundaries
    indices_ptr,    # int32[N, d]            output
    N, d,
    b_eff: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    For each value in x, count how many boundaries it exceeds.
    This equals the index of the nearest centroid (equivalent to searchsorted).
    """
    n_boundaries: tl.constexpr = (1 << b_eff) - 1

    row = tl.program_id(0)
    d_start = tl.program_id(1) * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < d

    x_vals = tl.load(x_ptr + row * d + d_offsets, mask=mask, other=0.0)

    count = tl.zeros([BLOCK_D], dtype=tl.int32)

    # Static loop: unrolled at compile time since n_boundaries is constexpr.
    # Each iteration loads one scalar boundary and updates the count vector.
    for i in tl.static_range(n_boundaries):
        bound = tl.load(bounds_ptr + i)    # scalar broadcast
        count = count + (x_vals > bound).to(tl.int32)

    tl.store(indices_ptr + row * d + d_offsets, count, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["d", "b"],
)
@triton.jit
def _lloyd_max_dequantize_kernel(
    indices_ptr,    # int32[N, d]
    centroids_ptr,  # float16[2^b]
    out_ptr,        # float16[N, d]
    N, d,
    BLOCK_D: tl.constexpr,
):
    """Gather centroid value for each index."""
    row = tl.program_id(0)
    d_start = tl.program_id(1) * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < d

    idx = tl.load(indices_ptr + row * d + d_offsets, mask=mask, other=0)
    val = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + row * d + d_offsets, val, mask=mask)


def lloyd_max_quantize(
    x: torch.Tensor,
    boundaries: torch.Tensor,
    b_eff: int,
) -> torch.Tensor:
    """
    Quantize float32[N, d] to packed uint8[N, ceil(d*b_eff/8)].

    Args:
        x:          float32[N, d] on CUDA, pre-rotated input values
        boundaries: float32[2^b_eff - 1] sorted codebook boundaries on CUDA
        b_eff:      effective bit width (b-1 for keys, b for values)

    Returns:
        uint8[N, ceil(d*b_eff/8)] bit-packed indices
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert boundaries.is_cuda
    N, d = x.shape
    expected_n_bounds = (1 << b_eff) - 1
    assert boundaries.shape[0] == expected_n_bounds, (
        f"Expected {expected_n_bounds} boundaries for b_eff={b_eff}, "
        f"got {boundaries.shape[0]}"
    )

    # Step 1: Triton kernel — compute raw integer indices
    indices = torch.empty(N, d, dtype=torch.int32, device=x.device)
    grid = lambda meta: (N, triton.cdiv(d, meta["BLOCK_D"]))
    _lloyd_max_quantize_kernel[grid](
        x.contiguous(), boundaries.float().contiguous(), indices,
        N, d, b_eff,
    )

    # Step 2: Pack indices into bits using pack_bits kernel
    return pack_bits(indices, b_eff)


def lloyd_max_dequantize(
    packed: torch.Tensor,
    centroids: torch.Tensor,
    b: int,
    d: int,
) -> torch.Tensor:
    """
    Dequantize packed uint8[N, ceil(d*b/8)] to float16[N, d].

    Args:
        packed:    uint8[N, ceil(d*b/8)] on CUDA
        centroids: float16[2^b] centroid lookup table on CUDA
        b:         bit width used during packing
        d:         original dimension

    Returns:
        float16[N, d]
    """
    assert packed.is_cuda and packed.dtype == torch.uint8
    N = packed.shape[0]

    # Step 1: Unpack to int32 indices
    indices = unpack_bits(packed, b, d)  # int32[N, d]

    # Step 2: Gather centroid values
    out = torch.empty(N, d, dtype=torch.float16, device=packed.device)
    grid = lambda meta: (N, triton.cdiv(d, meta["BLOCK_D"]))
    _lloyd_max_dequantize_kernel[grid](
        indices, centroids.half().contiguous(), out, N, d,
    )
    return out
```

- [ ] **Step 4: Run quantize tests**

```bash
uv run pytest tests/test_kernels.py -v -m "" -k "quantize or dequantize"
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turboquant_pytorch/kernels/quantize.py tests/test_kernels.py
git commit -m "feat: add lloyd_max_quantize/dequantize Triton kernels"
```

---

## Task 4: `asymmetric_attention_scores` Triton kernel

**Files:**
- Create: `src/turboquant_pytorch/kernels/attention.py`
- Modify: `tests/test_kernels.py`

### Design notes

The kernel receives **pre-unpacked** indices (from `unpack_bits`, called in the Python wrapper). This keeps the kernel focused on the fused centroid-lookup + dot-product computation. Keys are never materialised as fp16 — centroid values are looked up directly from packed indices in registers.

S@q is precomputed via `torch.matmul` (cuBLAS) in the wrapper, once per query, and passed in as `float32[B*H*S_q, m]`. This avoids recomputing it for every key.

Grid: `(B*H, S_q, ceil(S_k/BLOCK_SK))`. Each program instance handles one `(batch*head, query, key_block)` triple.

For each key `sk` in the block:
- **Term 1:** `sum_{d} q[d] * centroid[key_idx[sk, d]]` — inner loop over d, scalar q load + gather centroid
- **Term 2:** `sum_{m} Sq[m] * (2*sign[sk, m] - 1)` — inner loop over m, scalar Sq load + gather sign bit

- [ ] **Step 1: Append failing attention test to test_kernels.py**

```python
@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
def test_asymmetric_attention_matches_pytorch(b):
    """
    Triton attention kernel must agree with PyTorch einsum reference
    within fp16 accumulation tolerance (atol=1e-2).
    """
    from turboquant_pytorch.kernels.attention import asymmetric_attention_scores
    from turboquant_pytorch.kernels.bit_ops import pack_bits
    from turboquant_pytorch.lloyd_max import solve_lloyd_max

    torch.manual_seed(42)
    B, H, S_q, S_k, d = 1, 4, 8, 32, 64
    mse_bits = b - 1
    m = d  # current impl always has m == d

    centroids, boundaries = solve_lloyd_max(d, mse_bits)
    centroids_gpu = centroids.cuda().half()  # [2^mse_bits]

    # Queries
    queries = torch.randn(B, H, S_q, d, device="cuda", dtype=torch.float16)

    # Build fake packed key indices (random valid indices)
    n_mse_levels = 1 << mse_bits
    raw_key_idx = torch.randint(
        0, n_mse_levels, (B * H * S_k, d), dtype=torch.int32, device="cuda"
    )
    packed_key_indices = pack_bits(raw_key_idx, mse_bits).reshape(B, H, S_k, -1)

    # Build fake packed QJL signs (0 or 1, packed as 1-bit)
    raw_sign_idx = torch.randint(0, 2, (B * H * S_k, m), dtype=torch.int32, device="cuda")
    packed_qjl_signs = pack_bits(raw_sign_idx, 1).reshape(B, H, S_k, -1)

    residual_norms = torch.rand(B, H, S_k, device="cuda", dtype=torch.float16)
    qjl_matrix = torch.randn(m, d, device="cuda", dtype=torch.float16)
    coeff = math.sqrt(math.pi / 2) / m

    # ------ PyTorch reference ------
    # Dequantise keys by looking up centroids
    k_mse = centroids_gpu[raw_key_idx.long()].reshape(B, H, S_k, d).float()
    # Signs: 0 -> -1, 1 -> +1
    signs = (raw_sign_idx.float() * 2.0 - 1.0).reshape(B, H, S_k, m)

    term1_ref = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
    q_proj_ref = torch.matmul(queries.float(), qjl_matrix.float().T)
    term2_ref = torch.matmul(q_proj_ref, signs.transpose(-2, -1))
    term2_ref = term2_ref * coeff * residual_norms.float().unsqueeze(-2)
    ref = term1_ref + term2_ref  # [B, H, S_q, S_k]

    # ------ Triton kernel ------
    # centroids_per_head: [H, 2^mse_bits] — same codebook for all heads here
    centroids_per_head = centroids_gpu.unsqueeze(0).expand(H, -1).contiguous()

    out = asymmetric_attention_scores(
        queries,
        packed_key_indices,
        packed_qjl_signs,
        residual_norms,
        centroids_per_head,
        qjl_matrix,
        b=b, d=d, m=m, coeff=coeff,
    )

    assert out.shape == (B, H, S_q, S_k), f"Shape mismatch: {out.shape}"
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2,
        msg=f"Attention mismatch for b={b}")
```

- [ ] **Step 2: Run to confirm ImportError**

```bash
uv run pytest tests/test_kernels.py::test_asymmetric_attention_matches_pytorch -v -m "" 2>&1 | head -20
```

- [ ] **Step 3: Implement attention.py**

Create `src/turboquant_pytorch/kernels/attention.py`:

```python
"""
Fused asymmetric attention score kernel.

Computes without materialising fp16 keys:
  score[b,h,q,k] = sum_d(q[d] * centroid[key_idx[k,d]])
                 + coeff * r_norm[k] * sum_m(Sq[m] * (2*sign[k,m] - 1))

where Sq = S @ q is precomputed by the Python wrapper via cuBLAS.

Grid: (B*H, S_q, ceil(S_k / BLOCK_SK))
Each program handles one (batch*head, query, key-block) triple.
"""

import math
import torch
import triton
import triton.language as tl
from .bit_ops import unpack_bits


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SK": 16}, num_warps=4),
        triton.Config({"BLOCK_SK": 32}, num_warps=4),
        triton.Config({"BLOCK_SK": 32}, num_warps=8),
        triton.Config({"BLOCK_SK": 64}, num_warps=8),
    ],
    key=["S_k", "d", "m", "mse_bits"],
)
@triton.jit
def _fused_attention_kernel(
    # inputs (all contiguous, row-major)
    q_ptr,          # float32 [B*H*S_q, d]  — queries
    key_idx_ptr,    # int32   [B*H*S_k, d]  — pre-unpacked MSE indices
    sign_ptr,       # int32   [B*H*S_k, m]  — pre-unpacked QJL signs (0 or 1)
    sq_ptr,         # float32 [B*H*S_q, m]  — precomputed S @ q
    r_norm_ptr,     # float32 [B*H, S_k]    — residual norms
    centroids_ptr,  # float32 [B*H, n_mse_levels] — codebook (broadcast over B)
    out_ptr,        # float32 [B*H, S_q, S_k]
    BH, H, S_q, S_k, d, m, n_mse_levels, coeff,
    mse_bits: tl.constexpr,
    BLOCK_SK: tl.constexpr,
):
    bh_idx   = tl.program_id(0)   # flattened batch*head index
    q_idx    = tl.program_id(1)   # query position
    sk_block = tl.program_id(2)   # key block index

    sk_start   = sk_block * BLOCK_SK
    sk_offsets = sk_start + tl.arange(0, BLOCK_SK)
    sk_mask    = sk_offsets < S_k

    # Pointer bases for this (bh, q) query and this bh key block
    q_row   = bh_idx * S_q + q_idx          # row in q_ptr [B*H*S_q, d]
    sq_row  = bh_idx * S_q + q_idx          # row in sq_ptr [B*H*S_q, m]
    key_row_base = bh_idx * S_k             # first key row for this bh
    cent_base    = bh_idx * n_mse_levels    # centroid base for this bh

    term1 = tl.zeros([BLOCK_SK], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_SK], dtype=tl.float32)

    # ---------------------------------------------------------------
    # Term 1: sum_d  q[d] * centroid[ key_idx[sk, d] ]
    #
    # For each d-coordinate:
    #   1. Load scalar q[d]
    #   2. Gather key_idx[sk, d] for all sk in block  -> [BLOCK_SK] int32
    #   3. Gather centroid[key_idx[sk,d]]             -> [BLOCK_SK] float32
    #   4. Accumulate q[d] * centroid[...]
    # ---------------------------------------------------------------
    for d_idx in range(d):
        q_val = tl.load(q_ptr + q_row * d + d_idx)   # scalar

        # Gather key indices at this d coordinate for all sk in block
        key_ptrs = key_idx_ptr + (key_row_base + sk_offsets) * d + d_idx
        key_idx_vals = tl.load(key_ptrs, mask=sk_mask, other=0)  # [BLOCK_SK] int32

        # Gather centroid values
        cent_ptrs = centroids_ptr + cent_base + key_idx_vals
        cent_vals = tl.load(cent_ptrs, mask=sk_mask, other=0.0)  # [BLOCK_SK]

        term1 += q_val * cent_vals

    # ---------------------------------------------------------------
    # Term 2: sum_m  Sq[m] * (2 * sign[sk, m] - 1)
    #
    # sign values are stored as 0 (negative) or 1 (positive); convert to -1/+1.
    # ---------------------------------------------------------------
    for m_idx in range(m):
        sq_val = tl.load(sq_ptr + sq_row * m + m_idx)   # scalar

        sign_ptrs = sign_ptr + (key_row_base + sk_offsets) * m + m_idx
        sign_vals = tl.load(sign_ptrs, mask=sk_mask, other=0).to(tl.float32)
        sign_pm1 = sign_vals * 2.0 - 1.0   # map 0->-1, 1->+1

        term2 += sq_val * sign_pm1

    # ---------------------------------------------------------------
    # Combine: score = term1 + coeff * r_norm * term2
    # ---------------------------------------------------------------
    r_norm_ptrs = r_norm_ptr + key_row_base + sk_offsets
    r_norms = tl.load(r_norm_ptrs, mask=sk_mask, other=0.0)

    scores = term1 + coeff * r_norms * term2

    out_row = bh_idx * S_q * S_k + q_idx * S_k
    tl.store(out_ptr + out_row + sk_offsets, scores, mask=sk_mask)


def asymmetric_attention_scores(
    queries: torch.Tensor,            # float16 [B, H, S_q, d]
    packed_key_indices: torch.Tensor, # uint8   [B, H, S_k, ceil(d*(b-1)/8)]
    packed_qjl_signs: torch.Tensor,   # uint8   [B, H, S_k, ceil(m/8)]
    residual_norms: torch.Tensor,     # float16 [B, H, S_k]
    centroids: torch.Tensor,          # float16 [H, 2^(b-1)]
    qjl_matrix: torch.Tensor,         # float16 [m, d]
    b: int,
    d: int,
    m: int,
    coeff: float,
) -> torch.Tensor:
    """
    Compute asymmetric attention scores without materialising fp16 keys.

    Returns float32 [B, H, S_q, S_k].
    """
    B, H, S_q, _ = queries.shape
    S_k = packed_key_indices.shape[2]
    mse_bits = b - 1
    n_mse_levels = 1 << mse_bits

    # ------------------------------------------------------------------
    # Step 1: Unpack bit-packed indices to int32 tensors.
    # This does NOT materialise fp16 keys — just integer indices.
    # ------------------------------------------------------------------
    key_indices = unpack_bits(
        packed_key_indices.reshape(B * H * S_k, -1), mse_bits, d
    )  # int32 [B*H*S_k, d]

    sign_indices = unpack_bits(
        packed_qjl_signs.reshape(B * H * S_k, -1), 1, m
    )  # int32 [B*H*S_k, m]  — values 0 or 1

    # ------------------------------------------------------------------
    # Step 2: Precompute S @ q for all queries via cuBLAS (once per query,
    # reused for all S_k keys).
    # queries: [B, H, S_q, d], qjl_matrix: [m, d] -> [B, H, S_q, m]
    # ------------------------------------------------------------------
    Sq = torch.matmul(
        queries.float().reshape(B * H * S_q, d),
        qjl_matrix.float().T,
    )  # [B*H*S_q, m]

    # ------------------------------------------------------------------
    # Step 3: Expand centroids to [B*H, n_mse_levels].
    # Centroids are per-head; broadcast same codebook across batch dim.
    # ------------------------------------------------------------------
    cent_expanded = (
        centroids.float()
        .unsqueeze(0)
        .expand(B, H, n_mse_levels)
        .reshape(B * H, n_mse_levels)
        .contiguous()
    )

    # ------------------------------------------------------------------
    # Step 4: Fused Triton kernel
    # ------------------------------------------------------------------
    q_flat       = queries.float().reshape(B * H * S_q, d).contiguous()
    r_norms_flat = residual_norms.float().reshape(B * H, S_k).contiguous()
    out          = torch.zeros(B * H, S_q, S_k, dtype=torch.float32, device=queries.device)

    grid = lambda meta: (B * H, S_q, triton.cdiv(S_k, meta["BLOCK_SK"]))
    _fused_attention_kernel[grid](
        q_flat,
        key_indices.contiguous(),
        sign_indices.contiguous(),
        Sq.contiguous(),
        r_norms_flat,
        cent_expanded,
        out,
        B * H, H, S_q, S_k, d, m, n_mse_levels, coeff,
        mse_bits,
    )

    return out.reshape(B, H, S_q, S_k)
```

- [ ] **Step 4: Run attention tests**

```bash
uv run pytest tests/test_kernels.py -v -m "" -k "attention"
```

Expected: PASS for all b values within atol=1e-2.

- [ ] **Step 5: Run full kernel test suite**

```bash
uv run pytest tests/test_kernels.py -v -m ""
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant_pytorch/kernels/attention.py tests/test_kernels.py
git commit -m "feat: add asymmetric_attention_scores Triton kernel"
```

---

## Task 5: Update `TurboQuantCompressorV2` with Triton backend

**Files:**
- Modify: `src/turboquant_pytorch/compressors.py`
- Create: `tests/test_compressors.py`

### What changes

Currently `compress()` stores fp16 `k_mse` (dequantised). With Triton it stores:
- `packed_key_indices`: `uint8[B, H, S, ceil(d*(b-1)/8)]`
- `packed_qjl_signs`: `uint8[B, H, S, ceil(d/8)]` (m==d in current impl)
- `residual_norm`: `float16[B, H, S]`
- `vec_norms`: `float16[B, H, S]`
- `shape`: tuple

`asymmetric_attention_scores(queries, compressed)` signature is identical. Internally it dispatches to the Triton kernel.

- [ ] **Step 1: Write failing V2 compressor equivalence tests**

Create `tests/test_compressors.py`:

```python
import math
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_compressor_v2_attention_scores_agree(b, d):
    """
    Triton and PyTorch backends of TurboQuantCompressorV2 must produce
    attention scores within atol=1e-2 on the same input.
    """
    from turboquant_pytorch.compressors import TurboQuantCompressorV2

    torch.manual_seed(0)
    B, H, S_k, S_q, seed = 1, 2, 16, 4, 42
    states  = torch.randn(B, H, S_k, d, device="cuda")
    queries = torch.randn(B, H, S_q, d, device="cuda", dtype=torch.float16)

    py_comp  = TurboQuantCompressorV2(d, b, seed, device="cuda", use_triton=False)
    tri_comp = TurboQuantCompressorV2(d, b, seed, device="cuda", use_triton=True)

    py_c  = py_comp.compress(states)
    tri_c = tri_comp.compress(states)

    py_scores  = py_comp.asymmetric_attention_scores(queries, py_c).float()
    tri_scores = tri_comp.asymmetric_attention_scores(queries, tri_c).float()

    assert py_scores.shape  == (B, H, S_q, S_k)
    assert tri_scores.shape == (B, H, S_q, S_k)
    torch.testing.assert_close(
        py_scores, tri_scores, atol=1e-2, rtol=1e-2,
        msg=f"V2 attention score mismatch at b={b}, d={d}",
    )


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
def test_compressor_v2_use_triton_flag_respects_cuda_absence(b, monkeypatch):
    """When torch.cuda.is_available() returns False, use_triton must fall back silently."""
    import turboquant_pytorch.compressors as comp_mod
    import warnings

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = comp_mod.TurboQuantCompressorV2(64, b, seed=0, device="cpu", use_triton=True)
        assert c.use_triton is False
        assert any("falling back" in str(warning.message).lower() for warning in w), \
            "Expected a fallback warning"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_compressors.py -v -m "" 2>&1 | head -20
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'use_triton'`

- [ ] **Step 3: Rewrite TurboQuantCompressorV2 in compressors.py**

Replace the existing `TurboQuantCompressorV2` class entirely. Keep `TurboQuantCompressorMSE` untouched for now.

```python
"""
TurboQuant KV cache v2: Asymmetric attention.
(See original module docstring for algorithm details.)
"""

import warnings
import math
import torch

try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _check_triton(use_triton: bool, device: str) -> bool:
    """Return resolved use_triton, emitting warnings if falling back."""
    if not use_triton:
        return False
    if not _TRITON_AVAILABLE:
        warnings.warn(
            "triton is not importable — falling back to PyTorch backend.",
            RuntimeWarning, stacklevel=3,
        )
        return False
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available — falling back to PyTorch backend.",
            RuntimeWarning, stacklevel=3,
        )
        return False
    return True


class TurboQuantCompressorV2:
    """
    Compressor for transformer keys.
    Stores bit-packed MSE indices + QJL signs (Triton backend) or fp16
    reconstructions (PyTorch backend). Same public API either way.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        device: str = "cpu",
        use_triton: bool = True,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device
        self.use_triton = _check_triton(use_triton, device)

        # Rotation matrix (Haar-distributed orthogonal)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        self.PiT = self.Pi.T.contiguous()

        # Lloyd-Max codebook for MSE quantizer
        self.centroids = self._solve_codebook(head_dim, self.mse_bits).to(device)

        # Boundaries for Triton quantize kernel (2^mse_bits - 1 values)
        c = self.centroids.cpu().tolist()
        self.boundaries = torch.tensor(
            [(c[i] + c[i + 1]) / 2.0 for i in range(len(c) - 1)],
            dtype=torch.float32,
            device=device,
        )

        # QJL projection matrix S: (head_dim, head_dim)
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

        # Correction coefficient: sqrt(pi/2) / m,  m == head_dim
        self._coeff = math.sqrt(math.pi / 2) / head_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        from scipy import integrate
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (
                (1.0 / math.sqrt(2 * math.pi * sigma ** 2))
                * math.exp(-x * x / (2 * sigma ** 2))
            )

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0
                          for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_centroids[i] - centroids[i])
                   for i in range(n_levels)) < 1e-10:
                break
            centroids = new_centroids
        return torch.tensor(centroids, dtype=torch.float32)

    def _mse_reconstruct(self, rotated: torch.Tensor, vec_norms: torch.Tensor) -> torch.Tensor:
        """Compute MSE reconstruction in original space (shared by both backends)."""
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).long()
        reconstructed_rotated = self.centroids[indices]
        return (reconstructed_rotated @ self.Pi) * vec_norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress states: (B, H, S, D) -> compressed dict.

        Triton backend dict keys:
            packed_key_indices: uint8[B, H, S, ceil(D*(b-1)/8)]
            packed_qjl_signs:   uint8[B, H, S, ceil(D/8)]
            residual_norm:      float16[B, H, S]
            vec_norms:          float16[B, H, S]
            shape:              (B, H, S, D)

        PyTorch backend dict keys (unchanged from original):
            k_mse:         float16[B, H, S, D]
            qjl_signs:     int8[B, H, S, D]
            residual_norm: float16[B, H, S]
            shape:         (B, H, S, D)
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()     # [N, D]  where N = B*H*S

        vec_norms = torch.norm(flat, dim=-1, keepdim=True)    # [N, 1]
        flat_norm = flat / (vec_norms + 1e-8)
        rotated   = flat_norm @ self.Pi.T                      # [N, D]

        if self.use_triton:
            from turboquant_pytorch.kernels.quantize import lloyd_max_quantize
            from turboquant_pytorch.kernels.bit_ops import pack_bits

            # Pack MSE indices using b_eff = mse_bits
            packed_key_indices = lloyd_max_quantize(
                rotated.contiguous(), self.boundaries, self.mse_bits
            ).reshape(B, H, S, -1)

            # Compute residual for QJL
            k_mse = self._mse_reconstruct(rotated, vec_norms)  # [N, D]
            residual = flat - k_mse
            residual_norm = torch.norm(residual, dim=-1)       # [N]

            # Project residual through S, take sign, pack as 1-bit
            projected = residual @ self.S.T                    # [N, D]
            sign_indices = (projected >= 0).to(torch.int32)   # 0 or 1
            packed_qjl_signs = pack_bits(sign_indices, 1).reshape(B, H, S, -1)

            return {
                "packed_key_indices": packed_key_indices,
                "packed_qjl_signs":   packed_qjl_signs,
                "residual_norm":      residual_norm.to(torch.float16).reshape(B, H, S),
                "vec_norms":          vec_norms.squeeze(-1).to(torch.float16).reshape(B, H, S),
                "shape":              (B, H, S, D),
            }
        else:
            # Original PyTorch path — unchanged
            diffs = rotated.unsqueeze(-1) - self.centroids
            indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
            reconstructed_rotated = self.centroids[indices.long()]
            k_mse = (reconstructed_rotated @ self.Pi) * vec_norms
            residual = flat - k_mse
            residual_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            return {
                "k_mse":         k_mse.to(torch.float16).reshape(B, H, S, D),
                "qjl_signs":     signs.reshape(B, H, S, D),
                "residual_norm": residual_norm.to(torch.float16).reshape(B, H, S),
                "shape":         (B, H, S, D),
            }

    @torch.no_grad()
    def asymmetric_attention_scores(
        self,
        queries: torch.Tensor,
        compressed: dict,
    ) -> torch.Tensor:
        """
        Compute attention scores <Q, K> directly from compressed K.
        queries: (B, H, S_q, D)  ->  scores: (B, H, S_q, S_k)
        """
        if self.use_triton:
            from turboquant_pytorch.kernels.attention import asymmetric_attention_scores

            B, H, S_k, D = compressed["shape"]
            H_actual = queries.shape[1]
            # Same codebook for all heads (one codebook per compressor instance)
            centroids_per_head = (
                self.centroids.half()
                .unsqueeze(0)
                .expand(H_actual, -1)
                .contiguous()
            )
            return asymmetric_attention_scores(
                queries.half(),
                compressed["packed_key_indices"],
                compressed["packed_qjl_signs"],
                compressed["residual_norm"],
                centroids_per_head,
                self.S.half(),
                b=self.bits,
                d=D,
                m=self.head_dim,
                coeff=self._coeff,
            )
        else:
            # Original PyTorch path — unchanged
            k_mse  = compressed["k_mse"].float()
            signs  = compressed["qjl_signs"].float()
            r_norm = compressed["residual_norm"].float()
            term1  = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
            q_proj = torch.matmul(queries.float(), self.S.T)
            qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))
            m = self.S.shape[0]
            scale  = math.sqrt(math.pi / 2) / m
            term2  = scale * qjl_ip * r_norm.unsqueeze(-2)
            return term1 + term2
```

- [ ] **Step 4: Run V2 compressor tests**

```bash
uv run pytest tests/test_compressors.py -v -m ""
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turboquant_pytorch/compressors.py tests/test_compressors.py
git commit -m "feat: add use_triton backend to TurboQuantCompressorV2"
```

---

## Task 6: Update `TurboQuantCompressorMSE` with Triton backend

**Files:**
- Modify: `src/turboquant_pytorch/compressors.py`
- Modify: `tests/test_compressors.py`

### What changes

`TurboQuantCompressorMSE.compress()` currently returns `"indices"` (uint8, full-byte per index). With Triton it returns `"packed_indices"` (truly bit-packed). `decompress()` dispatches to `lloyd_max_dequantize`.

- [ ] **Step 1: Append failing MSE compressor tests**

Append to `tests/test_compressors.py`:

```python
@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_compressor_mse_decompress_agrees(b, d):
    """
    Triton and PyTorch backends of TurboQuantCompressorMSE must produce
    identical reconstructions (same codebook, same rotation, same input).
    """
    from turboquant_pytorch.compressors import TurboQuantCompressorMSE

    torch.manual_seed(1)
    B, H, S, seed = 1, 2, 16, 42
    states = torch.randn(B, H, S, d, device="cuda")

    py_comp  = TurboQuantCompressorMSE(d, b, seed, device="cuda", use_triton=False)
    tri_comp = TurboQuantCompressorMSE(d, b, seed, device="cuda", use_triton=True)

    py_c  = py_comp.compress(states)
    tri_c = tri_comp.compress(states)

    py_out  = py_comp.decompress(py_c).float()
    tri_out = tri_comp.decompress(tri_c).float()

    assert py_out.shape  == (B, H, S, d)
    assert tri_out.shape == (B, H, S, d)
    torch.testing.assert_close(
        py_out, tri_out, atol=1e-2, rtol=1e-2,
        msg=f"MSE decompress mismatch at b={b}, d={d}",
    )


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
def test_compressor_mse_use_triton_fallback(b, monkeypatch):
    import turboquant_pytorch.compressors as comp_mod
    import warnings

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = comp_mod.TurboQuantCompressorMSE(64, b, seed=0, device="cpu", use_triton=True)
        assert c.use_triton is False
        assert any("falling back" in str(x.message).lower() for x in w)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_compressors.py -v -m "" -k "mse" 2>&1 | head -20
```

- [ ] **Step 3: Rewrite TurboQuantCompressorMSE in compressors.py**

Replace the existing `TurboQuantCompressorMSE` class. Note: `_check_triton` is already defined at module level from Task 5.

```python
class TurboQuantCompressorMSE:
    """
    Compressor for transformer values (MSE-only, no QJL).
    Triton backend stores bit-packed indices; PyTorch backend stores
    uint8 indices (one byte per index, unchanged from original).
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        device: str = "cpu",
        use_triton: bool = True,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.use_triton = _check_triton(use_triton, device)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)

        self.centroids = self._solve_codebook(head_dim, bits).to(device)

        # Boundaries for Triton quantize kernel (2^bits - 1 values)
        c = self.centroids.cpu().tolist()
        self.boundaries = torch.tensor(
            [(c[i] + c[i + 1]) / 2.0 for i in range(len(c) - 1)],
            dtype=torch.float32,
            device=device,
        )

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        from scipy import integrate
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (
                (1.0 / math.sqrt(2 * math.pi * sigma ** 2))
                * math.exp(-x * x / (2 * sigma ** 2))
            )

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0
                          for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_c = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_c
        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress states: (B, H, S, D).

        Triton backend:  {"packed_indices": uint8[B,H,S,ceil(D*b/8)],
                          "vec_norms": float16[B,H,S], "shape": ...}
        PyTorch backend: {"indices": uint8[N,D], "vec_norms": float16[N],
                          "shape": ...}   (unchanged from original)
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()

        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated   = flat_norm @ self.Pi.T

        if self.use_triton:
            from turboquant_pytorch.kernels.quantize import lloyd_max_quantize

            packed_indices = lloyd_max_quantize(
                rotated.contiguous(), self.boundaries, self.bits
            ).reshape(B, H, S, -1)

            return {
                "packed_indices": packed_indices,
                "vec_norms":      vec_norms.squeeze(-1).to(torch.float16).reshape(B, H, S),
                "shape":          (B, H, S, D),
            }
        else:
            diffs   = rotated.unsqueeze(-1) - self.centroids
            indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
            return {
                "indices":   indices,
                "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
                "shape":     (B, H, S, D),
            }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Reconstruct values from compressed representation."""
        B, H, S, D = compressed["shape"]
        N = B * H * S

        if self.use_triton:
            from turboquant_pytorch.kernels.quantize import lloyd_max_dequantize

            packed = compressed["packed_indices"].reshape(N, -1)
            reconstructed = lloyd_max_dequantize(
                packed, self.centroids.half(), self.bits, D
            ).float()  # [N, D]
        else:
            indices       = compressed["indices"].reshape(N, D).long()
            reconstructed = self.centroids[indices]  # [N, D]

        # Unrotate and rescale
        reconstructed = reconstructed @ self.Pi
        vec_norms = compressed["vec_norms"].float().reshape(N, 1)
        return (reconstructed * vec_norms).reshape(B, H, S, D)
```

- [ ] **Step 4: Run all compressor tests**

```bash
uv run pytest tests/test_compressors.py -v -m ""
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turboquant_pytorch/compressors.py tests/test_compressors.py
git commit -m "feat: add use_triton backend to TurboQuantCompressorMSE"
```

---

## Task 7: Memory regression tests

**Files:**
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write memory regression tests**

Create `tests/test_memory.py`:

```python
"""
Verify that Triton bit-packed storage is strictly smaller than the fp16
baseline for all (b, d) configurations. Catches any accidental regression
to storing full fp16 tensors.
"""

import math
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

B, H, S = 1, 4, 64  # fixed tensor shape for all tests


def fp16_key_bytes(d: int) -> int:
    """Bytes for fp16 MSE reconstruction + int8 signs (old storage)."""
    return B * H * S * d * 2 + B * H * S * d * 1  # fp16 k_mse + int8 signs


def fp16_value_bytes(d: int) -> int:
    """Bytes for fp16 value reconstruction (old storage)."""
    return B * H * S * d * 2


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_triton_key_storage_smaller_than_fp16(b, d):
    from turboquant_pytorch.compressors import TurboQuantCompressorV2

    states = torch.randn(B, H, S, d, device="cuda")
    comp   = TurboQuantCompressorV2(d, b, seed=0, device="cuda", use_triton=True)
    c      = comp.compress(states)

    actual = (
        c["packed_key_indices"].nbytes
        + c["packed_qjl_signs"].nbytes
        + c["residual_norm"].nbytes
    )
    baseline = fp16_key_bytes(d)

    assert actual < baseline, (
        f"Triton key storage ({actual}B) is not smaller than fp16 "
        f"baseline ({baseline}B) at b={b}, d={d}"
    )


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_triton_key_storage_meets_minimum_compression_ratio(b, d):
    """
    At b=3, d=128 the spec promises ~5.7x compression over fp16.
    Enforce a minimum of 3x for all (b, d) to catch partial regressions.
    """
    from turboquant_pytorch.compressors import TurboQuantCompressorV2

    states = torch.randn(B, H, S, d, device="cuda")
    comp   = TurboQuantCompressorV2(d, b, seed=0, device="cuda", use_triton=True)
    c      = comp.compress(states)

    actual   = c["packed_key_indices"].nbytes + c["packed_qjl_signs"].nbytes + c["residual_norm"].nbytes
    baseline = fp16_key_bytes(d)
    ratio    = baseline / actual

    assert ratio >= 3.0, (
        f"Compression ratio {ratio:.2f}x is below minimum 3x at b={b}, d={d}"
    )


@cuda_only
@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_triton_value_storage_smaller_than_fp16(b, d):
    from turboquant_pytorch.compressors import TurboQuantCompressorMSE

    states = torch.randn(B, H, S, d, device="cuda")
    comp   = TurboQuantCompressorMSE(d, b, seed=0, device="cuda", use_triton=True)
    c      = comp.compress(states)

    actual   = c["packed_indices"].nbytes + c["vec_norms"].nbytes
    baseline = fp16_value_bytes(d)

    assert actual < baseline, (
        f"Triton value storage ({actual}B) not smaller than fp16 "
        f"({baseline}B) at b={b}, d={d}"
    )
```

- [ ] **Step 2: Run memory tests**

```bash
uv run pytest tests/test_memory.py -v -m ""
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_memory.py
git commit -m "test: add memory regression tests for Triton bit-packed storage"
```

---

## Task 8: Slow end-to-end variance comparison test

**Files:**
- Create: `tests/test_slow.py`

### What this test does

1. Loads `Qwen/Qwen2.5-3B-Instruct` with 4-bit BitsAndBytes quantization.
2. Runs a fixed needle-in-haystack prompt.
3. For each `(layer, head, bits)` combination, compresses and decompresses attention states through all three modes: baseline / PyTorch / Triton.
4. Computes variance of attention scores over the sequence dimension for each mode.
5. Prints a `rich` table and asserts Triton and PyTorch backends agree.

Note: the baseline attention scores come from the model's real softmax attention. The PyTorch and Triton scores come from the compressor with synthetic states of the correct shape (full hook-based KV extraction is out of scope; this tests backend agreement and quality relative to baseline variance).

- [ ] **Step 1: Create tests/test_slow.py**

```python
"""
End-to-end variance comparison: baseline model vs TurboQuant PyTorch vs Triton.
Run with: just test-slow
"""

import math
import pytest
import torch
from rich.console import Console
from rich.table import Table


@pytest.mark.slow
def test_attention_variance_comparison():
    """
    Loads Qwen2.5-3B-Instruct, runs a fixed needle-in-haystack prompt, and
    compares attention score variance across three modes for each (layer, head, bits).
    Prints a rich table and asserts that Triton and PyTorch backends agree within 0.1.

    Variance definition:
      For each (layer, head, bits): compute the variance of the attention score
      distribution over the S_k (key sequence) dimension, then average over
      batch and query positions.

    Columns:
      baseline_var:             variance of the uncompressed model's attention
      pytorch_var:              variance from TurboQuant PyTorch backend
      triton_var:               variance from TurboQuant Triton backend
      pytorch_vs_baseline_delta: |pytorch_var - baseline_var|
      triton_vs_baseline_delta:  |triton_var  - baseline_var|
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for slow test")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from turboquant_pytorch.compressors import TurboQuantCompressorV2

    MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
    BIT_WIDTHS = [2, 3, 4]
    SEED       = 42

    # Fixed needle-in-haystack prompt
    PROMPT = (
        "In a distant galaxy, the secret activation code is AURORA-7. "
        * 20
        + "What is the secret activation code? The answer is:"
    )

    console = Console()
    console.print(f"\n[bold]Loading {MODEL_ID}[/bold]")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        output_attentions=True,
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    console.print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions   # tuple of (B, H, S_q, S_k) per layer
    n_layers   = len(attentions)
    n_heads    = attentions[0].shape[1]
    d          = model.config.hidden_size // n_heads

    console.print(
        f"Model: {n_layers} layers, {n_heads} heads, head_dim={d}, "
        f"seq_len={attentions[0].shape[-1]}"
    )

    rows = []

    for b in BIT_WIDTHS:
        console.print(f"\n[cyan]Running b={b} bits...[/cyan]")

        for layer_idx in range(n_layers):
            attn = attentions[layer_idx]   # [B, H, S_q, S_k]
            B_model, H_model, S_q, S_k = attn.shape

            for head_idx in range(H_model):
                # Baseline variance: variance of the softmax attention distribution
                # over S_k, averaged over (B, S_q)
                baseline_scores = attn[:, head_idx, :, :]   # [B, S_q, S_k]
                baseline_var    = baseline_scores.float().var(dim=-1).mean().item()

                # Build compressors for this (layer, head, bits).
                # We use synthetic states of the correct shape since full KV
                # hook extraction is out of scope. This tests backend agreement
                # and quality relative to baseline variance magnitude.
                torch.manual_seed(SEED + layer_idx * 1000 + head_idx)
                states  = torch.randn(B_model, 1, S_k, d, device="cuda")
                queries = torch.randn(B_model, 1, S_q, d, device="cuda", dtype=torch.float16)

                py_comp  = TurboQuantCompressorV2(d, b, seed=SEED, device="cuda", use_triton=False)
                tri_comp = TurboQuantCompressorV2(d, b, seed=SEED, device="cuda", use_triton=True)

                py_c   = py_comp.compress(states)
                tri_c  = tri_comp.compress(states)

                py_scores  = py_comp.asymmetric_attention_scores(queries, py_c)
                tri_scores = tri_comp.asymmetric_attention_scores(queries, tri_c)

                # Variance over S_k, averaged over (B, H=1, S_q)
                pytorch_var = py_scores.float().var(dim=-1).mean().item()
                triton_var  = tri_scores.float().var(dim=-1).mean().item()

                rows.append({
                    "layer":          layer_idx,
                    "head":           head_idx,
                    "bits":           b,
                    "baseline_var":   baseline_var,
                    "pytorch_var":    pytorch_var,
                    "triton_var":     triton_var,
                    "pytorch_delta":  abs(pytorch_var  - baseline_var),
                    "triton_delta":   abs(triton_var   - baseline_var),
                })

    # ------------------------------------------------------------------
    # Print rich table
    # ------------------------------------------------------------------
    table = Table(title="TurboQuant Attention Variance Comparison", show_lines=False)
    table.add_column("layer",                    style="cyan",   justify="right")
    table.add_column("head",                     style="cyan",   justify="right")
    table.add_column("bits",                     style="yellow", justify="right")
    table.add_column("baseline_var",                             justify="right")
    table.add_column("pytorch_var",                              justify="right")
    table.add_column("triton_var",                               justify="right")
    table.add_column("pytorch_vs_baseline_Δ",   style="green",  justify="right")
    table.add_column("triton_vs_baseline_Δ",    style="green",  justify="right")

    for r in rows:
        table.add_row(
            str(r["layer"]),
            str(r["head"]),
            str(r["bits"]),
            f"{r['baseline_var']:.6f}",
            f"{r['pytorch_var']:.6f}",
            f"{r['triton_var']:.6f}",
            f"{r['pytorch_delta']:.6f}",
            f"{r['triton_delta']:.6f}",
        )

    console.print(table)

    # ------------------------------------------------------------------
    # Assertion: PyTorch and Triton backends must agree on variance
    # ------------------------------------------------------------------
    failures = [
        r for r in rows
        if abs(r["pytorch_var"] - r["triton_var"]) >= 0.1
    ]
    if failures:
        msg = "\n".join(
            f"  layer={r['layer']} head={r['head']} bits={r['bits']}: "
            f"pytorch={r['pytorch_var']:.6f} triton={r['triton_var']:.6f} "
            f"diff={abs(r['pytorch_var']-r['triton_var']):.6f}"
            for r in failures[:5]
        )
        pytest.fail(
            f"{len(failures)} (layer, head, bits) combinations show backend "
            f"variance disagreement >= 0.1:\n{msg}"
        )
```

- [ ] **Step 2: Verify slow test is excluded from default run**

```bash
uv run pytest --collect-only 2>&1 | grep "test_slow"
```

Expected: no test_slow entries collected.

- [ ] **Step 3: Run the slow test (downloads model on first run — ~6GB)**

```bash
just test-slow
```

Expected: model loads, runs forward pass, prints the rich table, assertions pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_slow.py
git commit -m "test: add slow end-to-end variance comparison with rich table"
```

---

## Task 9: Full test suite verification and lint

- [ ] **Step 1: Run full default test suite**

```bash
just test
```

Expected: all non-slow tests PASS (kernel + compressor + memory tests). Slow tests excluded.

- [ ] **Step 2: Run lint**

```bash
just lint
```

If lint fails, fix with:

```bash
just lint-fix
git add -u
git commit -m "chore: lint fixes"
```

- [ ] **Step 3: Run typecheck**

```bash
just typecheck
```

Fix any type errors. Common ones to expect:
- `Optional` imports needed for `use_triton` defaults
- Return type annotations on `compress()`/`decompress()` methods

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "chore: typecheck and lint cleanup for Triton kernel implementation"
```

---

## Appendix: Common failure modes and fixes

**`tl.constexpr` errors:** If Triton complains that a value used in `tl.static_range` is not constexpr, ensure `b`, `b_eff`, `mse_bits` are declared as `tl.constexpr` in the kernel signature.

**Gather load shapes:** `tl.load(ptr + vector_offsets)` performs a gather. If `vector_offsets` is out of bounds, the `mask=` and `other=` parameters prevent incorrect reads. Always pass both.

**Autotune cache:** Triton caches autotune results in `~/.triton/cache`. If a kernel is modified, delete the cache or the old config may be used: `rm -rf ~/.triton/cache`.

**Mixed fp16/fp32:** The attention kernel uses float32 accumulators internally. Queries are cast to float32 inside the wrapper. The output is float32. Assertions use `atol=1e-2` to account for fp16 rounding in intermediate computations.

**`just test` coverage warning:** The coverage config tracks `transformers_turboquant` (the CLI package), not `turboquant_pytorch`. The new kernel tests will show 0% coverage for `turboquant_pytorch` unless `[tool.coverage.run] source` is updated to include it. This is a nice-to-have fix, not a blocker.
