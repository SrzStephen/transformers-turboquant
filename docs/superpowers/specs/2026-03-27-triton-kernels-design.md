# Triton GPU Kernels for TurboQuant — Design Spec

**Date:** 2026-03-27
**Project:** `turboquant_pytorch`
**Status:** Approved

---

## Overview

Replace the pure-PyTorch internals of `TurboQuantCompressorV2` and `TurboQuantCompressorMSE` with custom Triton GPU kernels. The public API of both classes is unchanged. The primary goals are:

1. **Reduce memory bandwidth** by storing bit-packed integer indices instead of fp16 dequantised tensors.
2. **Fuse compute** in the asymmetric attention hot path to avoid materialising intermediate fp16 key tensors.
3. **Portable performance** via Triton's `@triton.autotune` — no architecture-specific tuning required at development time.

Approach: **modular kernels with strategic fusion** (Option B). Rotation (`torch.matmul` → cuBLAS) is left untouched. Custom Triton is written only where PyTorch cannot express the operation (bit-packing) or where fusion across multiple logical steps yields meaningful bandwidth savings (fused attention scoring).

---

## Architecture

### New Module Layout

```
src/turboquant_pytorch/
├── kernels/
│   ├── __init__.py          # exports all kernel functions
│   ├── bit_ops.py           # pack_bits / unpack_bits
│   ├── quantize.py          # lloyd_max_quantize / lloyd_max_dequantize
│   └── attention.py         # asymmetric_attention_scores
├── compressors.py           # modified: same API, Triton backend
├── turboquant.py            # modified: LloydMaxCodebook uses Triton kernels
└── lloyd_max.py             # unchanged
```

### Fallback Flag

Both compressor classes and `LloydMaxCodebook` accept `use_triton: bool = True`. When `False` (or when CUDA is unavailable), they fall back to the existing PyTorch path with no behaviour change. This enables direct backend comparison in tests.

---

## Storage Format Change

This is the most significant change. Currently `TurboQuantCompressorV2.compress()` dequantises immediately and stores fp16 reconstructions. The new format stores packed integer indices and dequantises on-the-fly inside the attention kernel.

### Per-vector layout (keys, `b`-bit total quantisation)

| Field | Before | After |
|---|---|---|
| MSE reconstruction | `d × 16` bits (fp16 tensor) | `d × (b-1)` bits (packed `uint8`) |
| QJL signs | `d × 8` bits (int8 tensor) | `d × 1` bit (packed `uint8`) |
| Residual norm | `float16` scalar | `float16` scalar — unchanged |
| Vector norm | `float16` scalar | `float16` scalar — unchanged |

### Per-vector layout (values, `b`-bit MSE only)

| Field | Before | After |
|---|---|---|
| MSE reconstruction | `d × 16` bits (fp16 tensor) | `d × b` bits (packed `uint8`) |
| Vector norm | `float16` scalar | `float16` scalar — unchanged |

### Example savings at `b=3, d=128`

| Mode | Bytes per key vector |
|---|---|
| Before | 256 (fp16 MSE) + 128 (int8 signs) + 4 = 388 |
| After | 48 (packed MSE) + 16 (packed signs) + 4 = 68 |
| Reduction | **~5.7×** |

Codebook centroids (`float16[2^(b-1)]` for keys, `float16[2^b]` for values) are small and stored as GPU tensors, one per compressor instance. They are loaded into shared memory inside the kernels.

---

## Kernel Designs

All kernels are decorated with `@triton.autotune` sweeping over `BLOCK_D`, `BLOCK_S`, and `num_warps`. The winning config is cached by Triton after first run.

### 1. `pack_bits_kernel` — `kernels/bit_ops.py`

- **Input:** `int32[N, d]` indices, scalar `b` (bits per index)
- **Output:** `uint8[N, ceil(d*b/8)]`
- Each program instance handles one row. Reads `d` indices, shifts and ORs them into packed bytes.
- Autotuned on `BLOCK_D`.

### 2. `unpack_bits_kernel` — `kernels/bit_ops.py`

- **Input:** `uint8[N, ceil(d*b/8)]`, scalar `b`, scalar `d`
- **Output:** `int32[N, d]`
- Masks with `(1 << b) - 1` to isolate each index.
- Autotuned on `BLOCK_D`.

### 3. `lloyd_max_quantize_kernel` — `kernels/quantize.py`

- **Input:** rotated `float32[N, d]`, codebook boundaries `float32[2^b_eff - 1]`, scalar `b_eff`
- **Output:** packed `uint8[N, ceil(d*b_eff/8)]` (fused with packing — no round-trip to HBM for raw indices)
- **Effective bit-width `b_eff`:** this kernel is called with `b_eff = b-1` for the key MSE path (reserving 1 bit for QJL), and `b_eff = b` for the value MSE path. Callers are responsible for passing the correct codebook and output buffer sized for `b_eff`.
- A `2^b_eff`-level codebook has exactly `2^b_eff - 1` interior boundaries (midpoints between adjacent centroids) — e.g. 255 boundaries for 8-bit. Loads boundaries into shared memory (max 255 entries at 8-bit). Each thread performs a binary search over shared memory for its coordinate.
- **Packing strategy:** each output byte is owned by exactly one thread. A thread processing coordinate `i` writes to byte `i * b_eff // 8` at bit offset `(i * b_eff) % 8`. Since threads are assigned to non-overlapping bit ranges within a byte boundary, no sub-byte atomics are needed — threads owning the same output byte are serialised by assigning consecutive coordinates to the same warp lane, then combining with warp-level shuffle (`tl.sum` over the lane mask) before a single scalar store. This avoids race conditions entirely.
- Replaces `torch.searchsorted` + separate pack step.
- Autotuned on `BLOCK_D`.

### 4. `lloyd_max_dequantize_kernel` — `kernels/quantize.py`

- **Input:** packed `uint8[N, ceil(d*b/8)]`, centroids `float16[2^b]`, scalar `b`, scalar `d`
- **Output:** `float16[N, d]`
- Unpacks indices inline, looks up centroid values. Used for value decompression only — keys are dequantised inside the attention kernel.
- Autotuned on `BLOCK_D`.

### 5. `asymmetric_attention_kernel` — `kernels/attention.py`

The performance-critical path. Never materialises a full fp16 key tensor.

In the current implementation `self.S` has shape `[head_dim, head_dim]`, so `m == d == head_dim`. The kernel is written with `m` as an explicit parameter to support future configurations where `m != d`, but all current callers pass `m == d`.

- **Inputs:**
  - `query: float16[B, H, S_q, d]`
  - `packed_key_indices: uint8[B, H, S_k, ceil(d*(b-1)/8)]`
  - `packed_qjl_signs: uint8[B, H, S_k, ceil(m/8)]`  ← note `m`, not `d`
  - `residual_norms: float16[B, H, S_k]`
  - `centroids: float16[H, 2^(b-1)]`
  - `qjl_matrix: float16[m, d]`
  - scalars: `b`, `d`, `m`, `coeff`
- **Output:** `float32[B, H, S_q, S_k]`

Each program instance handles a `(head, sequence_tile)` block. In a single pass:

1. Unpack MSE indices for the tile → look up centroids → accumulate `q · k_mse` dot product in registers (Term 1).
2. Unpack QJL signs for the tile → compute `(S@q) · sign(S@r)` correction in registers (Term 2).
3. Multiply Term 2 by `residual_norm[s, h] * coeff`.
4. Write `Term1 + Term2` to output.

Autotuned on `BLOCK_S` and `num_warps`.

---

## Class Integration

### `TurboQuantCompressorV2`

The existing public signatures are preserved exactly — callers pass the opaque dict from `compress()` straight into `asymmetric_attention_scores()` and never inspect its keys, so changing the dict's internal layout is transparent to call sites.

```python
class TurboQuantCompressorV2:
    def __init__(self, ..., use_triton: bool = True): ...

    def compress(self, states: Tensor) -> dict:
        # returns dict with keys:
        #   Triton backend:  "packed_key_indices" (uint8), "packed_qjl_signs" (uint8),
        #                    "residual_norm" (fp16), "vec_norms" (fp16), "shape"
        #   PyTorch backend: "k_mse" (fp16), "qjl_signs" (int8),
        #                    "residual_norm" (fp16), "shape"  ← unchanged from today

    def asymmetric_attention_scores(self, queries: Tensor, compressed: dict) -> Tensor:
        # same signature as today: (B, H, S_q, D), dict -> (B, H, S_q, S_k)
        # dispatches to asymmetric_attention_kernel (use_triton=True)
        # or original matmul/einsum path (use_triton=False)
```

### `TurboQuantCompressorMSE`

`compress()` already returns indices + vec_norms (not fp16 reconstructions). The only change is bit-packing those indices:

```python
class TurboQuantCompressorMSE:
    def __init__(self, ..., use_triton: bool = True): ...

    def compress(self, states: Tensor) -> dict:
        # returns dict with keys:
        #   Triton backend:  "packed_indices" (uint8, bit-packed), "vec_norms" (fp16), "shape"
        #   PyTorch backend: "indices" (uint8, full-byte), "vec_norms" (fp16), "shape"

    def decompress(self, compressed: dict) -> Tensor:
        # same signature as today
        # dispatches to lloyd_max_dequantize_kernel (use_triton=True)
        # or original centroid-lookup path (use_triton=False)
```

### `LloydMaxCodebook`

`quantize()` calls `lloyd_max_quantize_kernel` (returns packed uint8).
`dequantize()` calls `lloyd_max_dequantize_kernel`.
Both fall back to `torch.searchsorted` / centroid lookup when `use_triton=False`.

---

## Testing Strategy

### `tests/test_kernels.py` — unit tests, always run

- **Round-trip property:** `unpack(pack(indices, b), b) == indices` for all `b ∈ {2,3,4,5,6,7,8}` and random index tensors.
- **Quantize equivalence:** `lloyd_max_quantize_kernel` output matches `torch.searchsorted` reference exactly.
- **Attention equivalence:** `asymmetric_attention_kernel` output matches PyTorch einsum reference within `atol=1e-2` (fp16 accumulation drift).
- All GPU tests skipped automatically via `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.

### `tests/test_compressors.py` — integration tests, always run

- Construct `TurboQuantCompressorV2(use_triton=True)` and `(use_triton=False)`, compress identical input, compare `asymmetric_attention_scores()` within `atol=1e-2`.
- Same for `TurboQuantCompressorMSE.decompress()`.
- Parameterised over `b ∈ {2, 3, 4}` and `d ∈ {64, 128, 256}`.

### `tests/test_memory.py` — regression tests, always run

- Assert that Triton-backend compressed storage (in bytes) is strictly less than fp16 baseline for all `(b, d)` combinations.
- Catches any accidental reversion to storing full fp16 tensors.

### `tests/test_slow.py` — `@pytest.mark.slow`, excluded from default run

Loads Qwen2.5-3B-Instruct (4-bit BitsAndBytes, same as `validate.py`) and runs a fixed needle-in-haystack prompt through three modes:

1. **Baseline** — original model, no TurboQuant
2. **TurboQuant PyTorch** — `use_triton=False`
3. **TurboQuant Triton** — `use_triton=True`

Collects attention scores per `(layer, head)` at `b ∈ {2, 3, 4}` and prints a `rich` table.

**Column definitions:**
- `baseline_var` / `pytorch_var` / `triton_var`: variance of the attention score distribution over the sequence dimension (`S_k`) for that `(layer, head, bits)` combination, averaged across batch and query positions. Higher variance means the model is more "peaked" in attention; lower means flatter.
- `pytorch_vs_baseline_Δ` / `triton_vs_baseline_Δ`: absolute difference in variance vs the uncompressed baseline — a measure of how much compression distorts the attention distribution.

```
┌───────┬──────┬──────┬──────────────┬─────────────┬─────────────┬─────────────────────────┬──────────────────────────┐
│ layer │ head │ bits │ baseline_var │ pytorch_var │ triton_var  │ pytorch_vs_baseline_Δ   │ triton_vs_baseline_Δ     │
└───────┴──────┴──────┴──────────────┴─────────────┴─────────────┴─────────────────────────┴──────────────────────────┘
```

Run via `just test-slow` (new justfile target: `uv run pytest -m slow -s`).

`pytest.ini_options` gains `addopts` filter `-m "not slow"` so slow tests are excluded from `just test`.

---

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    ...
    "triton>=3.0.0",   # GPU kernel compilation
]
```

Triton is included with `torch` nightlies but pinned explicitly for reproducibility. On CPU-only machines, `triton` is still installable but kernels are skipped at runtime via the `use_triton=False` fallback.

**ImportError handling:** If `import triton` raises at runtime (e.g., installed but unable to compile on the current hardware), the compressor classes catch the `ImportError`, emit a `warnings.warn` with the reason, and silently fall back to `use_triton=False`. They do not raise. This ensures the library remains usable in CPU-only or unsupported GPU environments.

---

## Out of Scope

- CPU Triton execution (Triton is GPU-only)
- Rewriting the rotation step (`torch.matmul` → cuBLAS is already optimal)
- Multi-GPU / tensor-parallel support
- INT4 / INT8 tensor core paths (codebook lookup is too irregular for tensor cores)
