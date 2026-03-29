# transformers-turboquant Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `transformers_turboquant` — a HuggingFace integration layer that patches any supported model's attention with TurboQuant KV cache compression via a single `apply_turboquant(model, bits=3)` call.

**Architecture:** A registry dispatches on `model.config.model_type` to a family-specific `TurboQuantAttentionBase` subclass. `apply_turboquant()` walks `named_modules()`, identifies attention layers by class name, creates one key compressor (`TurboQuantCompressorV2`) and one value compressor (`TurboQuantCompressorMSE`) per layer, and replaces each attention module in-place via `setattr`. Training and inference use the same forward: compress K/V under `torch.no_grad()`, compute scores via `asymmetric_attention_scores`, reconstruct V with `decompress`.

**Tech Stack:** PyTorch, HuggingFace `transformers>=5.4.0`, TRL `>=0.29.1`, `turboquant_pytorch` (existing), `pytest`, `uv_build`, GitHub Actions

---

## File Map

| File | Created / Modified | Responsibility |
|------|--------------------|----------------|
| `pyproject.toml` | Modified | Add dual-module build + `validate` script entry point |
| `src/turboquant_pytorch/validate.py` | Modified | Remove `sys.path.insert` hack; fix broken import |
| `src/transformers_turboquant/__init__.py` | Modified | Export `apply_turboquant`; trigger family registration |
| `src/transformers_turboquant/base.py` | Created | `TurboQuantAttentionBase` ABC |
| `src/transformers_turboquant/registry.py` | Created | `@register_family` decorator + `get_attention_class()` + `UnsupportedModelError` |
| `src/transformers_turboquant/patch.py` | Created | `apply_turboquant()` implementation |
| `src/transformers_turboquant/cli.py` | Modified | Replace hello-world stub with `validate` command |
| `src/transformers_turboquant/families/__init__.py` | Created | Import all family modules to fire decorators |
| `src/transformers_turboquant/families/gpt2.py` | Created | `TurboQuantGPT2Attention` |
| `src/transformers_turboquant/families/qwen2.py` | Created | `TurboQuantQwen2Attention` |
| `src/transformers_turboquant/families/llama.py` | Created | `TurboQuantLlamaAttention` |
| `src/transformers_turboquant/families/deepseek.py` | Created | `TurboQuantDeepSeekAttention` |
| `tests/test_registry.py` | Created | Registry dispatch + `UnsupportedModelError` |
| `tests/test_apply_turboquant.py` | Created | `apply_turboquant` + `generate()` on tiny-gpt2 |
| `tests/test_pipeline.py` | Created | `pipeline("text-generation")` with patched tiny-gpt2 |
| `tests/test_sft.py` | Created | Single `SFTTrainer` train step on patched tiny-gpt2 |
| `tests/test_slow_models.py` | Created | Qwen2.5-3B, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen |
| `justfile` | Modified | Add `validate` recipe |
| `.github/workflows/fast-tests.yml` | Created | CI: fast tests on push/PR |
| `.github/workflows/publish.yml` | Created | PyPI publish on `v*` tag |

---

### Task 1: pyproject.toml — dual module + validate entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `[tool.uv.build-backend]` section and `validate` entry point**

Open `pyproject.toml`. After the `[build-system]` block, add the build-backend module config. Also add the `validate` script under `[project.scripts]`:

```toml
[build-system]
requires = ["uv_build>=0.10.0,<0.11.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = ["transformers_turboquant", "turboquant_pytorch"]
module-root = "src"

[project]
name = "transformers-turboquant"
version = "0.1.0"
description = ""
authors = [{ name = "", email = "" }]
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "accelerate>=1.13.0",
    "bitsandbytes>=0.49.2",
    "scipy>=1.17.1",
    "torch>=2.11.0",
    "transformers>=5.4.0",
    "triton>=3.0.0",
    "trl>=0.29.1",
    "typer>=0.12",
]

[project.scripts]
transformers-turboquant = "transformers_turboquant.cli:app"
validate = "turboquant_pytorch.validate:main"
```

- [ ] **Step 2: Verify uv can sync**

```bash
uv sync --all-groups
```
Expected: resolves without error. Both packages now discoverable.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add dual-module build config and validate entry point"
```

---

### Task 2: base.py + registry.py — ABC and dispatch (TDD)

**Files:**
- Create: `src/transformers_turboquant/base.py`
- Create: `src/transformers_turboquant/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_registry.py`:

```python
import pytest
from transformers_turboquant.registry import (
    UnsupportedModelError,
    get_attention_class,
    register_family,
)
from transformers_turboquant.base import TurboQuantAttentionBase


def test_register_and_dispatch_single_type():
    @register_family("_test_model_a")
    class _FakeAttentionA(TurboQuantAttentionBase):
        ATTENTION_CLASS_NAMES = ("_FakeA",)

        @classmethod
        def from_original(cls, original, key_compressor, val_compressor):
            raise NotImplementedError

        def forward(self, hidden_states, **kwargs):
            raise NotImplementedError

    cls = get_attention_class("_test_model_a")
    assert cls is _FakeAttentionA


def test_register_multiple_types_same_class():
    @register_family("_test_b1", "_test_b2")
    class _FakeAttentionB(TurboQuantAttentionBase):
        ATTENTION_CLASS_NAMES = ("_FakeB",)

        @classmethod
        def from_original(cls, original, key_compressor, val_compressor):
            raise NotImplementedError

        def forward(self, hidden_states, **kwargs):
            raise NotImplementedError

    assert get_attention_class("_test_b1") is _FakeAttentionB
    assert get_attention_class("_test_b2") is _FakeAttentionB


def test_unsupported_model_raises_with_helpful_message():
    with pytest.raises(UnsupportedModelError) as exc_info:
        get_attention_class("__nonexistent_xyz__")
    msg = str(exc_info.value)
    assert "__nonexistent_xyz__" in msg
    assert "Supported:" in msg
    assert "@register_family" in msg


def test_known_families_present_after_import():
    import transformers_turboquant  # noqa: F401 — triggers registration
    for model_type in ("gpt2", "qwen2", "llama", "mistral"):
        cls = get_attention_class(model_type)
        assert issubclass(cls, TurboQuantAttentionBase)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_registry.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — `base.py` and `registry.py` don't exist yet.

- [ ] **Step 3: Create `src/transformers_turboquant/base.py`**

```python
"""Abstract base class for TurboQuant attention replacements."""

from abc import ABC, abstractmethod
from typing import ClassVar

import torch.nn as nn

from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2


class TurboQuantAttentionBase(ABC, nn.Module):
    """
    Base class for all TurboQuant attention replacements.

    Each family subclass declares which HuggingFace attention class names it
    can replace via ATTENTION_CLASS_NAMES, and implements from_original() to
    copy weights from the original module and forward() to run compressed attention.
    """

    ATTENTION_CLASS_NAMES: ClassVar[tuple[str, ...]]

    @classmethod
    @abstractmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantAttentionBase":
        """Copy weights from original attention module and store compressors."""
        ...

    @abstractmethod
    def forward(self, hidden_states, **kwargs):
        """Run compressed attention forward pass."""
        ...
```

- [ ] **Step 4: Create `src/transformers_turboquant/registry.py`**

```python
"""Registry mapping model_type strings to TurboQuantAttentionBase subclasses."""

from .base import TurboQuantAttentionBase

_REGISTRY: dict[str, type[TurboQuantAttentionBase]] = {}


class UnsupportedModelError(ValueError):
    """Raised when apply_turboquant encounters an unregistered model_type."""


def register_family(*model_types: str):
    """
    Class decorator that registers an attention class for one or more model_type strings.

    Usage:
        @register_family("llama", "mistral")
        class TurboQuantLlamaAttention(TurboQuantAttentionBase): ...
    """
    def decorator(cls: type[TurboQuantAttentionBase]) -> type[TurboQuantAttentionBase]:
        for mt in model_types:
            _REGISTRY[mt] = cls
        return cls
    return decorator


def get_attention_class(model_type: str) -> type[TurboQuantAttentionBase]:
    """Return the registered attention class for the given model_type."""
    if model_type not in _REGISTRY:
        supported = sorted(_REGISTRY.keys())
        raise UnsupportedModelError(
            f"model_type={model_type!r} is not supported. "
            f"Supported: {supported}. "
            f"To add support, implement TurboQuantAttentionBase in a new file "
            f"and register it with @register_family({model_type!r})."
        )
    return _REGISTRY[model_type]
```

- [ ] **Step 5: Create stub `src/transformers_turboquant/families/__init__.py`** (empty for now — populated in Task 3)

```python
# Family modules are imported here so their @register_family decorators fire at package init.
# Each task adds an import as the family is implemented.
```

- [ ] **Step 6: Update `src/transformers_turboquant/__init__.py`** to import families

```python
""""""

from . import families  # noqa: F401 — triggers @register_family decorators

__version__ = "0.1.0"
```

- [ ] **Step 7: Run the first three registry tests (skip the last one — families not registered yet)**

```bash
uv run pytest tests/test_registry.py::test_register_and_dispatch_single_type tests/test_registry.py::test_register_multiple_types_same_class tests/test_registry.py::test_unsupported_model_raises_with_helpful_message -v
```
Expected: all 3 PASS.

- [ ] **Step 8: Commit**

```bash
git add src/transformers_turboquant/base.py src/transformers_turboquant/registry.py \
        src/transformers_turboquant/families/__init__.py src/transformers_turboquant/__init__.py \
        tests/test_registry.py
git commit -m "feat: add TurboQuantAttentionBase ABC and family registry"
```

---

### Task 3: families/gpt2.py — GPT2 attention family

**Files:**
- Create: `src/transformers_turboquant/families/gpt2.py`
- Modify: `src/transformers_turboquant/families/__init__.py`

GPT2Attention uses Conv1D projections: `c_attn` (3×hidden combined QKV) and `c_proj` (output). The model uses absolute positional embeddings at the embedding layer — attention gets no position args. Causal masking is passed in via `attention_mask`.

- [ ] **Step 1: Create `src/transformers_turboquant/families/gpt2.py`**

```python
"""TurboQuant attention replacement for GPT-2 family models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers_turboquant.base import TurboQuantAttentionBase
from transformers_turboquant.registry import register_family
from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2


@register_family("gpt2")
class TurboQuantGPT2Attention(TurboQuantAttentionBase):
    """
    Drop-in replacement for GPT2Attention that uses TurboQuant compressed KV.

    Copies all projections from the original module. The forward recomputes
    full-sequence attention on every call (no incremental KV cache), which is
    correct for both training and inference.
    """

    ATTENTION_CLASS_NAMES = ("GPT2Attention",)

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantGPT2Attention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        # Copy projection layers (Conv1D — these are nn.Modules so they register as submodules)
        obj.c_attn = original.c_attn
        obj.c_proj = original.c_proj
        obj.attn_dropout = original.attn_dropout
        obj.resid_dropout = original.resid_dropout
        # Scalars
        obj.embed_dim = original.embed_dim
        obj.num_heads = original.num_heads
        obj.head_dim = original.head_dim
        # Compressors (plain Python objects, not nn.Modules — initialized on correct device)
        obj.key_compressor = key_compressor
        obj.val_compressor = val_compressor
        return obj

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        B, T, _ = hidden_states.shape

        # Project Q, K, V from combined c_attn weight
        qkv = self.c_attn(hidden_states)  # (B, T, 3 * embed_dim)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape to (B, H, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compress K and V — no_grad is enforced by compressor decorators
        compressed_k = self.key_compressor.compress(k)
        compressed_v = self.val_compressor.compress(v)

        # Asymmetric attention scores: <Q, K> estimated from compressed K
        scores = self.key_compressor.asymmetric_attention_scores(q, compressed_k)  # (B, H, T, T)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask  # additive causal mask (−inf for future tokens)

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = self.attn_dropout(weights)

        # Reconstruct V and compute weighted sum
        values_recon = self.val_compressor.decompress(compressed_v).to(q.dtype)  # (B, H, T, head_dim)
        attn_output = torch.matmul(weights, values_recon)  # (B, H, T, head_dim)

        # Merge heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs
```

- [ ] **Step 2: Register gpt2 in `families/__init__.py`**

```python
# Family modules are imported here so their @register_family decorators fire at package init.
from . import gpt2  # noqa: F401
```

- [ ] **Step 3: Verify the last registry test now passes**

```bash
uv run pytest tests/test_registry.py::test_known_families_present_after_import -v
```
Expected: FAIL (qwen2/llama/mistral not registered yet). That's OK — we add those families later.

Instead, verify gpt2 is registered:

```bash
uv run python -c "from transformers_turboquant.registry import get_attention_class; print(get_attention_class('gpt2'))"
```
Expected: `<class 'transformers_turboquant.families.gpt2.TurboQuantGPT2Attention'>`

- [ ] **Step 4: Commit**

```bash
git add src/transformers_turboquant/families/gpt2.py src/transformers_turboquant/families/__init__.py
git commit -m "feat: add GPT2 attention family"
```

---

### Task 4: patch.py — apply_turboquant()

**Files:**
- Create: `src/transformers_turboquant/patch.py`

`apply_turboquant` walks the model tree, finds attention layers by class name, and replaces them. It also sets `model.config.use_cache = False` so HuggingFace `generate()` doesn't expect an incremental KV cache — our forward computes full-sequence compressed attention on every call.

- [ ] **Step 1: Create `src/transformers_turboquant/patch.py`**

```python
"""apply_turboquant: replace attention layers in a PreTrainedModel."""

from transformers import PreTrainedModel

from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2

from .registry import get_attention_class


def apply_turboquant(
    model: PreTrainedModel,
    bits: int = 3,
    seed: int = 42,
    use_triton: bool = True,
) -> PreTrainedModel:
    """
    Replace all attention layers in *model* with TurboQuant compressed versions.

    The model is mutated in-place and returned for chaining. ``use_cache`` is
    set to False on the model config because the replacement attention modules
    compute full-sequence compressed attention on every forward call.

    Args:
        model: Any HuggingFace PreTrainedModel whose model_type is registered.
        bits: Quantisation bit-width for KV cache (2, 3, or 4).
        seed: Base RNG seed; layer i gets seed+i (keys) and seed+i+10000 (values).
        use_triton: Enable Triton kernels if available; falls back to PyTorch.

    Returns:
        The same *model* object with attention layers replaced.
    """
    model_type = model.config.model_type
    attention_cls = get_attention_class(model_type)

    device = next(model.parameters()).device
    device_str = str(device)

    i = 0
    # Walk the module tree; named_children() gives (child_name, child_module)
    # so we have the parent reference needed for setattr.
    for _parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if type(child_module).__name__ not in attention_cls.ATTENTION_CLASS_NAMES:
                continue

            # Resolve head_dim: prefer attribute on the module, fall back to config
            head_dim = getattr(child_module, "head_dim", None)
            if head_dim is None:
                hidden_size = getattr(
                    model.config, "hidden_size", getattr(model.config, "n_embd", None)
                )
                num_heads = getattr(
                    model.config,
                    "num_attention_heads",
                    getattr(model.config, "n_head", None),
                )
                head_dim = hidden_size // num_heads

            key_comp = TurboQuantCompressorV2(
                head_dim, bits, seed=seed + i, device=device_str, use_triton=use_triton
            )
            val_comp = TurboQuantCompressorMSE(
                head_dim, bits, seed=seed + i + 10000, device=device_str, use_triton=use_triton
            )
            new_attn = attention_cls.from_original(child_module, key_comp, val_comp)
            setattr(parent_module, child_name, new_attn)
            i += 1

    # Disable incremental KV caching — our forward does full-sequence attention
    model.config.use_cache = False
    return model
```

- [ ] **Step 2: Export from `src/transformers_turboquant/__init__.py`**

```python
""""""

from . import families  # noqa: F401 — triggers @register_family decorators
from .patch import apply_turboquant

__version__ = "0.1.0"
__all__ = ["apply_turboquant"]
```

- [ ] **Step 3: Smoke-test the import**

```bash
uv run python -c "from transformers_turboquant import apply_turboquant; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/transformers_turboquant/patch.py src/transformers_turboquant/__init__.py
git commit -m "feat: add apply_turboquant patch function"
```

---

### Task 5: test_apply_turboquant.py — end-to-end generate test (TDD)

**Files:**
- Create: `tests/test_apply_turboquant.py`

Uses `sshleifer/tiny-gpt2` — a 6-layer, 64-hidden model (~6 MB). Downloads from HuggingFace Hub on first run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apply_turboquant.py`:

```python
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_turboquant import apply_turboquant

MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_gpt2():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


def test_apply_turboquant_replaces_attention_layers(tiny_gpt2):
    model, _ = tiny_gpt2
    # Count attention layers before patching (use a fresh copy so other tests aren't affected)
    import copy
    m = copy.deepcopy(model)
    from transformers_turboquant.families.gpt2 import TurboQuantGPT2Attention
    before = sum(1 for _, mod in m.named_modules() if type(mod).__name__ == "GPT2Attention")
    apply_turboquant(m, bits=3)
    after = sum(1 for _, mod in m.named_modules() if isinstance(mod, TurboQuantGPT2Attention))
    assert before > 0, "tiny-gpt2 should have GPT2Attention layers"
    assert after == before, f"expected {before} replacements, got {after}"


def test_apply_turboquant_generate_runs(tiny_gpt2):
    import copy
    model, tokenizer = tiny_gpt2
    m = copy.deepcopy(model)
    apply_turboquant(m, bits=3)
    m.eval()

    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        output = m.generate(**inputs, max_new_tokens=5, do_sample=False)

    assert output.shape[0] == 1
    assert output.shape[1] == inputs["input_ids"].shape[1] + 5


def test_apply_turboquant_use_cache_disabled(tiny_gpt2):
    import copy
    model, _ = tiny_gpt2
    m = copy.deepcopy(model)
    apply_turboquant(m, bits=3)
    assert m.config.use_cache is False


def test_apply_turboquant_returns_same_object(tiny_gpt2):
    import copy
    model, _ = tiny_gpt2
    m = copy.deepcopy(model)
    result = apply_turboquant(m, bits=3)
    assert result is m
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_apply_turboquant.py -v
```
Expected: FAIL — `apply_turboquant` is importable but `generate()` is broken because the replaced attention forward has a shape or interface bug, or passes (if the implementation in Task 4 is correct). Either way, run now to see baseline.

- [ ] **Step 3: Run all fast tests to get baseline**

```bash
uv run pytest tests/test_apply_turboquant.py tests/test_registry.py -v --no-cov
```
Expected: registry tests pass; apply tests either pass or fail with a specific error.

- [ ] **Step 4: Fix any failures**

Common issues to check if generate fails:

**Issue A: `scores` dtype mismatch** — `asymmetric_attention_scores` returns float32, `q` may be float32 too, but if model is fp16 there can be issues. Fix: cast scores to `q.dtype` before softmax if they differ.

**Issue B: `attention_mask` shape** — GPT2's attention mask is `(B, 1, 1, T)` with large negative values. Check it's added (not multiplied) to scores.

**Issue C: `c_attn` output dim** — `sshleifer/tiny-gpt2` may have `embed_dim=64`, `num_heads=2`. Verify `qkv.split(self.embed_dim, dim=2)` yields three tensors of shape `(B, T, 64)`.

Run the generate test with verbose output to diagnose:
```bash
uv run pytest tests/test_apply_turboquant.py::test_apply_turboquant_generate_runs -v -s
```

- [ ] **Step 5: All apply tests pass**

```bash
uv run pytest tests/test_apply_turboquant.py -v --no-cov
```
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_apply_turboquant.py
git commit -m "test: add apply_turboquant end-to-end tests with tiny-gpt2"
```

---

### Task 6: test_pipeline.py — HuggingFace pipeline integration (TDD)

**Files:**
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from transformers_turboquant import apply_turboquant

MODEL_ID = "sshleifer/tiny-gpt2"


def test_pipeline_text_generation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = apply_turboquant(copy.deepcopy(model), bits=3)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    with torch.no_grad():
        result = pipe("The quick brown fox", max_new_tokens=5, do_sample=False)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "generated_text" in result[0]
    assert len(result[0]["generated_text"]) > len("The quick brown fox")
```

- [ ] **Step 2: Run to verify it fails (or passes)**

```bash
uv run pytest tests/test_pipeline.py -v --no-cov
```
Expected: PASS if apply_turboquant + GPT2 attention is working. FAIL otherwise — diagnose as in Task 5 Step 4.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add pipeline integration test with patched tiny-gpt2"
```

---

### Task 7: test_sft.py — SFTTrainer single training step (TDD)

**Files:**
- Create: `tests/test_sft.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sft.py`:

```python
import copy
import os

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from transformers_turboquant import apply_turboquant

MODEL_ID = "sshleifer/tiny-gpt2"


def test_sft_single_step_completes():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = apply_turboquant(copy.deepcopy(model), bits=3)

    dataset = Dataset.from_dict({"text": ["Hello world, this is a test."] * 4})

    config = SFTConfig(
        max_steps=1,
        output_dir="/tmp/tq-sft-test",
        max_seq_length=32,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    # If we get here without an exception, training ran successfully
```

- [ ] **Step 2: Run to verify it fails (or passes)**

```bash
uv run pytest tests/test_sft.py -v --no-cov -s
```
Expected: PASS if gradients flow correctly through Q. FAIL if:
- **`RuntimeError: element 0 of tensors does not require grad`** — the `compress()` call is stripping the computation graph for Q. Check that only K/V are compressed, Q goes through standard matmul. Our implementation compresses K and V but uses `q` directly in `asymmetric_attention_scores` (Q participates in the differentiable part).
- **`TypeError` on `SFTTrainer` args** — TRL version mismatch. If `processing_class` isn't accepted, try `tokenizer=tokenizer` instead.

- [ ] **Step 3: Fix failures if any, then confirm pass**

```bash
uv run pytest tests/test_sft.py -v --no-cov
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sft.py
git commit -m "test: add SFT single-step training test with patched tiny-gpt2"
```

---

### Task 8: families/qwen2.py — Qwen2 attention family

**Files:**
- Create: `src/transformers_turboquant/families/qwen2.py`
- Modify: `src/transformers_turboquant/families/__init__.py`

In transformers>=5.x, rotary embeddings are pre-computed in the decoder layer and passed to attention as `position_embeddings: Tuple[Tensor, Tensor]` (cos, sin). No `rotary_emb` module is needed on the attention class itself.

GQA: Qwen2 uses `num_key_value_heads < num_heads`. K and V are repeated `num_key_value_groups` times before attention.

- [ ] **Step 1: Create `src/transformers_turboquant/families/qwen2.py`**

```python
"""TurboQuant attention replacement for Qwen2 family models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from transformers_turboquant.base import TurboQuantAttentionBase
from transformers_turboquant.registry import register_family
from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2


@register_family("qwen2")
class TurboQuantQwen2Attention(TurboQuantAttentionBase):
    """
    Drop-in replacement for Qwen2Attention / Qwen2SdpaAttention.

    Handles GQA by expanding K and V to num_heads before compressing.
    Rotary embeddings are received pre-computed as (cos, sin) from the decoder layer.
    """

    ATTENTION_CLASS_NAMES = ("Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2")

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantQwen2Attention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.q_proj = original.q_proj
        obj.k_proj = original.k_proj
        obj.v_proj = original.v_proj
        obj.o_proj = original.o_proj
        obj.num_heads = original.num_heads
        obj.num_key_value_heads = original.num_key_value_heads
        obj.num_key_value_groups = original.num_heads // original.num_key_value_heads
        obj.head_dim = original.head_dim
        obj.hidden_size = original.hidden_size
        obj.key_compressor = key_compressor
        obj.val_compressor = val_compressor
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand K, V to match number of query heads for GQA
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)  # (B, num_heads, T, head_dim)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        compressed_k = self.key_compressor.compress(k)
        compressed_v = self.val_compressor.compress(v)

        scores = self.key_compressor.asymmetric_attention_scores(q, compressed_k)  # (B, H, T, T)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        values_recon = self.val_compressor.decompress(compressed_v).to(q.dtype)
        attn_output = torch.matmul(weights, values_recon)  # (B, H, T, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        attn_weights = weights if output_attentions else None
        return attn_output, attn_weights, None
```

- [ ] **Step 2: Add to `families/__init__.py`**

```python
from . import gpt2, qwen2  # noqa: F401
```

- [ ] **Step 3: Verify registration**

```bash
uv run python -c "from transformers_turboquant.registry import get_attention_class; print(get_attention_class('qwen2'))"
```
Expected: `<class 'transformers_turboquant.families.qwen2.TurboQuantQwen2Attention'>`

- [ ] **Step 4: Commit**

```bash
git add src/transformers_turboquant/families/qwen2.py src/transformers_turboquant/families/__init__.py
git commit -m "feat: add Qwen2 attention family (GQA + rotary embeddings)"
```

---

### Task 9: families/llama.py — LLaMA/Mistral attention family

**Files:**
- Create: `src/transformers_turboquant/families/llama.py`
- Modify: `src/transformers_turboquant/families/__init__.py`

LLaMA and Mistral share the same attention architecture as Qwen2 (GQA, RoPE, `q/k/v/o_proj`). The only differences are the class names and import path for `apply_rotary_pos_emb`.

- [ ] **Step 1: Create `src/transformers_turboquant/families/llama.py`**

```python
"""TurboQuant attention replacement for LLaMA and Mistral family models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from transformers_turboquant.base import TurboQuantAttentionBase
from transformers_turboquant.registry import register_family
from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2


@register_family("llama", "mistral")
class TurboQuantLlamaAttention(TurboQuantAttentionBase):
    """
    Drop-in replacement for LlamaAttention, LlamaSdpaAttention,
    MistralAttention, and MistralSdpaAttention.

    Architecture identical to Qwen2 (GQA + pre-computed RoPE).
    """

    ATTENTION_CLASS_NAMES = (
        "LlamaAttention",
        "LlamaSdpaAttention",
        "LlamaFlashAttention2",
        "MistralAttention",
        "MistralSdpaAttention",
        "MistralFlashAttention2",
    )

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantLlamaAttention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.q_proj = original.q_proj
        obj.k_proj = original.k_proj
        obj.v_proj = original.v_proj
        obj.o_proj = original.o_proj
        obj.num_heads = original.num_heads
        obj.num_key_value_heads = original.num_key_value_heads
        obj.num_key_value_groups = original.num_heads // original.num_key_value_heads
        obj.head_dim = original.head_dim
        obj.hidden_size = original.hidden_size
        obj.key_compressor = key_compressor
        obj.val_compressor = val_compressor
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        compressed_k = self.key_compressor.compress(k)
        compressed_v = self.val_compressor.compress(v)

        scores = self.key_compressor.asymmetric_attention_scores(q, compressed_k)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        values_recon = self.val_compressor.decompress(compressed_v).to(q.dtype)
        attn_output = torch.matmul(weights, values_recon)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        attn_weights = weights if output_attentions else None
        return attn_output, attn_weights, None
```

- [ ] **Step 2: Add to `families/__init__.py`**

```python
from . import gpt2, llama, qwen2  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
git add src/transformers_turboquant/families/llama.py src/transformers_turboquant/families/__init__.py
git commit -m "feat: add LLaMA/Mistral attention family"
```

---

### Task 10: families/deepseek.py — DeepSeek attention family

**Files:**
- Create: `src/transformers_turboquant/families/deepseek.py`
- Modify: `src/transformers_turboquant/families/__init__.py`

DeepSeek-V1 uses `DeepseekAttention` (LLaMA-like architecture with RoPE). DeepSeek-V2/V3 use MLA — registered here with the same class but MLA support can be added later. `DeepSeek-R1-Distill-Qwen` has `model_type="qwen2"` and is handled by the qwen2 family.

- [ ] **Step 1: Create `src/transformers_turboquant/families/deepseek.py`**

```python
"""TurboQuant attention replacement for DeepSeek family models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers_turboquant.base import TurboQuantAttentionBase
from transformers_turboquant.registry import register_family
from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2

try:
    from transformers.models.deepseek.modeling_deepseek import apply_rotary_pos_emb
except ImportError:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


@register_family("deepseek", "deepseek_v2", "deepseek_v3")
class TurboQuantDeepSeekAttention(TurboQuantAttentionBase):
    """
    Drop-in replacement for DeepseekAttention (V1 architecture, LLaMA-like).

    Note: DeepSeek-R1-Distill-Qwen uses model_type="qwen2" and is handled by
    the qwen2 family. This class handles native DeepSeek model checkpoints.
    """

    ATTENTION_CLASS_NAMES = (
        "DeepseekAttention",
        "DeepseekSdpaAttention",
        "DeepseekFlashAttention2",
        "DeepseekV2Attention",
        "DeepseekV3Attention",
    )

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantDeepSeekAttention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.q_proj = original.q_proj
        obj.k_proj = original.k_proj
        obj.v_proj = original.v_proj
        obj.o_proj = original.o_proj
        obj.num_heads = original.num_heads
        obj.num_key_value_heads = original.num_key_value_heads
        obj.num_key_value_groups = original.num_heads // original.num_key_value_heads
        obj.head_dim = original.head_dim
        obj.hidden_size = original.hidden_size
        obj.key_compressor = key_compressor
        obj.val_compressor = val_compressor
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        compressed_k = self.key_compressor.compress(k)
        compressed_v = self.val_compressor.compress(v)

        scores = self.key_compressor.asymmetric_attention_scores(q, compressed_k)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        values_recon = self.val_compressor.decompress(compressed_v).to(q.dtype)
        attn_output = torch.matmul(weights, values_recon)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        attn_weights = weights if output_attentions else None
        return attn_output, attn_weights, None
```

- [ ] **Step 2: Add to `families/__init__.py`**

```python
from . import deepseek, gpt2, llama, qwen2  # noqa: F401
```

- [ ] **Step 3: Run the full known-families registry test**

```bash
uv run pytest tests/test_registry.py::test_known_families_present_after_import -v --no-cov
```
Expected: PASS (gpt2, qwen2, llama, mistral all registered).

- [ ] **Step 4: Run all fast tests**

```bash
uv run pytest tests/test_registry.py tests/test_apply_turboquant.py tests/test_pipeline.py tests/test_sft.py -v --no-cov
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transformers_turboquant/families/deepseek.py src/transformers_turboquant/families/__init__.py
git commit -m "feat: add DeepSeek attention family"
```

---

### Task 11: test_slow_models.py — slow tests for real models

**Files:**
- Create: `tests/test_slow_models.py`

These tests require a GPU with sufficient VRAM and download large checkpoints (~0.8–3B parameters). They are excluded from normal CI runs via `@pytest.mark.slow`.

- [ ] **Step 1: Create `tests/test_slow_models.py`**

```python
"""
Slow integration tests: apply_turboquant on real models and verify generate() works.

Run with: just test-slow
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_turboquant import apply_turboquant

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]


@pytest.mark.slow
@pytest.mark.parametrize("model_id", MODELS)
def test_apply_turboquant_generate_slow(model_id):
    """apply_turboquant + generate() works on a real model checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = apply_turboquant(model, bits=3, use_triton=True)
    model.eval()

    inputs = tokenizer("What is the capital of France?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    assert output.shape[0] == 1
    assert output.shape[1] > inputs["input_ids"].shape[1]


@pytest.mark.slow
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_apply_turboquant_bits_slow(model_id, bits):
    """apply_turboquant works at 2, 3, and 4 bits."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = apply_turboquant(model, bits=bits, use_triton=True)
    model.eval()

    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    assert output.shape[1] > inputs["input_ids"].shape[1]
```

- [ ] **Step 2: Verify tests are collected with slow marker**

```bash
uv run pytest tests/test_slow_models.py --collect-only
```
Expected: lists the test cases but none are selected by default (`-m "not slow"` excludes them).

- [ ] **Step 3: Commit**

```bash
git add tests/test_slow_models.py
git commit -m "test: add slow integration tests for Qwen2.5, Qwen3.5, DeepSeek-R1-Distill"
```

---

### Task 12: Cleanup — validate.py, cli.py, justfile

**Files:**
- Modify: `src/turboquant_pytorch/validate.py`
- Modify: `src/transformers_turboquant/cli.py`
- Modify: `justfile`

- [ ] **Step 1: Fix validate.py — remove sys.path hack, fix import**

In `src/turboquant_pytorch/validate.py`, remove lines 9–16:

```python
# REMOVE these lines:
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turboquant.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
```

Replace with:

```python
from turboquant_pytorch.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
```

Also verify the `import os` and `import sys` lines are removed (they were only used for the path hack). The final imports section should look like:

```python
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquant_pytorch.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2
```

- [ ] **Step 2: Update cli.py — replace hello-world with validate command**

```python
"""CLI entrypoint for transformers-turboquant."""

import typer

app = typer.Typer(help="transformers-turboquant: HuggingFace integration for TurboQuant KV compression.")


@app.command()
def validate() -> None:
    """Run needle-in-haystack validation on Qwen2.5-3B-Instruct (requires GPU)."""
    from turboquant_pytorch.validate import main
    main()


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Add validate recipe to justfile**

Add after the `test-slow` recipe:

```makefile
validate:
    uv run transformers-turboquant validate
```

Full relevant section of justfile:

```makefile
test-slow:
    uv run pytest -m slow -s

validate:
    uv run transformers-turboquant validate
```

- [ ] **Step 4: Verify CLI entry point works**

```bash
uv run transformers-turboquant --help
```
Expected: shows `validate` as a subcommand.

- [ ] **Step 5: Run all fast tests to ensure nothing broke**

```bash
uv run pytest --no-cov -m "not slow"
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant_pytorch/validate.py src/transformers_turboquant/cli.py justfile
git commit -m "chore: fix validate.py import, add validate CLI command and justfile recipe"
```

---

### Task 13: GitHub Actions — fast-tests.yml and publish.yml

**Files:**
- Create: `.github/workflows/fast-tests.yml`
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create `.github/workflows/fast-tests.yml`**

```yaml
name: Fast Tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Run fast tests
        run: just test-fast
```

- [ ] **Step 2: Create `.github/workflows/publish.yml`**

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # required for trusted publishing

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Build packages
        run: uv build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/fast-tests.yml .github/workflows/publish.yml
git commit -m "ci: add fast-tests workflow and PyPI publish workflow"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|-----------------|------|
| `pyproject.toml` dual module `[tool.uv.build-backend]` | Task 1 |
| Package layout: `patch.py`, `registry.py`, `base.py`, `families/` | Tasks 2–4 |
| `@register_family` decorator + `get_attention_class` | Task 2 |
| `UnsupportedModelError` with helpful message | Task 2 |
| `TurboQuantAttentionBase` ABC with `ATTENTION_CLASS_NAMES`, `from_original`, `forward` | Task 2 |
| `apply_turboquant(model, bits, seed, use_triton)` signature | Task 4 |
| Resolve device from model's first parameter | Task 4 |
| Walk `named_modules`, identify by `type(module).__name__` | Task 4 |
| Create `TurboQuantCompressorV2(head_dim, bits, seed=seed+i)` | Task 4 |
| Create `TurboQuantCompressorMSE(head_dim, bits, seed=seed+i+10000)` | Task 4 |
| Replace via `setattr` on parent | Task 4 |
| Returns same model object | Task 4 |
| `families/gpt2.py` model_type "gpt2" | Task 3 |
| `families/qwen2.py` model_type "qwen2" | Task 8 |
| `families/llama.py` model_type "llama", "mistral" | Task 9 |
| `families/deepseek.py` model_type "deepseek", "deepseek_v2", "deepseek_v3" | Task 10 |
| All families imported at package init | Tasks 3, 8, 9, 10 |
| Inference: `compress()`, `asymmetric_attention_scores()`, `softmax @ decompress()` | Tasks 3, 8, 9, 10 |
| Training: same forward, `compress()` is `@torch.no_grad()` | Tasks 3, 8, 9, 10 |
| `test_registry.py` | Task 2 |
| `test_apply_turboquant.py` (tiny-gpt2 generate) | Task 5 |
| `test_pipeline.py` | Task 6 |
| `test_sft.py` | Task 7 |
| `test_slow_models.py` (Qwen2.5-3B, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen) | Task 11 |
| `validate.py` remove `sys.path.insert`, fix import | Task 12 |
| `cli.py` validate command | Task 12 |
| `justfile` validate recipe | Task 12 |
| `fast-tests.yml` GitHub Action | Task 13 |
| `publish.yml` GitHub Action | Task 13 |

All spec requirements covered. ✓

### Placeholder scan

No TBDs or TODOs in code blocks. All code is complete. ✓

### Type consistency check

- `ATTENTION_CLASS_NAMES` is `tuple[str, ...]` in base.py and all family classes ✓
- `from_original` signature matches base class in all families ✓
- `compress()` called with `(B, H, T, head_dim)` tensors consistently across families ✓
- `asymmetric_attention_scores(q, compressed_k)` where `q` is `(B, H, T, head_dim)` ✓
- `decompress(compressed_v)` returns `(B, H, T, head_dim)` ✓
- `register_family` returns the class unchanged (used as decorator) ✓
- `get_attention_class` returns `type[TurboQuantAttentionBase]` ✓
