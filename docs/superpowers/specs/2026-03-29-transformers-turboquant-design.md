# transformers-turboquant Integration Design

**Date:** 2026-03-29
**Status:** Approved

## Overview

Add a proper integration layer (`transformers_turboquant`) that makes the existing `turboquant_pytorch` KV cache quantization algorithm easy to use with HuggingFace `transformers` — including `pipeline`, `AutoModelForCausalLM`, and SFT training via TRL. Also includes repo cleanup and new GitHub Actions.

---

## Section 1: Module structure & registry

### Package layout

Both packages are declared as installable modules in `pyproject.toml` via `uv_build`:

```toml
[tool.uv.build-backend]
module-name = ["transformers_turboquant", "turboquant_pytorch"]
module-root = "src"
```

```
src/
  turboquant_pytorch/           # existing — unchanged
  transformers_turboquant/
    __init__.py                 # exports apply_turboquant
    patch.py                    # apply_turboquant() implementation
    registry.py                 # @register_family decorator + dispatch
    base.py                     # TurboQuantAttentionBase ABC
    families/
      __init__.py
      gpt2.py                   # model_type: "gpt2"
      qwen2.py                  # model_type: "qwen2" (Qwen2, Qwen2.5, Qwen3, Qwen3.5)
      deepseek.py               # model_type: "deepseek", "deepseek_v2", "deepseek_v3"
      llama.py                  # model_type: "llama", "mistral"
    cli.py                      # entrypoints
```

### Registry pattern

Dispatch key is `model.config.model_type` — always present in HuggingFace configs and reliable across model families, including cases like DeepSeek-R1-Distill-Qwen which uses `Qwen2Config` internally but has a distinct `model_type`.

```python
# registry.py
_REGISTRY: dict[str, type[TurboQuantAttentionBase]] = {}

def register_family(*model_types: str):
    """Decorator to register an attention class for one or more model_type strings."""
    def decorator(cls):
        for mt in model_types:
            _REGISTRY[mt] = cls
        return cls
    return decorator

def get_attention_class(model_type: str) -> type[TurboQuantAttentionBase]:
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

Adding a new family is one file + one decorator:

```python
# families/newmodel.py
@register_family("newmodel")
class TurboQuantNewModelAttention(TurboQuantAttentionBase):
    @classmethod
    def from_original(cls, orig, key_comp, val_comp): ...
    def forward(self, ...): ...
```

All family modules are imported at package init so their `@register_family` decorators fire automatically.

---

## Section 2: `apply_turboquant` API and attention replacement

### Public API

```python
from transformers_turboquant import apply_turboquant

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = apply_turboquant(model, bits=3)

# Works with pipeline:
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Works with SFTTrainer:
trainer = SFTTrainer(model=model, train_dataset=dataset)
trainer.train()
```

### Signature

```python
def apply_turboquant(
    model: PreTrainedModel,
    bits: int = 3,
    seed: int = 42,
    use_triton: bool = True,   # auto-falls back to PyTorch if Triton unavailable
) -> PreTrainedModel:
```

Returns the same model object (mutated in-place) for chaining.

### Implementation steps inside `apply_turboquant`

1. Resolve `device` from the model's first parameter
2. Look up `TurboQuantXxxAttention` via `get_attention_class(model.config.model_type)`
3. Walk `model.named_modules()`, identify attention layers by checking `type(module).__name__ in attention_cls.ATTENTION_CLASS_NAMES`
4. For each attention layer `i`: create `TurboQuantCompressorV2(head_dim, bits, seed=seed+i)` and `TurboQuantCompressorMSE(head_dim, bits, seed=seed+i+10000)`
5. Replace the module via `setattr` on its parent: `TurboQuantXxxAttention.from_original(orig, key_comp, val_comp)`
6. Return model

### Base class contract

```python
class TurboQuantAttentionBase(nn.Module):
    # Each family subclass declares which attention class names it replaces.
    # patch.py uses this to find attention layers while walking named_modules().
    ATTENTION_CLASS_NAMES: ClassVar[tuple[str, ...]]

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantAttentionBase":
        """Copy all weights from original, store compressors."""
        ...

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Must use:
          - key_compressor.compress(K) for key storage
          - key_compressor.asymmetric_attention_scores(Q, compressed_K) for scores
          - val_compressor.compress(V) then val_compressor.decompress(...) for values
        """
        ...
```

### Attention forward behavior

**Inference** (`model.generate()`, `pipeline`):
- K/V tensors compressed immediately after projection via `compress()` (runs under `torch.no_grad()`)
- Attention scores computed via `asymmetric_attention_scores(Q, compressed_K)` — uses the full TurboQuant two-stage estimator
- Output computed as `softmax(scores) @ decompress(compressed_V)`

**Training** (`SFTTrainer.train()`):
- Same forward pass — `compress()` is `@torch.no_grad()`, so KV memory is freed immediately
- Gradients flow through Q (and back to `q_proj` weights) normally
- KV projections receive gradients via the loss on the output token predictions, not through the compressed representation
- This is the standard behavior for activation-quantized fine-tuning and converges correctly for SFT

---

## Section 3: Tests, cleanup, and GitHub Actions

### Tests

**Fast tests** (no marker, always run in CI):

| File | What it tests |
|------|--------------|
| `tests/test_registry.py` | Registry dispatch, `UnsupportedModelError` on unknown `model_type` |
| `tests/test_apply_turboquant.py` | `apply_turboquant` on `sshleifer/tiny-gpt2`: model runs `generate()`, output shape correct |
| `tests/test_pipeline.py` | `pipeline("text-generation", model=apply_turboquant(...))` with tiny-gpt2 |
| `tests/test_sft.py` | Single `SFTTrainer` train step completes with patched tiny-gpt2 |

**Slow tests** (`@pytest.mark.slow`):

| File | Models tested |
|------|--------------|
| `tests/test_slow_models.py` | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen2.5-3B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |

### Cleanup

- **`validate.py`**: remove `sys.path.insert(...)`, replace with `from turboquant_pytorch import ...`
- **`pyproject.toml`**: add `turboquant_pytorch` as second installable module; add `validate` CLI script entry point
- **`cli.py`**: replace hello-world stub with a `validate` command that invokes the validation script
- **`justfile`**: add `validate` recipe: `uv run transformers-turboquant validate`

### GitHub Actions

**`.github/workflows/fast-tests.yml`** — triggers on push and PR:
```yaml
- run: uv sync --all-groups
- run: just test-fast
```

**`.github/workflows/publish.yml`** — triggers on tag `v*`, uses PyPI trusted publishing:
```yaml
- run: uv build
- uses: pypa/gh-action-pypi-publish@release/v1
```

**`ci.yml`** — ruff linting already present, no changes needed.

---

## Out of scope

- Weight quantization (bitsandbytes integration) — separate concern
- Differentiable quantization / QAT — not needed for SFT
- Support for fused QKV projections (e.g. some Falcon variants) — can be added via registry later
