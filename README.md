# transformers-turboquant

[![CI](https://github.com/SrzStephen/transformers-turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/SrzStephen/transformers-turboquant/actions/workflows/ci.yml)

HuggingFace Transformers integration for **TurboQuant** KV cache compression, based on the paper ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)](https://arxiv.org/abs/2410.06088).

The `turboquant_pytorch` package is derived from the [original CPU based Python implementation](https://github.com/tonbistudio/turboquant-pytorch). This repository contains `Triton` kernels to support GPU based inference.

TurboQuant compresses the KV cache of attention layers using a two-stage vector quantization scheme:

- **Stage 1 (MSE):** Random rotation + per-coordinate Lloyd-Max quantization
- **Stage 2 (QJL):** 1-bit Quantized Johnson-Lindenstrauss projection on residuals for unbiased inner product estimation

Attention scores are computed **directly from the compressed representation** using an asymmetric inner product estimator, avoiding full decompression of keys.

## Supported Models

| Family          | `model_type` values |
| --------------- | ------------------- |
| LLaMA / Mistral | `llama`             |
| Qwen2           | `qwen2`             |
| DeepSeek        | `deepseek`          |
| GPT-2           | `gpt2`              |

## Usage

### Python API

```python
from transformers import AutoModelForCausalLM
from transformers_turboquant import apply_turboquant

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Replace all attention layers with TurboQuant compressed versions
model = apply_turboquant(model, bits=3, seed=42, use_triton=True)
```

`apply_turboquant` mutates the model in-place and returns it. `bits` can be 2, 3, or 4.

> **Note:** `use_cache` is set to `False` on the model config because TurboQuant computes full-sequence compressed attention on every forward call.

### CLI

```bash
# Run needle-in-haystack validation (requires GPU, downloads Qwen2.5-3B-Instruct)
transformers-turboquant validate

# Or via the standalone entrypoint
validate
```

The validation runs a needle-in-haystack benchmark at 2048, 4096, and 8192 token contexts, comparing TurboQuant attention scores against ground-truth for 2-, 3-, and 4-bit quantization. It reports compression ratio, score cosine similarity, top-1/top-5 attention head match rate, and average needle rank.

## Adding a New Model Family

1. Create `src/transformers_turboquant/families/<family>.py`
2. Subclass `TurboQuantAttentionBase` and implement `from_original()` and `forward()`
3. Decorate with `@register_family("model_type_string")`
4. Import it in `src/transformers_turboquant/families/__init__.py`

```python
from transformers_turboquant.registry import register_family
from transformers_turboquant.base import TurboQuantAttentionBase

@register_family("my_model")
class TurboQuantMyModelAttention(TurboQuantAttentionBase):
    ATTENTION_CLASS_NAMES = ("MyModelAttention",)

    @classmethod
    def from_original(cls, original, key_compressor, val_compressor):
        ...

    def forward(self, hidden_states, **kwargs):
        ...
```

## Development

Common tasks are managed via [just](https://github.com/casey/just):

| Command           | Description                       |
| ----------------- | --------------------------------- |
| `just`            | Run lint, typecheck, and tests    |
| `just lint`       | Lint and check formatting         |
| `just lint-fix`   | Auto-fix lint and formatting      |
| `just typecheck`  | Run type checking                 |
| `just test`       | Run tests                         |
| `just test-fast`  | Run tests in parallel (no cov)    |
| `just pre-commit` | Run pre-commit on all files       |
| `just setup`      | Install deps and pre-commit hooks |

## Setup

This repository is set up to run out of a [devcontainer](https://code.visualstudio.com/docs/devcontainers/create-dev-container) via Visual Studio Code.

### Python

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Run `uv sync --all-groups` to install dependencies

### Pre-commit

Pre-commit is installed as a dev dependency. Register it with:

```bash
uv run pre-commit install
```

Run ad hoc with `uv run pre-commit`.

This repository enforces [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope]: <description>
```

Examples:

```
feat: add Mistral attention family
fix: correct QJL correction scale factor
feat(triton)!: switch keys to bit-packed uint8 storage
```

### Just

Install [just](https://github.com/casey/just?tab=readme-ov-file#installation).

### CUDA

For CUDA/Triton kernel support, install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html). On Windows, CUDA requires Docker with WSL2. When CUDA is unavailable, the library falls back to pure PyTorch implementations automatically.
