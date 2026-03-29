"""apply_turboquant: replace attention layers in a PreTrainedModel."""

from transformers import PreTrainedModel

from turboquant_pytorch.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)

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
                head_dim,
                bits,
                seed=seed + i,
                device=device_str,
                use_triton=use_triton,
            )
            val_comp = TurboQuantCompressorMSE(
                head_dim,
                bits,
                seed=seed + i + 10000,
                device=device_str,
                use_triton=use_triton,
            )
            new_attn = attention_cls.from_original(child_module, key_comp, val_comp)
            setattr(parent_module, child_name, new_attn)
            i += 1

    # Disable incremental KV caching — our forward does full-sequence attention
    model.config.use_cache = False
    return model
