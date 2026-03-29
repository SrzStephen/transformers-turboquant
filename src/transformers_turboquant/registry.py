"""Registry mapping model_type strings to TurboQuantAttentionBase subclasses."""

from .base import TurboQuantAttentionBase

_REGISTRY: dict[str, type[TurboQuantAttentionBase]] = {}


class UnsupportedModelError(ValueError):
    """Raised when apply_turboquant encounters an unregistered model_type."""


def register_family(*model_types: str):
    """
    Class decorator that registers an attention class for one or more
    model_type strings.

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
