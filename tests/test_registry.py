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
