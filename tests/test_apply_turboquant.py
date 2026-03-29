import copy

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
    # Use a fresh copy so other tests aren't affected
    m = copy.deepcopy(model)
    from transformers_turboquant.families.gpt2 import TurboQuantGPT2Attention

    before = sum(
        1 for _, mod in m.named_modules() if type(mod).__name__ == "GPT2Attention"
    )
    apply_turboquant(m, bits=3)
    after = sum(
        1 for _, mod in m.named_modules() if isinstance(mod, TurboQuantGPT2Attention)
    )
    assert before > 0, "tiny-gpt2 should have GPT2Attention layers"
    assert after == before, f"expected {before} replacements, got {after}"


def test_apply_turboquant_generate_runs(tiny_gpt2):
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
    model, _ = tiny_gpt2
    m = copy.deepcopy(model)
    apply_turboquant(m, bits=3)
    assert m.config.use_cache is False


def test_apply_turboquant_returns_same_object(tiny_gpt2):
    model, _ = tiny_gpt2
    m = copy.deepcopy(model)
    result = apply_turboquant(m, bits=3)
    assert result is m
