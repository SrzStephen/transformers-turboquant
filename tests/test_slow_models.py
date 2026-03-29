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
