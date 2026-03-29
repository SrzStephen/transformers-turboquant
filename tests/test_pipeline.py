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
