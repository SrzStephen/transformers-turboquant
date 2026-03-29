"""SFTTrainer single-step training test with patched tiny-gpt2."""

import copy

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from transformers_turboquant import apply_turboquant

MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cant run without a GPU (bfloat16)"
)
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
        max_length=32,
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
