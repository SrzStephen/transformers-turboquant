# New Task

## Implementation

Create a way to use the turboquant cpu and gpu implementations easily with the `transformers` library.
This can be a wrapper or another method but at minimum it should support

- transformers `pipeline` (`from transformers import pipeline`)
- AutoModelForCausalLM `from transformers import AutoModelForCausalLM,`
- SFT Training with trl

## Trl training example

Example of SFT training without turboquant

```
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

## tests

Write tests to ensure that it works, slow tests should use the `marker` test.
Make sure to test at least `sshleifer/tiny-gpt2`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen2.5-3B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

# Cleanup tasks

- Remove any path hacking such as `sys.path.insert` and instead make sure to use uv modules

## UV module example

```python
from agents import some_function
```

```zsh
.
├── src/
│   ├── agents/
│   │   └── __init__.py
│   └── client/
│       └── __init__.py
├── test/
└── pyproject.toml
```

```yaml
[build-system]
requires = ["uv_build>=0.10.0,<0.11.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = ["agents", "client"]
module-root = "src"
```

- Add CLI entrypoints where they make sense for the `pyproject.toml`
    - Also add these to the `justfile`

# Github Action

- Make a github action to run the fast tests, use the `fast` justfile
- Make a github action to publish to pypi
- Make the github action check for ruff linting
