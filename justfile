default: all

all: lint typecheck test

lint:
    uvx ruff check .
    uvx ruff format --check .

lint-fix:
    uvx ruff check --fix .
    uvx ruff format .

typecheck:
    uv run ty check src/

test:
    uv run pytest

test-fast:
    uv run pytest -n auto --no-cov

test-slow:
    uv run pytest -m slow -s

validate:
    uv run transformers-turboquant validate

pre-commit:
    uv run pre-commit run --all-files

setup:
    uv sync --all-groups
    uv run pre-commit install
