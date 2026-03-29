#!/bin/bash
set -euo pipefail

if [ -f pyproject.toml ]; then
  echo "pyproject.toml found, running uv sync..."
  uv sync --all-extras
else
  echo "No pyproject.toml found, skipping uv sync."
fi

if [ -f .pre-commit-config.yaml ]; then
  if uv pip show pre-commit &>/dev/null; then
    echo "pre-commit config and package found, installing hooks..."
    uv run pre-commit install
  else
    echo "Skipping pre-commit install (pre-commit package not installed)."
  fi
else
  echo "Skipping pre-commit install (no .pre-commit-config.yaml found)."
fi
