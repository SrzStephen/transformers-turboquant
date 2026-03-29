"""Tests for transformers_turboquant CLI."""

from typer.testing import CliRunner

from transformers_turboquant.cli import app

runner = CliRunner()


def test_main_command() -> None:
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Hello from transformers-turboquant!" in result.output
