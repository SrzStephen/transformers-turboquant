"""CLI entrypoint for transformers-turboquant."""

import typer

app = typer.Typer(
    help="HuggingFace integration for TurboQuant KV compression."
)


@app.command()
def validate() -> None:
    """Run needle-in-haystack validation on Qwen2.5-3B-Instruct (requires GPU)."""
    from turboquant_pytorch.validate import main

    main()


if __name__ == "__main__":
    app()
