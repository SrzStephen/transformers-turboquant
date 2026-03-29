"""CLI entrypoint for transformers-turboquant."""

import typer

app = typer.Typer(help="")


@app.command()
def main() -> None:
    """Run transformers-turboquant."""
    typer.echo("Hello from transformers-turboquant!")


if __name__ == "__main__":
    app()
