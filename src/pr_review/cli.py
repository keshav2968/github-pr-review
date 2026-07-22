import typer

app = typer.Typer(help="AI-powered GitHub PR review agent.")


@app.command()
def review(
    pr_number: int = typer.Argument(..., help="GitHub pull request number."),
    request: str = typer.Option(
        "Review this PR for correctness, bugs, and missing tests.",
        "--requests",
        "-r",
        help="Natural language review instruction.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--publish",
        help="Print results locally instead of posting to GitHub.",
    ),
) -> None:
    typer.echo(f"PR: #{pr_number}")
    typer.echo(f"Request: {request}")
    typer.echo(f"Mode: {'dry-run' if dry_run else 'publish'}")


if __name__ == "__request__":
    app()
