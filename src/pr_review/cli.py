import logging
import shutil
import subprocess
import sys

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pr_review.config import get_settings
from pr_review.github.gh_client import GitHubClient, GitHubClientError

app = typer.Typer(help="AI-powered GitHub PR review agent.", no_args_is_help=True)
console = Console()

DEFAULT_FOCUS = "Review this PR for correctness, bugs, security problems and missing tests."


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs.")) -> None:
    # Load .env into os.environ so libraries that read env vars directly
    # (e.g. langsmith reads LANGSMITH_API_KEY) see the same values as Settings.
    load_dotenv()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def review(
    pr_number: int = typer.Argument(..., help="GitHub pull request number."),
    repo: str | None = typer.Option(None, "--repo", help="owner/repo (default: current folder)."),
    focus: str = typer.Option(
        DEFAULT_FOCUS, "--focus", "--request", "-r", help="What the reviewer should focus on."
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--publish", help="Print results locally instead of posting to GitHub."
    ),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore any saved result for this commit."),
) -> None:
    """Review a pull request (dry run by default)."""
    from pr_review.agent.run import run_review
    from pr_review.llm.structured import StructuredLLM

    settings = get_settings()
    if settings.llm_api_key is None:
        console.print("[red]LLM_API_KEY is not set.[/red] Run `pr-review doctor` for setup help.")
        raise typer.Exit(1)
    github = GitHubClient(repository=repo or settings.github_repository)
    try:
        with console.status("Reviewing..."):
            outcome = run_review(
                settings=settings,
                github=github,
                llm=StructuredLLM(settings),
                pr_number=pr_number,
                focus=focus,
                dry_run=dry_run,
                fresh=fresh,
            )
    except GitHubClientError as error:
        console.print(f"[red]GitHub error:[/red] {error}")
        raise typer.Exit(1) from error

    state = outcome.state
    pr, result = state["pr"], state["review"]
    console.print(
        f"[bold]{pr.repository}#{pr.number}[/bold] {pr.title}  [dim]({outcome.status})[/dim]"
    )
    console.print(result.summary)
    for c in result.comments:
        console.print(Panel(c.render(), title=f"{c.path}:{c.line}", title_align="left"))

    table = Table("metric", "value", title="run details")
    table.add_row("candidate issues", str(len(state.get("candidates", []))))
    table.add_row("posted comments", str(len(result.comments)))
    for reason, n in (state.get("drops") or {}).items():
        table.add_row(f"  filtered: {reason}", str(n))
    for key in ("calls", "schema_valid_rate", "repairs", "retries", "prompt_tokens", "latency_s"):
        if key in outcome.llm_stats:
            table.add_row(f"llm {key}", f"{outcome.llm_stats[key]:.2f}".rstrip("0").rstrip("."))
    for error in state.get("errors", []):
        table.add_row("[yellow]error[/yellow]", error[:120])
    if not dry_run:
        table.add_row("publish", state.get("publish_note", ""))
    table.add_row("thread", outcome.thread_id)
    table.add_row("langsmith", outcome.trace_url or f"run id {outcome.run_id}")
    console.print(table)


def _check(cmd: list[str]) -> tuple[bool, str]:
    if shutil.which(cmd[0]) is None:
        return False, f"`{cmd[0]}` not found on PATH"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    msg = (r.stdout or r.stderr).strip().splitlines()
    return r.returncode == 0, msg[0] if msg else ""


def _check_llm() -> tuple[bool, str]:
    """One tiny real structured call: proves key, model name and JSON schema all work."""
    from pydantic import BaseModel

    from pr_review.llm.structured import StructuredLLM

    class Ping(BaseModel):
        answer: int

    llm = StructuredLLM(get_settings())
    try:
        out = llm.generate([{"role": "user", "content": "What is 2 + 3?"}], Ping, name="doctor")
    except Exception as error:
        return False, str(error).splitlines()[0][:80]
    mode = "schema mode" if llm.schema_supported else "JSON-mode fallback"
    return out.answer == 5, f"2+3={out.answer} ({mode})"


@app.command()
def doctor(
    live: bool = typer.Option(False, "--live", help="Also make one real (free) LLM call."),
) -> None:
    """Check that everything this project needs is installed and configured."""
    s = get_settings()
    checks: list[tuple[str, bool, str, str]] = []  # name, ok, detail, how to fix

    py_ok = sys.version_info >= (3, 12)
    checks.append(("python >= 3.12", py_ok, sys.version.split()[0], "install Python 3.12+"))

    ok, detail = _check(["gh", "auth", "status"])
    checks.append(("gh logged in", ok, detail, "run: gh auth login"))

    ok, detail = _check(["docker", "info", "--format", "{{.ServerVersion}}"])
    checks.append(("docker running", ok, detail, "start Docker Desktop (needed for test sandbox)"))

    key_ok = s.llm_api_key is not None
    checks.append(
        (
            "LLM_API_KEY set",
            key_ok,
            s.llm_model,
            "get a key at https://aistudio.google.com and put it in .env",
        )
    )
    if live and key_ok:
        ok, detail = _check_llm()
        checks.append(
            ("LLM structured call", ok, detail, "check LLM_MODEL is a model your key can use")
        )
    checks.append(
        (
            "LANGSMITH_API_KEY set",
            s.langsmith_api_key is not None,
            s.langsmith_project,
            "create one at https://smith.langchain.com -> Settings -> API Keys",
        )
    )

    table = Table("check", "status", "detail", "fix")
    for name, ok, detail, fix in checks:
        table.add_row(
            name, "[green]ok[/green]" if ok else "[red]FAIL[/red]", detail, "" if ok else fix
        )
    console.print(table)
    if not all(ok for _, ok, _, _ in checks):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
