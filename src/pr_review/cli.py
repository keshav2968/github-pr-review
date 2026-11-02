import shutil
import subprocess
import sys

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from pr_review.config import get_settings
from pr_review.github.diff import parse_diff
from pr_review.github.gh_client import GitHubClient, GitHubClientError

app = typer.Typer(help="AI-powered GitHub PR review agent.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    # Load .env into os.environ so libraries that read env vars directly
    # (e.g. langsmith reads LANGSMITH_API_KEY) see the same values as Settings.
    load_dotenv()


@app.command()
def review(
    pr_number: int = typer.Argument(..., help="GitHub pull request number."),
    repo: str | None = typer.Option(None, "--repo", help="owner/repo (default: current folder)."),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--publish",
        help="Print results locally instead of posting to GitHub.",
    ),
) -> None:
    """Review a pull request."""
    client = GitHubClient(repository=repo or get_settings().github_repository)
    try:
        pr = client.get_pull_request(pr_number)
        files = parse_diff(client.get_pull_request_diff(pr_number))
    except GitHubClientError as error:
        console.print(f"[red]GitHub error:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[bold]{pr.repository}#{pr.number}[/bold] {pr.title}")
    console.print(
        f"head {pr.head_sha[:8]} -> base {pr.base_sha[:8]} | +{pr.additions} -{pr.deletions}"
    )
    table = Table("file", "added", "commentable lines", "flags")
    for fd in files.values():
        flags = ",".join(
            name
            for name, on in [
                ("new", fd.is_new),
                ("deleted", fd.is_deleted),
                ("renamed", fd.is_renamed),
                ("binary", fd.is_binary),
            ]
            if on
        )
        table.add_row(fd.path, str(fd.additions), str(len(fd.commentable_lines)), flags)
    console.print(table)
    console.print(f"Mode: {'dry-run' if dry_run else 'publish'} (agent arrives in step 2)")


def _check(cmd: list[str]) -> tuple[bool, str]:
    if shutil.which(cmd[0]) is None:
        return False, f"`{cmd[0]}` not found on PATH"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    msg = (r.stdout or r.stderr).strip().splitlines()
    return r.returncode == 0, msg[0] if msg else ""


@app.command()
def doctor() -> None:
    """Check that everything this project needs is installed and configured."""
    s = get_settings()
    checks: list[tuple[str, bool, str, str]] = []  # name, ok, detail, how to fix

    py_ok = sys.version_info >= (3, 12)
    checks.append(("python >= 3.12", py_ok, sys.version.split()[0], "install Python 3.12+"))

    ok, detail = _check(["gh", "auth", "status"])
    checks.append(("gh logged in", ok, detail, "run: gh auth login"))

    ok, detail = _check(["docker", "info", "--format", "{{.ServerVersion}}"])
    checks.append(("docker running", ok, detail, "start Docker Desktop (needed for test sandbox)"))

    checks.append(
        (
            "LLM_API_KEY set",
            s.llm_api_key is not None,
            s.llm_model,
            "get a key at https://aistudio.google.com and put it in .env",
        )
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
