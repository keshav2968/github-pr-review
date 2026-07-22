import json
import subprocess
from typing import Any

from pr_review.github.models import PullRequest


class GitHubClientError(RuntimeError):
    """Raised when a GitHub CLI command fails."""


class GitHubClient:
    def __init__(self, repository: str | None = None) -> None:
        self.repository = repository

    def get_pull_request(self, pr_number: int) -> PullRequest:
        data = self._run_json(
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,body,state,author,baseRefName,headRefName,files",
        )
        return PullRequest.model_validate(data)

    def get_pull_request_diff(self, pr_number: int) -> str:
        return self._run_text("pr", "diff", str(pr_number))

    def _run_json(self, *arguments: str) -> dict[str, Any]:
        output = self._run_text(*arguments)

        try:
            data = json.loads(output)
        except json.JSONDecodeError as error:
            raise GitHubClientError(
                "GitHub CLI returned invalid JSON."
            ) from error

        if not isinstance(data, dict):
            raise GitHubClientError("Expected a JSON object from GitHub CLI.")

        return data

    def _run_text(self, *arguments: str) -> str:
        command = ["gh", *arguments]

        if self.repository is not None:
            command.extend(["--repo", self.repository])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise GitHubClientError(
                "GitHub CLI was not found. Install gh and ensure it is on PATH."
            ) from error

        if result.returncode != 0:
            message = result.stderr.strip() or "GitHub CLI command failed."
            raise GitHubClientError(message)

        return result.stdout
