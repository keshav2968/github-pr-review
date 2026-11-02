"""Thin wrapper around the GitHub CLI (`gh`).

Why shell out to `gh` instead of calling the REST API with a token?
- Auth is already solved: `gh auth login` stores credentials securely (OS keyring),
  so this project never handles a raw token during local use.
- `gh` handles pagination, rate-limit headers and GitHub Enterprise hosts for us.
The trade-off is a subprocess per call (~100ms), which is fine for a PR reviewer.
"""

import json
import subprocess
from typing import Any
from urllib.parse import quote

from pr_review.github.models import ExistingReview, PullRequestContext, ReviewResult, review_marker

PR_FIELDS = (
    "number,title,body,state,url,author,isDraft,baseRefName,headRefName,"
    "baseRefOid,headRefOid,additions,deletions,files"
)


class GitHubClientError(RuntimeError):
    """Raised when a GitHub CLI command fails."""


class GitHubClient:
    def __init__(self, repository: str | None = None, timeout_s: float = 60) -> None:
        self._repository = repository
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ reads
    @property
    def repository(self) -> str:
        """`owner/repo`. If not given, ask gh which repo the current folder belongs to."""
        if self._repository is None:
            data = self._run_json("repo", "view", "--json", "nameWithOwner")
            self._repository = data["nameWithOwner"]
        return self._repository

    def get_pull_request(self, pr_number: int) -> PullRequestContext:
        data = self._run_json("pr", "view", str(pr_number), "--json", PR_FIELDS, repo_flag=True)
        if not isinstance(data, dict):
            raise GitHubClientError("Expected a JSON object from `gh pr view`.")
        data["repository"] = self.repository
        return PullRequestContext.model_validate(data)

    def get_pull_request_diff(self, pr_number: int) -> str:
        return self._run_text("pr", "diff", str(pr_number), repo_flag=True)

    def get_file(self, path: str, ref: str) -> str:
        """Full content of a file at a commit (gives the agent context beyond the diff)."""
        return self._run_text(
            "api",
            f"repos/{self.repository}/contents/{quote(path)}?ref={ref}",
            "-H",
            "Accept: application/vnd.github.raw",
        )

    def list_reviews(self, pr_number: int) -> list[ExistingReview]:
        pages = self._run_json(
            "api", f"repos/{self.repository}/pulls/{pr_number}/reviews", "--paginate", "--slurp"
        )
        return [ExistingReview.model_validate(r) for page in pages for r in page]

    # ----------------------------------------------------------------- writes
    def post_review(self, pr_number: int, head_sha: str, review: ReviewResult) -> dict[str, Any]:
        """Post ONE review containing all inline comments.

        - event is always COMMENT: the bot can never approve or block a PR.
        - commit_id pins comments to the commit we reviewed, so line numbers stay
          correct even if the author pushes again while we're running.
        - the hidden marker lets us skip re-posting for the same commit.
        """
        payload = {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": f"{review.summary}\n\n{review_marker(head_sha)}",
            "comments": [c.to_github() for c in review.comments],
        }
        return self._run_json(
            "api",
            f"repos/{self.repository}/pulls/{pr_number}/reviews",
            "--method",
            "POST",
            "--input",
            "-",
            stdin=json.dumps(payload),
        )

    def has_reviewed(self, pr_number: int, head_sha: str) -> bool:
        return any(r.is_ours_for(head_sha) for r in self.list_reviews(pr_number))

    # ---------------------------------------------------------------- plumbing
    def _run_json(self, *arguments: str, repo_flag: bool = False, stdin: str | None = None) -> Any:
        output = self._run_text(*arguments, repo_flag=repo_flag, stdin=stdin)
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise GitHubClientError(
                f"GitHub CLI returned invalid JSON for: gh {arguments[0]}"
            ) from error

    def _run_text(self, *arguments: str, repo_flag: bool = False, stdin: str | None = None) -> str:
        command = ["gh", *arguments]
        # Only `gh pr ...` accepts --repo. `gh api` takes the repo inside the URL path.
        if repo_flag and self._repository is not None:
            command.extend(["--repo", self._repository])

        try:
            result = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                # Windows defaults to cp1252; diffs often contain UTF-8 (emoji, non-English).
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as error:
            raise GitHubClientError(
                "GitHub CLI was not found. Install gh and ensure it is on PATH."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise GitHubClientError(
                f"gh {arguments[0]} timed out after {self.timeout_s}s"
            ) from error

        if result.returncode != 0:
            message = (result.stderr or "").strip() or "GitHub CLI command failed."
            raise GitHubClientError(message)

        return result.stdout
