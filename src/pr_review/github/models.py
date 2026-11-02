"""Data shapes for GitHub objects and for the review we post back.

Pydantic validates data the moment it enters our program, so a missing or
renamed field from GitHub fails loudly here instead of deep inside the agent.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "low", "medium", "high", "critical"]

# Hidden HTML comment we put in every review body. GitHub doesn't render it, but we
# can search for it later to know "we already reviewed this exact commit" (idempotency).
MARKER_PREFIX = "<!-- pr-review:sha="


def review_marker(head_sha: str) -> str:
    return f"{MARKER_PREFIX}{head_sha} -->"


class GitHubModel(BaseModel):
    # GitHub CLI returns camelCase keys (baseRefName). We use snake_case in Python
    # and map with aliases; populate_by_name lets tests build models with either.
    model_config = ConfigDict(populate_by_name=True)


class GitHubUser(GitHubModel):
    login: str
    is_bot: bool = False


class ChangedFile(GitHubModel):
    path: str
    additions: int = 0
    deletions: int = 0
    change_type: str | None = Field(default=None, alias="changeType")


class PullRequestContext(GitHubModel):
    number: int
    title: str
    body: str | None = None
    state: str
    url: str | None = None
    repository: str  # "owner/repo" (added by our client; gh doesn't return it)
    author: GitHubUser
    is_draft: bool = Field(default=False, alias="isDraft")
    base_ref_name: str = Field(alias="baseRefName")
    head_ref_name: str = Field(alias="headRefName")
    base_sha: str = Field(alias="baseRefOid")
    head_sha: str = Field(alias="headRefOid")
    additions: int = 0
    deletions: int = 0
    files: list[ChangedFile] = Field(default_factory=list)

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions


class ReviewComment(BaseModel):
    """One inline comment, anchored to a line of the *new* version of a file."""

    path: str
    line: int = Field(ge=1)
    body: str
    severity: Severity = "medium"
    category: str | None = None
    # Exact replacement code for the commented line. Rendered as a GitHub
    # ```suggestion block so the author can apply it with one click.
    suggested_code: str | None = None

    def render(self) -> str:
        label = f"**[{self.severity}{' · ' + self.category if self.category else ''}]**"
        text = f"{label} {self.body}"
        if self.suggested_code is not None:
            text += f"\n\n```suggestion\n{self.suggested_code}\n```"
        return text

    def to_github(self) -> dict:
        # side=RIGHT means "the new version of the file" in GitHub's review API.
        return {"path": self.path, "line": self.line, "side": "RIGHT", "body": self.render()}


class ReviewResult(BaseModel):
    summary: str
    comments: list[ReviewComment] = Field(default_factory=list)


class ExistingReview(BaseModel):
    """A review already on the PR (we only need enough to detect our own)."""

    id: int
    body: str | None = ""
    commit_id: str | None = None
    user: GitHubUser | None = None

    def is_ours_for(self, head_sha: str) -> bool:
        return review_marker(head_sha) in (self.body or "")
