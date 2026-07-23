from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GitHubUser(BaseModel):
    login: str


class ChangedFile(BaseModel):
    path: str
    status: str | None = None
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch: str | None = None


class PullRequestContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)    # added to keep an option of using aliases or original field names.

    number: int
    title: str
    body: str | None = None
    state: str
    repository: str
    author: GitHubUser
    base_ref_name: str = Field(alias = "baseRefName")   # aliasing done because github cli json fields are in camelCase
    head_ref_name: str = Field(alias = "headRefName")
    files: list[ChangedFile]

class ReviewComment(BaseModel):
    path: str
    line: int | None = None
    body: str
    severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    suggestion: str | None = None


class ReviewResult(BaseModel):
    summary: str
    comments: list[ReviewComment] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)