from pydantic import BaseModel


class GitHubUser(BaseModel):
    login: str


class PullRequestFile(BaseModel):
    path: str
    additions: int
    deletions: int
    changes: int


class PullRequest(BaseModel):
    number: int
    title: str
    body: str | None
    state: str
    author: GitHubUser
    base_ref_name: str
    head_ref_name: str
    files: list[PullRequestFile]