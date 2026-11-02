import json
import subprocess

import pytest

from pr_review.github.gh_client import GitHubClient, GitHubClientError
from pr_review.github.models import ReviewComment, ReviewResult, review_marker

PR_JSON = {
    "number": 7,
    "title": "Fix bug",
    "body": None,
    "state": "OPEN",
    "url": "https://github.com/o/r/pull/7",
    "author": {"login": "alice", "is_bot": False},
    "isDraft": False,
    "baseRefName": "main",
    "headRefName": "fix",
    "baseRefOid": "b" * 40,
    "headRefOid": "h" * 40,
    "additions": 3,
    "deletions": 1,
    "files": [{"path": "a.py", "additions": 3, "deletions": 1, "changeType": "MODIFIED"}],
}


class FakeRun:
    """Records every gh command and returns canned output."""

    def __init__(self, outputs, returncode=0, stderr=""):
        self.outputs = list(outputs)
        self.calls = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        out = self.outputs.pop(0) if self.outputs else ""
        return subprocess.CompletedProcess(command, self.returncode, out, self.stderr)


@pytest.fixture
def fake(monkeypatch):
    def install(*outputs, **kw):
        f = FakeRun(outputs, **kw)
        monkeypatch.setattr(subprocess, "run", f)
        return f

    return install


def test_get_pull_request_adds_repo_and_maps_aliases(fake):
    f = fake(json.dumps(PR_JSON))
    pr = GitHubClient("o/r").get_pull_request(7)
    assert pr.repository == "o/r" and pr.head_sha == "h" * 40 and pr.changed_lines == 4
    assert pr.files[0].change_type == "MODIFIED"
    command, kwargs = f.calls[0]
    assert command[:3] == ["gh", "pr", "view"] and command[-2:] == ["--repo", "o/r"]
    assert kwargs["encoding"] == "utf-8"  # Windows-safe decoding


def test_api_calls_do_not_get_repo_flag(fake):
    f = fake(json.dumps([[{"id": 1, "body": f"x {review_marker('abc')}"}]]))
    assert GitHubClient("o/r").has_reviewed(7, "abc")
    command, _ = f.calls[0]
    assert "--repo" not in command
    assert command[2] == "repos/o/r/pulls/7/reviews"


def test_post_review_payload(fake):
    f = fake(json.dumps({"id": 99}))
    review = ReviewResult(
        summary="2 issues",
        comments=[
            ReviewComment(
                path="a.py",
                line=3,
                body="Division by zero",
                severity="high",
                category="bug",
                suggested_code="return a / b if b else 0",
            )
        ],
    )
    GitHubClient("o/r").post_review(7, "h" * 40, review)
    command, kwargs = f.calls[0]
    payload = json.loads(kwargs["input"])
    assert payload["event"] == "COMMENT"  # never APPROVE / REQUEST_CHANGES
    assert payload["commit_id"] == "h" * 40
    assert review_marker("h" * 40) in payload["body"]
    c = payload["comments"][0]
    assert c["side"] == "RIGHT" and c["line"] == 3
    assert "```suggestion\nreturn a / b if b else 0\n```" in c["body"]


def test_errors_are_wrapped(fake):
    fake("", returncode=1, stderr="HTTP 404: Not Found")
    with pytest.raises(GitHubClientError, match="404"):
        GitHubClient("o/r").get_pull_request_diff(7)


def test_invalid_json(fake):
    fake("not json")
    with pytest.raises(GitHubClientError, match="invalid JSON"):
        GitHubClient("o/r").get_pull_request(7)


def test_missing_gh(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(GitHubClientError, match="not found"):
        GitHubClient("o/r").get_pull_request_diff(7)
