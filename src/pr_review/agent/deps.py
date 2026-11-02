"""What the agent needs from the outside world, described as interfaces.

The graph never imports `GitHubClient` or the real LLM directly. It receives
"something with these methods". That is dependency injection, and it pays off
three times:
- tests pass in fakes, so the whole graph is tested with no network and no API key;
- step 3 swaps the direct GitHub client for MCP tools without touching any node;
- the eval harness can wrap the dependencies to record extra data.
"""

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from pr_review.config import Settings
from pr_review.github.models import PullRequestContext, ReviewResult

T = TypeVar("T", bound=BaseModel)


class GitHubPort(Protocol):
    def get_pull_request(self, pr_number: int) -> PullRequestContext: ...
    def get_pull_request_diff(self, pr_number: int) -> str: ...
    def get_file(self, path: str, ref: str) -> str: ...
    def has_reviewed(self, pr_number: int, head_sha: str) -> bool: ...
    def post_review(
        self, pr_number: int, head_sha: str, review: ReviewResult
    ) -> dict[str, Any]: ...


class LLMPort(Protocol):
    def generate(self, messages: list[dict[str, str]], schema: type[T], *, name: str = "") -> T: ...


@dataclass
class AgentDeps:
    github: GitHubPort
    llm: LLMPort
    settings: Settings
