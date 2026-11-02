"""The single object that flows through the graph.

LangGraph passes this dict from node to node. Each node returns only the keys
it changed, and LangGraph merges them in. `Annotated[list, operator.add]` is a
*reducer*: when several parallel `review_chunk` nodes all return `candidates`,
their lists are concatenated instead of overwriting each other.

Because the state is checkpointed to SQLite after every node, a crashed run can
resume from the last finished step instead of paying for LLM calls again.
"""

import operator
from typing import Annotated, TypedDict

from pr_review.agent.schemas import Candidate
from pr_review.agent.triage import Chunk
from pr_review.github.diff import FileDiff
from pr_review.github.models import PullRequestContext, ReviewResult


class ReviewState(TypedDict, total=False):
    # inputs
    repository: str
    pr_number: int
    focus: str
    dry_run: bool
    # fetch_context / triage
    pr: PullRequestContext
    files: dict[str, FileDiff]
    skipped: dict[str, str]  # path -> reason
    chunks: list[Chunk]
    # review (filled by parallel branches, merged by the reducer)
    candidates: Annotated[list[Candidate], operator.add]
    errors: Annotated[list[str], operator.add]
    # verify
    verified: list[Candidate]  # every candidate, each with a verdict
    # finalize
    drops: dict[str, int]  # why issues were filtered out (for metrics)
    review: ReviewResult
    incomplete: bool  # every review chunk failed -> never publish
    # publish
    published: bool
    publish_note: str


class ChunkTask(TypedDict):
    """Private input for one parallel review branch (sent with LangGraph's Send)."""

    chunk: Chunk
    pr: PullRequestContext
    focus: str
