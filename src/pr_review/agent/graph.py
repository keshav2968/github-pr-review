"""Wire the nodes into a LangGraph state machine.

    fetch_context -> triage --(Send x N chunks, in parallel)--> review_chunk --> verify
                       \\--(nothing to review)----------------------------------^
    verify -> finalize --(dry run)--> END
                       \\--(publish)--> publish -> END

Why a graph and not one big prompt?
- Each step has one job and its own prompt, so each can be tested and tuned.
- Branching is explicit: skip the LLM when there's nothing to review; publish
  only when asked.
- Map-reduce: big PRs are split into chunks reviewed in parallel (`Send`), and
  the results are merged by the `candidates` reducer.
- Checkpointing: state is saved after every node, so runs can be resumed, and a
  dry-run review can be published later without paying for the LLM again.
- Every node is a separate span in LangSmith, so a bad comment can be traced to
  the exact step and prompt that produced it.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from pr_review.agent.deps import AgentDeps
from pr_review.agent.nodes import Nodes
from pr_review.agent.schemas import Candidate, Issue
from pr_review.agent.state import ReviewState
from pr_review.agent.triage import Chunk
from pr_review.github.diff import FileDiff
from pr_review.github.models import (
    ChangedFile,
    GitHubUser,
    PullRequestContext,
    ReviewComment,
    ReviewResult,
)

# Only these classes may be rebuilt when loading a checkpoint. A checkpoint DB is
# data on disk; letting it instantiate *any* class would be a code-execution risk.
CHECKPOINT_TYPES = [
    PullRequestContext, GitHubUser, ChangedFile, FileDiff, Chunk,
    Candidate, Issue, ReviewResult, ReviewComment,
]  # fmt: skip


def _fan_out(state: ReviewState) -> list[Send] | str:
    chunks = state.get("chunks", [])
    if not chunks:
        return "verify"
    return [
        Send("review_chunk", {"chunk": c, "pr": state["pr"], "focus": state.get("focus", "")})
        for c in chunks
    ]


def _after_finalize(state: ReviewState) -> str:
    return END if state.get("dry_run", True) else "publish"


def build_graph(deps: AgentDeps, checkpointer: SqliteSaver | None = None):
    nodes = Nodes(deps)
    g = StateGraph(ReviewState)
    g.add_node("fetch_context", nodes.fetch_context)
    g.add_node("triage", nodes.triage)
    g.add_node("review_chunk", nodes.review_chunk)
    g.add_node("verify", nodes.verify)
    g.add_node("finalize", nodes.finalize)
    g.add_node("publish", nodes.publish)

    g.add_edge(START, "fetch_context")
    g.add_edge("fetch_context", "triage")
    g.add_conditional_edges("triage", _fan_out, ["review_chunk", "verify"])
    g.add_edge("review_chunk", "verify")
    g.add_edge("verify", "finalize")
    g.add_conditional_edges("finalize", _after_finalize, ["publish", END])
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)


@contextmanager
def open_checkpointer(db_path: str) -> Iterator[SqliteSaver]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
        yield SqliteSaver(conn, serde=serde)
    finally:
        conn.close()
