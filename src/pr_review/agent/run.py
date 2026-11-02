"""Run one review end to end. Shared by the CLI, the webhook server and the evals.

Checkpoint behaviour (thread id = "owner/repo#PR@head-sha"):
- same commit, finished before   -> return the saved result (no LLM calls, no cost)
- same commit, crashed half-way  -> resume from the last finished node
- finished as dry-run, now --publish -> publish the saved review without re-reviewing
- `fresh=True` ignores any saved state
A new push gives a new head SHA, so it automatically gets a new review.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pr_review.agent.deps import AgentDeps, GitHubPort, LLMPort
from pr_review.agent.graph import build_graph, open_checkpointer
from pr_review.config import Settings

Status = Literal["new", "cached", "resumed", "published-from-cache"]


@dataclass
class RunOutcome:
    state: dict[str, Any]
    status: Status
    thread_id: str
    run_id: str
    llm_stats: dict[str, float] = field(default_factory=dict)
    trace_url: str | None = None


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")


def trace_url(run_id: str) -> str | None:
    if not tracing_enabled():
        return None
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()  # traces upload in the background; flush first
        return Client().read_run(run_id).url
    except Exception:
        return None


def run_review(
    *,
    settings: Settings,
    github: GitHubPort,
    llm: LLMPort,
    pr_number: int,
    focus: str,
    dry_run: bool = True,
    fresh: bool = False,
    tags: list[str] | None = None,
) -> RunOutcome:
    pr = github.get_pull_request(pr_number)
    thread_id = f"{pr.repository}#{pr.number}@{pr.head_sha[:12]}"
    if fresh:
        thread_id += f"/{uuid.uuid4().hex[:8]}"
    run_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": settings.llm_concurrency,
        "recursion_limit": 50,
        # These show up in LangSmith, so traces can be filtered by repo/PR/model.
        "run_id": run_id,
        "run_name": "pr-review",
        "tags": ["pr-review", *(tags or [])],
        "metadata": {
            "repository": pr.repository,
            "pr_number": pr.number,
            "head_sha": pr.head_sha,
            "model": settings.llm_model,
            "min_confidence": settings.min_confidence,
        },
    }
    inputs = {"repository": pr.repository, "pr_number": pr_number, "focus": focus,
              "dry_run": dry_run}  # fmt: skip

    with open_checkpointer(settings.checkpoint_db) as saver:
        graph = build_graph(AgentDeps(github=github, llm=llm, settings=settings), saver)
        saved = graph.get_state(config)
        finished = bool(saved.values.get("review")) and not saved.next

        if finished and (dry_run or saved.values.get("published") is not None):
            status: Status = "cached"
            state = saved.values
        elif finished:
            # Pretend `finalize` just ran with dry_run=False -> graph continues to publish.
            graph.update_state(config, {"dry_run": False}, as_node="finalize")
            state, status = graph.invoke(None, config), "published-from-cache"
        elif saved.next:
            state, status = graph.invoke(None, config), "resumed"
        else:
            state, status = graph.invoke(inputs, config), "new"

    stats = getattr(llm, "stats", None)
    return RunOutcome(
        state=state,
        status=status,
        thread_id=thread_id,
        run_id=run_id,
        llm_stats=stats.as_dict() if stats is not None else {},
        trace_url=trace_url(run_id) if status != "cached" else None,
    )
