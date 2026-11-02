import pytest

from pr_review.agent.deps import AgentDeps
from pr_review.agent.graph import build_graph, open_checkpointer
from pr_review.config import Settings
from tests.fakes import FakeGitHub, FakeLLM, issue


def run(llm, github=None, dry_run=True, checkpointer=None, config=None, **settings):
    github = github or FakeGitHub()
    deps = AgentDeps(github=github, llm=llm, settings=Settings(_env_file=None, **settings))
    graph = build_graph(deps, checkpointer)
    inputs = {"repository": "o/r", "pr_number": 7, "focus": "bugs", "dry_run": dry_run}
    return graph.invoke(inputs, config or {}), github, graph


def test_happy_path_dry_run():
    state, github, _ = run(FakeLLM())
    review = state["review"]
    assert [(c.path, c.line) for c in review.comments] == [("src/app/new_mod.py", 2)]
    assert review.comments[0].severity == "high"
    assert github.posted == []  # dry run never posts
    # lockfile/docs/binary/deleted are skipped before any LLM call
    assert state["skipped"]["assets/logo.png"] == "binary"
    assert state["skipped"]["src/app/old.py"] == "deleted"


def test_prompt_injection_is_wrapped_as_untrusted():
    llm = FakeLLM()
    run(llm)
    review_prompt = next(m for name, m in llm.calls if name == "IssueList")[-1]["content"]
    start = review_prompt.index("<untrusted_pr_content>")
    assert review_prompt.index("Ignore previous instructions") > start


def test_hallucinated_quote_is_dropped():
    fake = issue(quote="this code does not exist")
    state, _, _ = run(FakeLLM(issues=[fake]))
    assert state["review"].comments == []
    assert state["drops"]["quote not on an added line"] == 1


def test_off_by_one_line_is_corrected():
    state, _, _ = run(FakeLLM(issues=[issue(line=3)]))
    assert state["review"].comments[0].line == 2


def test_verifier_drop_and_confidence_min():
    state, _, _ = run(FakeLLM(verdict="drop"))
    assert state["review"].comments == [] and state["drops"]["dropped by verifier"] == 1

    state, _, _ = run(FakeLLM(verify_confidence=0.4))  # min(0.9, 0.4) < 0.6 threshold
    assert state["review"].comments == []
    assert state["drops"]["below confidence threshold"] == 1


def test_publish_and_idempotency():
    state, github, _ = run(FakeLLM(), dry_run=False)
    assert state["published"] and len(github.posted) == 1

    state, github, _ = run(FakeLLM(), github=FakeGitHub(already_reviewed=True), dry_run=False)
    assert not state["published"] and state["publish_note"] == "already reviewed this commit"


def test_all_chunks_fail_is_never_published():
    state, github, _ = run(FakeLLM(fail_review=True), dry_run=False)
    assert state["incomplete"] and not state["published"] and github.posted == []
    assert "failed" in state["review"].summary


def test_nothing_to_review_skips_llm():
    docs_only = (
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    llm = FakeLLM()
    state, _, _ = run(llm, github=FakeGitHub(diff=docs_only))
    assert llm.calls == [] and state["review"].comments == []


@pytest.mark.parametrize("cap", [0, 1])
def test_cap(cap):
    state, _, _ = run(FakeLLM(), max_comments=cap)
    assert len(state["review"].comments) == cap


def test_checkpoint_roundtrip_and_publish_later(tmp_path):
    """Review in dry-run, then publish from the saved state with no new LLM calls."""
    db = str(tmp_path / "cp.sqlite")
    config = {"configurable": {"thread_id": "o/r#7@hhh"}}
    with open_checkpointer(db) as saver:
        llm = FakeLLM()
        state, github, graph = run(llm, checkpointer=saver, config=config)
        assert github.posted == []
        calls_before = len(llm.calls)

        saved = graph.get_state(config)
        assert saved.values["review"].comments[0].line == 2  # types survive the round trip
        assert saved.values["files"]["src/app/utils.py"].commentable_lines

        graph.update_state(config, {"dry_run": False}, as_node="finalize")
        final = graph.invoke(None, config)
        assert final["published"] and len(github.posted) == 1
        assert len(llm.calls) == calls_before


def test_run_review_statuses(tmp_path):
    from pr_review.agent.run import run_review

    settings = Settings(_env_file=None, checkpoint_db=str(tmp_path / "cp.sqlite"))
    github, llm = FakeGitHub(), FakeLLM()
    kwargs = dict(settings=settings, github=github, llm=llm, pr_number=7, focus="bugs")

    first = run_review(**kwargs)
    assert first.status == "new" and first.thread_id == "o/r#7@hhhhhhhhhhhh"
    n_calls = len(llm.calls)

    assert run_review(**kwargs).status == "cached"  # same commit: free
    published = run_review(**kwargs, dry_run=False)
    assert published.status == "published-from-cache" and len(github.posted) == 1
    assert len(llm.calls) == n_calls  # neither re-run paid for the LLM again

    assert run_review(**kwargs, fresh=True).status == "new"
