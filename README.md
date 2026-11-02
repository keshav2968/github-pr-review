# github-pr-review

An agentic pull-request reviewer: a LangGraph state machine that reviews a PR's diff,
runs its tests in an isolated sandbox, and posts grounded inline comments back to GitHub.

> Status: being built in steps (see the PRs). Done: foundations, LLM + LangGraph agent.

## How it works (so far)

```
fetch_context -> triage -> review_chunk (x N, parallel) -> verify -> finalize -> [publish]
```

| step | what it does | why it raises precision |
|---|---|---|
| triage | skips lockfiles, docs, generated, binary; splits big diffs into chunks | no noise comments on files that can't contain our bug types |
| review_chunk | LLM finds issues; output forced into a schema by Outlines | schema has no "style" category; must quote the exact line |
| verify | second skeptical LLM pass with the **full file** as context | removes "x may be None" when x is checked outside the hunk |
| finalize | grounding (quote must be on an added line), dedupe, confidence threshold, cap | drops invented issues; fixes off-by-N line numbers |
| publish | one COMMENT-only review, pinned to the head commit, idempotent | never approves/blocks; never double-posts |

## Setup

```bash
uv sync
cp .env.example .env      # then fill in the keys
uv run pr-review doctor   # checks gh, docker, keys
```

## Usage

```bash
# Dry run (default): prints the review, posts nothing
uv run pr-review review 6963 --repo psf/requests

# Same commit again -> instant, from the saved checkpoint (no LLM cost)
# Publish a review you already ran in dry-run, without re-reviewing:
uv run pr-review review 12 --repo you/your-repo --publish
uv run pr-review review 6963 --repo psf/requests --fresh   # ignore the saved result
uv run pr-review doctor --live                             # one real LLM call to test setup
```

## Development

```bash
uv run pytest            # unit tests (no network, no LLM, no Docker needed)
uv run ruff check src tests
```
