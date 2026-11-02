# github-pr-review

An agentic pull-request reviewer: a LangGraph state machine that reviews a PR's diff,
runs its tests in an isolated sandbox, and posts grounded inline comments back to GitHub.

> Status: being built in steps (see the PRs). Step 1 = foundations.

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
```

## Development

```bash
uv run pytest            # unit tests (no network, no LLM, no Docker needed)
uv run ruff check src tests
```
