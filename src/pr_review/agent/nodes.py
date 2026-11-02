"""One method per graph node. Each takes the state and returns the keys it changed.

Error policy (robustness):
- A failure in ONE chunk or ONE file is recorded in `errors` and the run goes on;
  a partial review is better than none.
- If EVERY review chunk failed, the result is marked incomplete and never
  published, because silently posting "no issues found" would be misleading.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from outlines.exceptions import APIError

from pr_review.agent import filters
from pr_review.agent.deps import AgentDeps
from pr_review.agent.schemas import Candidate, IssueList, VerifyResult
from pr_review.agent.state import ChunkTask, ReviewState
from pr_review.agent.triage import make_chunks, skip_reason
from pr_review.github.diff import parse_diff
from pr_review.github.models import ReviewComment, ReviewResult, Severity
from pr_review.llm import prompts
from pr_review.llm.structured import StructuredOutputError

log = logging.getLogger(__name__)
LLM_ERRORS = (StructuredOutputError, APIError)
UNVERIFIED_PENALTY = 0.8  # verifier failed: keep the issue, but trust it a bit less
CONTEXT_RADIUS = 80  # lines of a long file shown around each issue


def file_excerpt(content: str, lines: list[int], max_chars: int) -> str:
    """Whole file if small; otherwise only the regions around the issue lines."""
    if len(content) <= max_chars:
        return content
    all_lines = content.splitlines()
    keep: set[int] = set()
    for n in lines:
        keep.update(range(max(1, n - CONTEXT_RADIUS), min(len(all_lines), n + CONTEXT_RADIUS) + 1))
    out, previous = [], 0
    for n in sorted(keep):
        if n != previous + 1:
            out.append("...")
        out.append(f"{n:>5}  {all_lines[n - 1]}")
        previous = n
    return "\n".join(out)[:max_chars]


class Nodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.s = deps.settings

    # ---------------------------------------------------------------- fetch
    def fetch_context(self, state: ReviewState) -> dict:
        gh = self.deps.github
        pr = gh.get_pull_request(state["pr_number"])
        files = parse_diff(gh.get_pull_request_diff(state["pr_number"]))
        return {"pr": pr, "files": files}

    # --------------------------------------------------------------- triage
    def triage(self, state: ReviewState) -> dict:
        skipped, reviewable = {}, []
        for fd in state["files"].values():
            reason = skip_reason(fd)
            if reason:
                skipped[fd.path] = reason
            else:
                reviewable.append(fd)
        return {"skipped": skipped, "chunks": make_chunks(reviewable, self.s.chunk_token_budget)}

    # --------------------------------------------------------------- review
    def review_chunk(self, task: ChunkTask) -> dict:
        chunk = task["chunk"]
        messages = prompts.review_messages(task["pr"], chunk.text, task["focus"])
        try:
            result = self.deps.llm.generate(messages, IssueList, name=f"review_chunk_{chunk.index}")
        except LLM_ERRORS as error:
            log.warning("review of chunk %s failed: %s", chunk.index, error)
            return {"errors": [f"review chunk {chunk.index} ({', '.join(chunk.paths)}): {error}"]}
        candidates = [
            Candidate(
                id=chunk.index * 1000 + i,  # unique across parallel chunks
                issue=issue,
                chunk=chunk.index,
                original_confidence=issue.confidence,
            )
            for i, issue in enumerate(result.issues)
        ]
        return {"candidates": candidates}

    # --------------------------------------------------------------- verify
    def verify(self, state: ReviewState) -> dict:
        """Second, skeptical pass per file, with the full file as extra context.

        Many false positives come from the reviewer seeing only the diff ("x may be
        None!" when x is checked 20 lines above, outside the hunk). The verifier
        sees the whole file at the PR's head commit.
        """
        by_path: dict[str, list[Candidate]] = defaultdict(list)
        for c in state.get("candidates", []):
            by_path[c.issue.path].append(c)
        if not by_path:
            return {"verified": []}

        with ThreadPoolExecutor(max_workers=max(1, self.s.llm_concurrency)) as pool:
            results = list(pool.map(lambda item: self._verify_file(state, *item), by_path.items()))

        verified = [c for cs, _ in results for c in cs]
        errors = [e for _, e in results if e]
        return {"verified": verified, "errors": errors}

    def _verify_file(
        self, state: ReviewState, path: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str | None]:
        pr, fd = state["pr"], state["files"].get(path)
        content = None
        try:
            raw = self.deps.github.get_file(path, pr.head_sha)
            content = file_excerpt(
                raw, [c.issue.line for c in candidates], self.s.max_file_context_chars
            )
        except Exception as error:  # context is a bonus; verification still runs
            log.info("no file context for %s: %s", path, error)

        patch = fd.numbered_patch if fd else ""
        messages = prompts.verify_messages(pr, path, patch, content, candidates)
        try:
            result = self.deps.llm.generate(messages, VerifyResult, name=f"verify:{path}")
        except LLM_ERRORS as error:
            penalised = [
                c.model_copy(
                    update={
                        "verdict": "unverified",
                        "issue": c.issue.model_copy(
                            update={"confidence": c.issue.confidence * UNVERIFIED_PENALTY}
                        ),
                    }
                )
                for c in candidates
            ]
            return penalised, f"verify {path}: {error}"

        verdicts = {v.issue_id: v for v in result.verdicts}
        out = []
        for c in candidates:
            v = verdicts.get(c.id)
            if v is None:  # verifier skipped it -> treat as not confirmed
                out.append(
                    c.model_copy(update={"verdict": "drop", "verifier_reasoning": "no verdict"})
                )
                continue
            # Conservative: an issue is only as trustworthy as the LESS confident reviewer.
            confidence = min(c.issue.confidence, v.confidence)
            out.append(
                c.model_copy(
                    update={
                        "verdict": v.verdict,
                        "verifier_reasoning": v.reasoning,
                        "issue": c.issue.model_copy(update={"confidence": confidence}),
                    }
                )
            )
        return out, None

    # ------------------------------------------------------------- finalize
    def finalize(self, state: ReviewState) -> dict:
        files = state["files"]
        verified = state.get("verified", [])
        kept_by_verifier = [c for c in verified if c.verdict in ("keep", "unverified")]
        kept, drops = filters.apply_all(
            kept_by_verifier,
            files,
            window=self.s.grounding_window,
            min_confidence=self.s.min_confidence,
            cap=self.s.max_comments,
        )
        dropped_by_verifier = len(verified) - len(kept_by_verifier)
        if dropped_by_verifier:
            drops["dropped by verifier"] = dropped_by_verifier

        comments = []
        for c in kept:
            fd = files[c.issue.path]
            if c.issue.line not in fd.commentable_lines:  # GitHub would answer 422
                drops["not commentable"] = drops.get("not commentable", 0) + 1
                continue
            comments.append(
                ReviewComment(
                    path=c.issue.path,
                    line=c.issue.line,
                    body=f"**{c.issue.title}**\n\n{c.issue.explanation}",
                    severity=c.issue.severity,
                    category=c.issue.category,
                    suggested_code=c.issue.suggested_code or None,
                )
            )

        chunks = state.get("chunks", [])
        review_errors = [e for e in state.get("errors", []) if e.startswith("review chunk")]
        incomplete = bool(chunks) and len(review_errors) == len(chunks)
        summary = self._summary(state, comments, incomplete)
        return {
            "review": ReviewResult(summary=summary, comments=comments),
            "drops": drops,
            "incomplete": incomplete,
        }

    def _summary(self, state: ReviewState, comments: list[ReviewComment], incomplete: bool) -> str:
        pr = state["pr"]
        if incomplete:
            return f"Automated review of `{pr.head_sha[:7]}` failed: the LLM was unavailable."
        counts: dict[Severity, int] = defaultdict(int)
        for c in comments:
            counts[c.severity] += 1
        parts = [
            f"{n} {sev}" for sev in ("critical", "high", "medium", "low") if (n := counts[sev])
        ]
        lines = [
            f"### Automated review of `{pr.head_sha[:7]}`",
            f"Found **{len(comments)}** issue(s)" + (f" ({', '.join(parts)})." if parts else "."),
        ]
        if state.get("skipped"):
            lines.append(f"Skipped {len(state['skipped'])} file(s) (lockfiles, docs, generated).")
        if state.get("errors"):
            lines.append(
                f"Note: {len(state['errors'])} step(s) failed; this review may be partial."
            )
        lines.append("_Comments only: this bot never approves or blocks a PR._")
        return "\n\n".join(lines)

    # -------------------------------------------------------------- publish
    def publish(self, state: ReviewState) -> dict:
        pr, review = state["pr"], state["review"]
        if state.get("incomplete"):
            return {"published": False, "publish_note": "not posted: review incomplete"}
        if self.deps.github.has_reviewed(pr.number, pr.head_sha):
            return {"published": False, "publish_note": "already reviewed this commit"}
        self.deps.github.post_review(pr.number, pr.head_sha, review)
        return {"published": True, "publish_note": f"posted {len(review.comments)} comment(s)"}
