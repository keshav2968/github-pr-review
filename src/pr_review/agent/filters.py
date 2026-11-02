"""Deterministic quality gates applied after the LLM steps.

These are plain functions (no LLM), so they are fast, fully unit-tested, and
explainable: for every dropped issue we record *why* it was dropped. The counts
end up in the eval report, e.g. "grounding removed 18% of candidate issues".
"""

from collections import Counter

from pr_review.agent.schemas import Candidate
from pr_review.github.diff import FileDiff, quote_matches_line

SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def ground(
    candidates: list[Candidate], files: dict[str, FileDiff], window: int, drops: Counter
) -> list[Candidate]:
    """Keep only issues whose quoted code really is on an added line.

    Also *fixes* small line-number mistakes: if the quote is found within
    `window` lines of what the model said, we move the comment there.
    """
    kept = []
    for c in candidates:
        fd = files.get(c.issue.path)
        if fd is None:
            drops["unknown file"] += 1
            continue
        line = quote_matches_line(fd, c.issue.line, c.issue.quote, window=window)
        if line is None:
            drops["quote not on an added line"] += 1
            continue
        if line != c.issue.line:
            drops["line corrected (kept)"] += 1
            c = c.model_copy(update={"issue": c.issue.model_copy(update={"line": line})})
        kept.append(c)
    return kept


def dedupe(candidates: list[Candidate], drops: Counter) -> list[Candidate]:
    """One comment per (file, line): overlapping chunks or loops can repeat an issue."""
    best: dict[tuple[str, int], Candidate] = {}
    for c in candidates:
        key = (c.issue.path, c.issue.line)
        if key in best:
            drops["duplicate"] += 1
            if c.issue.confidence <= best[key].issue.confidence:
                continue
        best[key] = c
    return list(best.values())


def threshold(candidates: list[Candidate], minimum: float, drops: Counter) -> list[Candidate]:
    kept = [c for c in candidates if c.issue.confidence >= minimum]
    drops["below confidence threshold"] += len(candidates) - len(kept)
    return kept


def limit_missing_tests(candidates: list[Candidate], drops: Counter) -> list[Candidate]:
    """At most one 'missing-test' comment per PR: the rest are the same point repeated."""
    out, seen = [], False
    for c in sorted(candidates, key=lambda c: -c.issue.confidence):
        if c.issue.category == "missing-test":
            if seen:
                drops["extra missing-test"] += 1
                continue
            seen = True
        out.append(c)
    return out


def rank_and_cap(candidates: list[Candidate], cap: int, drops: Counter) -> list[Candidate]:
    ranked = sorted(
        candidates, key=lambda c: (SEVERITY_RANK[c.issue.severity], -c.issue.confidence)
    )
    drops["over per-PR cap"] += max(0, len(ranked) - cap)
    return ranked[:cap]


def apply_all(
    candidates: list[Candidate],
    files: dict[str, FileDiff],
    *,
    window: int,
    min_confidence: float,
    cap: int,
) -> tuple[list[Candidate], dict[str, int]]:
    drops: Counter = Counter()
    kept = ground(candidates, files, window, drops)
    kept = dedupe(kept, drops)
    kept = threshold(kept, min_confidence, drops)
    kept = limit_missing_tests(kept, drops)
    kept = rank_and_cap(kept, cap, drops)
    return kept, {k: v for k, v in drops.items() if v}
