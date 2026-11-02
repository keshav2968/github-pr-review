"""Decide what is worth reviewing, and cut it into LLM-sized pieces.

Skipping files is a precision lever: lockfiles, generated code and docs produce
noise comments and burn tokens without ever containing the bugs we look for.
"""

import fnmatch
from dataclasses import dataclass

from pr_review.github.diff import FileDiff

SKIP_PATTERNS = [
    # lockfiles
    "*.lock", "package-lock.json", "pnpm-lock.yaml", "go.sum", "Pipfile.lock",
    # generated / vendored / build output
    "*.min.js", "*.min.css", "*.map", "*_pb2.py", "*_pb2_grpc.py", "*.pb.go",
    "dist/*", "build/*", "vendor/*", "*/vendor/*", "node_modules/*", "*/node_modules/*",
    "*.snap", "*.svg",
    # docs and data (our categories don't apply)
    "*.md", "*.rst", "*.txt", "*.csv", "LICENSE*", "CHANGELOG*",
]  # fmt: skip


def skip_reason(fd: FileDiff) -> str | None:
    if fd.is_binary:
        return "binary"
    if fd.is_deleted:
        return "deleted"
    if not fd.added_lines:
        return "no added lines"
    name = fd.path.rsplit("/", 1)[-1]
    for pattern in SKIP_PATTERNS:
        if fnmatch.fnmatch(fd.path, pattern) or fnmatch.fnmatch(name, pattern):
            return f"matches {pattern}"
    return None


def estimate_tokens(text: str) -> int:
    # ~4 characters per token for code in English-like tokenizers. An estimate is
    # enough here: we only need to stay safely under the budget.
    return len(text) // 4 + 1


@dataclass
class Chunk:
    index: int
    paths: list[str]
    text: str


def _split_by_hunk(fd: FileDiff, budget: int) -> list[str]:
    """Split one huge file's numbered patch at hunk boundaries ('@@' lines)."""
    header, *rest = fd.numbered_patch.split("\n")
    pieces: list[str] = []
    current: list[str] = []
    for line in rest:
        if line.startswith("@@") and current and estimate_tokens("\n".join(current)) > budget:
            pieces.append("\n".join([header + " (part)", *current]))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join([header + (" (part)" if pieces else ""), *current]))
    return pieces


def make_chunks(files: list[FileDiff], budget: int) -> list[Chunk]:
    """Pack whole files into chunks under `budget` tokens (big files get split).

    Keeping a file together matters: a bug is often only visible when you see two
    hunks of the same function at once.
    """
    chunks: list[Chunk] = []
    paths: list[str] = []
    texts: list[str] = []

    def flush() -> None:
        if texts:
            chunks.append(Chunk(len(chunks), list(paths), "\n\n".join(texts)))
            paths.clear()
            texts.clear()

    for fd in files:
        tokens = estimate_tokens(fd.numbered_patch)
        if tokens > budget:
            flush()
            for piece in _split_by_hunk(fd, budget):
                chunks.append(Chunk(len(chunks), [fd.path], piece))
            continue
        if texts and estimate_tokens("\n\n".join(texts)) + tokens > budget:
            flush()
        paths.append(fd.path)
        texts.append(fd.numbered_patch)
    flush()
    return chunks
