"""Turn a raw unified diff into something both the LLM and GitHub can use.

Two problems this solves:

1. **LLMs are bad at counting lines.** A raw diff only says "@@ -219,14 +219,7 @@";
   the model has to count from there to know a line is #226. It often gets this
   wrong, and a comment on the wrong line is a false positive. So we print every
   line *with its real line number* before giving it to the model.

2. **GitHub only accepts comments on lines that appear in the diff.** Commenting on
   any other line returns HTTP 422 and the whole review fails. `commentable_lines`
   is the exact set of allowed line numbers per file, so we can check every comment
   before posting.

We use the `unidiff` library for parsing rather than hand-written regexes because
diffs have many edge cases (renames, binary files, "\\ No newline at end of file").
"""

from dataclasses import dataclass, field

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError


class DiffParseError(ValueError):
    pass


@dataclass
class FileDiff:
    path: str  # path in the NEW version (what GitHub comments use)
    source_path: str | None = None  # old path (differs on rename)
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_binary: bool = False
    added_lines: dict[int, str] = field(default_factory=dict)  # new line no -> text
    commentable_lines: set[int] = field(default_factory=set)  # added + context lines
    numbered_patch: str = ""  # patch with real line numbers, for the LLM

    @property
    def additions(self) -> int:
        return len(self.added_lines)


def _strip_prefix(path: str | None) -> str | None:
    if path is None or path == "/dev/null":
        return None
    return path[2:] if path.startswith(("a/", "b/")) else path


def parse_diff(diff_text: str) -> dict[str, FileDiff]:
    """Parse `gh pr diff` output into {new_path: FileDiff}."""
    if not diff_text.strip():
        return {}
    try:
        patch = PatchSet(diff_text)
    except UnidiffParseError as error:
        raise DiffParseError(f"Could not parse diff: {error}") from error

    files: dict[str, FileDiff] = {}
    for patched in patch:
        fd = FileDiff(
            path=patched.path,
            source_path=_strip_prefix(patched.source_file),
            is_new=patched.is_added_file,
            is_deleted=patched.is_removed_file,
            is_renamed=patched.is_rename,
            is_binary=patched.is_binary_file,
        )
        out: list[str] = [
            f"### {fd.path}" + (f" (renamed from {fd.source_path})" if fd.is_renamed else "")
        ]
        for hunk in patched:
            out.append(f"@@ {hunk.section_header}".rstrip())
            for line in hunk:
                text = line.value.rstrip("\n")
                if line.is_added:
                    fd.added_lines[line.target_line_no] = text
                    fd.commentable_lines.add(line.target_line_no)
                    out.append(f"{line.target_line_no:>5} + {text}")
                elif line.is_context:
                    fd.commentable_lines.add(line.target_line_no)
                    out.append(f"{line.target_line_no:>5}   {text}")
                elif line.is_removed:
                    out.append(f"{'':>5} - {text}")
        fd.numbered_patch = "\n".join(out)
        files[fd.path] = fd
    return files


def normalize_code(text: str) -> str:
    """Compare code ignoring whitespace differences (LLMs often re-indent quotes)."""
    return " ".join(text.split())


def quote_matches_line(fd: FileDiff, line: int, quote: str, window: int = 0) -> int | None:
    """Return the added-line number whose text matches `quote`, or None.

    Used for *grounding*: the model must quote the exact code it complains about.
    If the quote doesn't appear on an added line, the issue is probably invented.
    `window` lets us fix small off-by-N line numbers by searching nearby lines.
    """
    wanted = normalize_code(quote)
    if not wanted:
        return None
    candidates = [line] + [n for d in range(1, window + 1) for n in (line - d, line + d)]
    for n in candidates:
        text = fd.added_lines.get(n)
        if text is not None and wanted in normalize_code(text):
            return n
    return None
