from pathlib import Path

import pytest

from pr_review.github.diff import DiffParseError, parse_diff, quote_matches_line

SAMPLE = (Path(__file__).parent / "fixtures" / "sample.diff").read_text(encoding="utf-8")


@pytest.fixture
def files():
    return parse_diff(SAMPLE)


def test_all_files_found(files):
    assert set(files) == {
        "src/app/utils.py",
        "src/app/new_mod.py",
        "src/app/old.py",
        "src/app/name_b.py",
        "assets/logo.png",
    }


def test_modified_file_line_numbers(files):
    fd = files["src/app/utils.py"]
    # Hunk starts at new line 10: lines 10,11,12 are context, 13,14 added, 15.. context.
    assert fd.added_lines == {
        13: "        loc = os.path.expanduser(f)",
        14: '        print("debug", loc)',
    }
    assert {10, 11, 12, 13, 14, 15, 16, 17} == fd.commentable_lines
    assert "   14 + " in fd.numbered_patch
    assert " - " in fd.numbered_patch  # removed lines are shown without a number


def test_flags(files):
    assert files["src/app/new_mod.py"].is_new
    assert files["src/app/old.py"].is_deleted
    assert files["src/app/old.py"].commentable_lines == set()
    renamed = files["src/app/name_b.py"]
    assert renamed.is_renamed and renamed.source_path == "src/app/name_a.py"
    assert files["assets/logo.png"].is_binary


def test_grounding_exact_and_window(files):
    fd = files["src/app/utils.py"]
    assert quote_matches_line(fd, 14, 'print("debug", loc)') == 14
    # whitespace differences are ignored
    assert quote_matches_line(fd, 14, '  print( "debug",   loc)'.replace("( ", "(")) == 14
    # off-by-one line from the model is corrected only when a window is allowed
    assert quote_matches_line(fd, 15, 'print("debug", loc)') is None
    assert quote_matches_line(fd, 15, 'print("debug", loc)', window=2) == 14
    # quotes that are not on an ADDED line are rejected (context line 16)
    assert quote_matches_line(fd, 16, "netrc_path = loc") is None
    assert quote_matches_line(fd, 14, "") is None


def test_empty_and_garbage():
    assert parse_diff("") == {}
    with pytest.raises(DiffParseError):
        parse_diff("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-only one line\n")
