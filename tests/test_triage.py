from pr_review.agent.triage import estimate_tokens, make_chunks, skip_reason
from pr_review.github.diff import FileDiff


def fd(path, n_lines=3, **kw):
    added = {i: f"x = {i}" for i in range(1, n_lines + 1)}
    patch = f"### {path}\n@@\n" + "\n".join(f"{i:>5} + x = {i}" for i in added)
    return FileDiff(
        path=path, added_lines=added, commentable_lines=set(added), numbered_patch=patch, **kw
    )


def test_skip_rules():
    assert skip_reason(fd("uv.lock")) == "matches *.lock"
    assert skip_reason(fd("docs/guide.md")) == "matches *.md"
    assert skip_reason(fd("web/node_modules/x.js")) == "matches */node_modules/*"
    assert skip_reason(fd("api/user_pb2.py")) == "matches *_pb2.py"
    assert skip_reason(fd("src/app.py", is_binary=True)) == "binary"
    assert skip_reason(fd("src/app.py", n_lines=0)) == "no added lines"
    assert skip_reason(fd("src/app.py")) is None


def test_small_files_are_packed_together():
    chunks = make_chunks([fd("a.py"), fd("b.py")], budget=1_000)
    assert len(chunks) == 1 and chunks[0].paths == ["a.py", "b.py"]


def test_budget_splits_into_chunks():
    files = [fd(f"f{i}.py", n_lines=40) for i in range(4)]
    budget = estimate_tokens(files[0].numbered_patch) + 10
    chunks = make_chunks(files, budget=budget)
    assert [c.paths for c in chunks] == [["f0.py"], ["f1.py"], ["f2.py"], ["f3.py"]]
    assert [c.index for c in chunks] == [0, 1, 2, 3]


def test_huge_file_is_split_at_hunks():
    big = fd("big.py", n_lines=10)
    hunk = "\n".join(f"{i:>5} + y = {i}" for i in range(200))
    big.numbered_patch = "### big.py\n@@ a\n" + hunk + "\n@@ b\n" + hunk + "\n@@ c\n" + hunk
    chunks = make_chunks([big], budget=estimate_tokens(hunk) + 50)
    assert len(chunks) >= 2 and all(c.text.startswith("### big.py") for c in chunks)
