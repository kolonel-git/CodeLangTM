import pytest

from codelangtm.audit import (
    _fence,
    audit_flags,
    comment_ratio,
    file_role,
    language_stats,
    permalink,
    pick_samples,
    render_markdown,
)
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.data import Snippet, save_snippets

LANGS = ("python", "go")


def snip(lang="python", repo="o/r", path="src/a.py", text=None, source="github", i=0):
    text = text or f"def f{i}(x):\n    return x + {i}\n"
    return Snippet(text, lang, source, repo, "abc123", path, "MIT", 3, 2 + text.count("\n"))


@pytest.mark.parametrize(
    ("path", "role"),
    [
        ("src/app.py", "source"),
        ("tests/test_app.py", "test"),
        ("pkg/server_test.go", "test"),
        ("src/test/java/FooTest.java", "test"),
        ("src/Contest.java", "source"),  # lowercase "test" suffix is not a test file
        ("web/app.spec.js", "test"),
        ("examples/demo.rs", "example"),
        ("db/migrations/001_init.sql", "migration"),
        ("docs/index.html", "docs"),
    ],
)
def test_file_role(path, role):
    assert file_role(path) == role


def test_file_role_wild():
    assert file_role("answer.txt", source="wild") == "wild"


def test_comment_ratio():
    assert comment_ratio("# a\nx = 1\n\n# b\n") == pytest.approx(2 / 3)
    assert comment_ratio("") == 0.0


def balanced(n_repos=12):
    ext = {"python": "py", "go": "go"}
    return {
        "train": [snip(lang, f"{lang}/r{r}", f"src/f.{ext[lang]}", i=r) for lang in LANGS
                  for r in range(n_repos)]
    }  # fmt: skip


def test_stats_and_clean_dataset_has_no_flags():
    stats = language_stats(balanced(), LANGS)
    st = stats["python"]
    assert (st.snippets, st.repos, st.splits["train"]) == (12, 12, 12)
    assert st.top_repo_share == pytest.approx(1 / 12)
    assert st.roles["source"] == 12
    assert audit_flags(stats) == []


def flags_for(extra):
    data = balanced()
    data["train"] += extra
    return "\n".join(audit_flags(language_stats(data, LANGS)))


def test_flag_dominant_repo_and_imbalance():
    joined = flags_for([snip("python", "big/repo", "src/b.py", i=100 + i) for i in range(10)])
    assert "python: big/repo supplies 45%" in joined  # 10 of 22
    assert "go: 12 snippets vs 22" in joined


def test_flag_test_heavy():
    extra = [snip("python", f"t/r{i}", f"tests/test_{i}.py", i=100 + i) for i in range(12)]
    assert "python: 50% of snippets come from test files" in flags_for(extra)


def test_flag_comment_heavy():
    text = "# a\n# b\n# c\nx = 1\n"
    extra = [snip("python", f"c/r{i}", "src/c.py", text=text + f"y = {i}\n") for i in range(13)]
    assert "python: median window is 60% comments" in flags_for(extra)


def test_missing_language_flagged():
    flags = audit_flags(language_stats(balanced(), ("python", "go", "rust")))
    assert "rust: no data" in flags


def test_samples_deterministic_and_capped():
    data = balanced()
    a = pick_samples(data, 5, seed=1, languages=LANGS)
    assert a == pick_samples(data, 5, seed=1, languages=LANGS)
    assert all(len(v) == 5 for v in a.values())
    assert len(pick_samples(data, 50, languages=LANGS)["go"]) == 12


def test_permalink():
    s = snip(path="src/my file.py")
    assert permalink(s) == "https://github.com/o/r/blob/abc123/src/my%20file.py#L3-L4"
    assert permalink(snip(source="wild")) is None


def test_fence_longer_than_backticks_in_text():
    assert _fence("x = 1") == "```"
    assert _fence("s = '````'") == "`````"


def test_render_markdown_contains_sections_and_checkboxes():
    data = balanced()
    stats = language_stats(data, LANGS)
    samples = pick_samples(data, 2, languages=LANGS)
    md = render_markdown("data/processed", stats, [], samples, 0, "2026-09-24 20:00:00")
    assert "Generated 2026-09-24 20:00:00" in md and "(24 snippets)" in md
    for heading in ("## Summary", "## Flags", "## File roles", "## Licenses", "### python"):
        assert heading in md
    assert md.count("- [ ] OK") == 4
    assert "- none" in md


def test_cli_audit_end_to_end(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    items = [
        snip(lang, f"{lang}/r{r}", f"src/f.{'py' if lang == 'python' else 'go'}",
             text=(f"def f{r}(x):\n    return x * {r}\n" if lang == "python"
                   else f"package p\n\nfunc F{r}() int {{ return {r} }}\n"))
        for lang in LANGS for r in range(8)
    ]  # fmt: skip
    save_snippets(items, raw / "all.jsonl")
    processed = tmp_path / "processed"
    build_dataset([raw], processed, languages=LANGS)

    out = tmp_path / "audit.md"
    args = ["data", "audit", "--data", str(processed), "--out", str(out), "--samples", "3"]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "language" in printed and "review file" in printed
    assert out.read_text(encoding="utf-8").count("- [ ] OK") == 6  # 3 python + 3 go; rest absent

    assert main(["data", "audit", "--data", str(tmp_path / "nope")]) == 1
    assert "data build" in capsys.readouterr().err
