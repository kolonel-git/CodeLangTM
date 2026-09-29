import pytest

from codelangtm import LANGUAGES
from codelangtm.data import Snippet, load_snippets, save_snippets


def make(**over):
    base = dict(
        text="def f():\n    return 1\n",
        language="python",
        source="github",
        repo="owner/name",
        commit="abc123",
        path="src/a.py",
        license="MIT",
        start_line=1,
        end_line=2,
    )
    base.update(over)
    return Snippet(**base)


def test_group_key_and_lines():
    s = make(start_line=10, end_line=29)
    assert s.group_key == "owner/name"
    assert s.n_lines == 20


@pytest.mark.parametrize(
    "over",
    [
        {"text": "  \n"},
        {"source": "made-up"},
        {"repo": ""},
        {"license": ""},
        {"start_line": 0},
        {"start_line": 5, "end_line": 4},
    ],
)
def test_validation_rejects(over):
    with pytest.raises(ValueError):
        make(**over)


def test_roundtrip(tmp_path):
    items = [make(), make(language="go", text="package main\n", path="m.go")]
    p = tmp_path / "s.jsonl"
    assert save_snippets(items, p) == 2
    assert list(load_snippets(p)) == items


def test_load_rejects_unknown_language(tmp_path):
    p = tmp_path / "s.jsonl"
    save_snippets([make(language="cobol")], p)
    with pytest.raises(ValueError, match="unknown language"):
        list(load_snippets(p, languages=LANGUAGES))


def test_language_sets_are_consistent():
    from codelangtm import ALL_LANGUAGES, CORE_LANGUAGES, STRETCH_LANGUAGES

    assert ALL_LANGUAGES == CORE_LANGUAGES + STRETCH_LANGUAGES
    assert len(set(ALL_LANGUAGES)) == len(ALL_LANGUAGES) == 14
    assert LANGUAGES == CORE_LANGUAGES  # evaluation defaults stay on the core 8 until v6


def test_languages_present_keeps_project_order():
    from codelangtm.data import languages_present

    items = [make(language="ruby"), make(language="go"), make(language="python"), make()]
    assert languages_present(items) == ("python", "go", "ruby")
    assert languages_present([]) == ()


def test_load_snippets_accepts_stretch_labels(tmp_path):
    from codelangtm import ALL_LANGUAGES

    save_snippets([make(language="kotlin", path="A.kt")], tmp_path / "k.jsonl")
    assert len(list(load_snippets(tmp_path / "k.jsonl", ALL_LANGUAGES))) == 1
    with pytest.raises(ValueError, match="unknown language"):
        list(load_snippets(tmp_path / "k.jsonl", LANGUAGES))
