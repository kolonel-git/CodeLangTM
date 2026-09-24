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
