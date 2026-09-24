import json
from collections import defaultdict

import pytest

from codelangtm.build import build_dataset, load_sources
from codelangtm.cli import main
from codelangtm.data import Snippet, load_snippets, save_snippets

LANGS = ("python", "go")


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j} + len(\"{tag}\")\n}}\n"
                   for j in range(8))  # fmt: skip
    return "package main\n\n" + body


def gh_snippet(lang, repo, i):
    tag = f"{repo.replace('/', '_')}_{i}"
    text, ext = (py_text(tag), "py") if lang == "python" else (go_text(tag), "go")
    return Snippet(text, lang, "github", repo, "sha", f"src/f{i}.{ext}", "MIT", 1,
                   text.count("\n"))  # fmt: skip


def wild_snippet(text, n):
    return Snippet(text, "python", "wild", f"stackoverflow/q{n}", "-", "answer.txt",
                   "CC-BY-SA-4.0", 1, text.count("\n"))  # fmt: skip


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    (root / "github").mkdir(parents=True)
    for lang in LANGS:
        items = [gh_snippet(lang, f"{lang}-org/r{r}", i) for r in range(8) for i in range(3)]
        save_snippets(items, root / "github" / f"{lang}.jsonl")
    return root


def build(raw, out, **kw):
    return build_dataset([raw], out, languages=LANGS, **kw)


def test_splits_are_leak_free_and_complete(raw, tmp_path):
    out = tmp_path / "out"
    report = build(raw, out)
    train = list(load_snippets(out / "train.jsonl"))
    test = list(load_snippets(out / "test.jsonl"))
    assert not {s.repo for s in train} & {s.repo for s in test}
    assert len(train) + len(test) == report.loaded == 48
    assert set(report.counts["test"]) == set(LANGS)
    assert list(load_snippets(out / "wild.jsonl")) == []

    folds = json.loads((out / "folds.json").read_text(encoding="utf-8"))
    assert len(folds["folds"]) == len(train)
    assert set(folds["folds"]) == set(range(5))
    repo_folds = defaultdict(set)
    for s, k in zip(train, folds["folds"], strict=True):
        repo_folds[s.repo].add(k)
    assert all(len(ks) == 1 for ks in repo_folds.values())  # a repo never spans folds

    manifest = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"train.jsonl", "test.jsonl", "wild.jsonl", "folds.json"}
    assert manifest["params"]["seed"] == 0


def test_wild_duplicate_of_training_data_is_dropped(raw, tmp_path):
    copied = gh_snippet("python", "python-org/r0", 0).text
    unique = "import sys\n\nprint(sys.argv[1:])\n"
    save_snippets([wild_snippet(copied, 1), wild_snippet(unique, 2)], _wild_file(raw))
    report = build(raw, tmp_path / "out")
    wild = list(load_snippets(tmp_path / "out" / "wild.jsonl"))
    assert [s.text for s in wild] == [unique]
    assert report.duplicates_removed == 1
    assert report.counts["train"]["python"] + report.counts["test"]["python"] == 24


def test_mislabelled_snippets_dropped(raw, tmp_path):
    bad = gh_snippet("python", "python-org/r0", 99)
    bad = Snippet(bad.text, "go", "github", "go-org/x", "sha", "src/x.py", "MIT", 1, 24)
    save_snippets([bad], raw / "github" / "extra.jsonl")
    report = build(raw, tmp_path / "out")
    assert any("extension implies" in r for r in report.label_dropped)
    assert report.loaded == 49


def test_deterministic(raw, tmp_path):
    a = build(raw, tmp_path / "a")
    b = build(raw, tmp_path / "b")
    assert a.files == b.files


def test_thin_language_warning(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    items = [gh_snippet("python", f"python-org/r{r}", 0) for r in range(8)]
    items += [gh_snippet("go", "go-org/only", i) for i in range(3)]
    save_snippets(items, root / "mixed.jsonl")
    report = build(root, tmp_path / "out")
    assert any(w.startswith("go:") for w in report.warnings)
    assert not any(w.startswith("python:") and "training repo" in w for w in report.warnings)


def test_missing_sources(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sources([tmp_path / "nope"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .jsonl"):
        load_sources([tmp_path / "empty"])


def test_cli_data_build(raw, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["data", "build", "--source", str(raw), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "language" in printed and "total" in printed
    assert (out / "dataset.json").exists()
    assert main(["data", "build", "--source", str(tmp_path / "nope")]) == 1
    assert "data build failed" in capsys.readouterr().err


def _wild_file(raw):
    (raw / "wild").mkdir(exist_ok=True)
    return raw / "wild" / "python.jsonl"
