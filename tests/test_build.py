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
    assert set(manifest["files"]) == {
        "train.jsonl", "test.jsonl", "wild.jsonl", "hard.jsonl", "folds.json"
    }
    assert manifest["params"]["split"] == "stable-hash"
    assert manifest["params"]["salt"] == "codelangtm-v1"


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


def kotlin_snippet(repo, i):
    text = "".join(
        f"fun f{repo[-1]}{i}x{j}(a: Int): Int {{\n    return a + {j}\n}}\n" for j in range(9)
    )
    return Snippet(
        text, "kotlin", "github", repo, "sha", f"src/A{i}.kt", "MIT", 1, text.count("\n")
    )


def test_default_build_accepts_all_14_and_reports_only_languages_present(raw, tmp_path):
    save_snippets(
        [kotlin_snippet(f"kt-org/r{r}", i) for r in range(8) for i in range(2)],
        raw / "github" / "kotlin.jsonl",
    )
    out = tmp_path / "out"
    report = build_dataset([raw], out)  # no `languages`: nothing restricts the labels
    assert report.counts["train"]["kotlin"] + report.counts["test"]["kotlin"] == 16
    summary = report.summary()
    assert "kotlin" in summary and "rust" not in summary  # absent languages are not listed
    assert not any(w.startswith(("rust", "ruby", "c:")) for w in report.warnings)
    manifest = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["languages"] == ["python", "go", "kotlin"]


def test_restricted_build_rejects_other_labels(raw, tmp_path):
    save_snippets([kotlin_snippet("kt-org/r0", 0)], raw / "github" / "kotlin.jsonl")
    with pytest.raises(ValueError, match="unknown language"):
        build(raw, tmp_path / "out")


# --- hard examples (Stage B, B-S2) -------------------------------------------------------------


def html_text(tag, script_lines):
    markup = "".join(f'<p class="{tag}{i}">text {tag} {i}</p>\n' for i in range(4))
    script = "".join(f"  var {tag}{i} = compute_{tag}({i}) + {i};\n" for i in range(script_lines))
    return f"<div>\n{markup}<script>\n{script}</script>\n</div>\n"


def html_snippet(repo, i, script_lines):
    text = html_text(f"{repo[-1]}{i}", script_lines)
    return Snippet(text, "html", "github", repo, "sha", f"web/p{i}.html", "MIT", 1,
                   text.count("\n"))  # fmt: skip


@pytest.fixture
def hard_dir(tmp_path):
    root = tmp_path / "hard"
    root.mkdir()
    # repo html-org/r0 has normal windows in `raw` (added below) and hard windows here;
    # html-org/only has nothing but hard windows
    hard = [html_snippet("html-org/r0", 50 + i, 30) for i in range(2)]
    hard += [html_snippet("html-org/only", 60 + i, 30) for i in range(2)]
    save_snippets(hard, root / "html.jsonl")
    return root


def add_html(raw):
    items = [html_snippet(f"html-org/r{r}", i, 1) for r in range(8) for i in range(3)]
    save_snippets(items, raw / "github" / "html.jsonl")


def test_hard_examples_are_kept_apart_from_train_and_test(raw, hard_dir, tmp_path):
    add_html(raw)
    out = tmp_path / "out"
    report = build_dataset([raw], out, hard_sources=[hard_dir])
    hard = list(load_snippets(out / "hard.jsonl"))
    assert len(hard) == 4 and report.hard["count"] == 4
    assert report.hard["by_language"] == {"html": 4}
    # r0 is a normal repo (train or test); "only" is in neither
    assert set(report.hard["repo_in"]) <= {"train", "test", "neither"}
    assert report.hard["repo_in"]["neither"] == 2
    texts = {
        s.text for name in ("train", "test", "wild") for s in load_snippets(out / f"{name}.jsonl")
    }
    assert not texts & {s.text for s in hard}
    assert "hard examples (evaluation only): 4" in report.summary()
    manifest = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["hard"]["count"] == 4 and "hard.jsonl" in manifest["files"]


def test_hard_windows_never_change_train_test_or_folds(raw, hard_dir, tmp_path):
    add_html(raw)
    with_hard = build_dataset([raw], tmp_path / "a", hard_sources=[hard_dir])
    without = build_dataset([raw], tmp_path / "b")
    for name in ("train.jsonl", "test.jsonl", "wild.jsonl", "folds.json"):
        assert with_hard.files[name] == without.files[name], name
    assert without.hard["count"] == 0
    assert list(load_snippets(tmp_path / "b" / "hard.jsonl")) == []


def test_normal_window_in_hard_file_is_ignored_with_a_warning(raw, hard_dir, tmp_path):
    add_html(raw)
    normal = html_snippet("html-org/elsewhere", 0, 1)  # not mostly embedded
    save_snippets([normal], hard_dir / "normal.jsonl")
    report = build_dataset([raw], tmp_path / "out", hard_sources=[hard_dir])
    assert report.hard["count"] == 4
    assert any("hard example(s) ignored" in w for w in report.warnings)


def test_duplicate_hard_windows_are_removed(raw, hard_dir, tmp_path):
    add_html(raw)
    save_snippets([html_snippet("html-org/r0", 50, 30)], hard_dir / "again.jsonl")  # copy of one
    report = build_dataset([raw], tmp_path / "out", hard_sources=[hard_dir])
    assert report.hard["count"] == 4


def test_hard_file_with_wrong_label_is_rejected(raw, tmp_path):
    root = tmp_path / "hard"
    root.mkdir()
    save_snippets([html_snippet("x/y", 0, 30)], root / "h.jsonl")
    with pytest.raises(ValueError, match="unknown language"):
        build_dataset([raw], tmp_path / "out", languages=LANGS, hard_sources=[root])


def test_cli_build_finds_data_hard_by_default(raw, hard_dir, tmp_path, monkeypatch, capsys):
    add_html(raw)
    monkeypatch.chdir(tmp_path)  # `data/hard` is resolved against the working directory
    (tmp_path / "data").mkdir()
    hard_dir.rename(tmp_path / "data" / "hard")
    out = tmp_path / "out"
    assert main(["data", "build", "--source", str(raw), "--out", str(out)]) == 0
    assert "hard examples (evaluation only): 4" in capsys.readouterr().out
    assert main(["data", "build", "--source", str(raw), "--out", str(out),
                 "--hard-source", str(tmp_path / "nowhere")]) == 1  # fmt: skip


def test_hard_windows_among_the_main_sources_are_moved_to_hard(raw, tmp_path):
    add_html(raw)
    save_snippets([html_snippet("html-org/r0", 70, 30)], raw / "github" / "html_extra.jsonl")
    report = build_dataset([raw], tmp_path / "out")
    assert report.hard["count"] == 1 and not report.label_dropped
    total = sum(report.counts[s]["html"] for s in ("train", "test"))
    assert total == 24  # the 24 normal windows; the hard one is in neither split
