import json

import numpy as np
import pytest

from codelangtm import diagnostics as dg
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.data import Snippet, load_snippets, save_snippets
from codelangtm.features import Binarizer

LANGS = ("python", "go")


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j}\n}}\n" for j in range(8))
    return "package main\n\n" + body


def snip(text, lang, repo, i=0):
    ext = {"python": "py", "go": "go"}[lang]
    return Snippet(text, lang, "github", repo, "sha", f"src/f{i}.{ext}", "MIT", 1,
                   max(1, text.count("\n")))  # fmt: skip


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("diag")
    raw = root / "raw"
    raw.mkdir()
    items = [
        snip((py_text if lang == "python" else go_text)(f"{lang}{r}x{i}"), lang, f"{lang}/r{r}", i)
        for lang in LANGS for r in range(10) for i in range(3)
    ]  # fmt: skip
    save_snippets(items, raw / "all.jsonl")
    build_dataset([raw], root / "processed", languages=LANGS)
    return root / "processed"


def load(processed):
    train = list(load_snippets(processed / "train.jsonl"))
    folds = np.asarray(json.loads((processed / "folds.json").read_text())["folds"])
    return train, folds


def test_oof_probabilities(processed, monkeypatch):
    train, folds = load(processed)
    seen = []
    original = Binarizer.fit

    def spy(self, snippets, y=None):
        snippets = list(snippets)
        seen.append(set(snippets))
        return original(self, snippets, y)

    monkeypatch.setattr(Binarizer, "fit", spy)
    probs, classes = dg.oof_probabilities(train, folds, n_features=60)
    assert classes == ["go", "python"]
    assert probs.shape == (len(train), 2)
    assert np.allclose(probs.sum(axis=1), 1)
    for k, fitted in enumerate(seen):
        assert not fitted & {s.text for s, f in zip(train, folds, strict=True) if f == k}
    again, _ = dg.oof_probabilities(train, folds, n_features=60)
    assert np.allclose(probs, again)


def test_confident_learning_flags_mislabel():
    train = [
        snip("a", "python", "r/1"), snip("b", "python", "r/2"), snip("c", "python", "r/3"),
        snip("d", "go", "r/4"), snip("e", "go", "r/5"),
    ]  # fmt: skip
    classes = ["go", "python"]
    probs = np.array([
        [0.1, 0.9], [0.2, 0.8],
        [0.95, 0.05],  # labelled python, model is sure it is go
        [0.9, 0.1], [0.7, 0.3],
    ])  # fmt: skip
    joint, issues = dg.confident_learning(train, probs, classes)
    assert [(i.snippet.text, i.given, i.suggested) for i in issues] == [("c", "python", "go")]
    assert issues[0].p_suggested == pytest.approx(0.95)
    # thresholds: go 0.8 (mean of 0.9, 0.7), python 0.583; "e" (0.7 / 0.3) clears neither
    assert joint.tolist() == [[1, 0], [1, 2]]  # rows: given go, python; cols: counted as


def test_confident_learning_orders_by_margin():
    train = [snip(t, "python", f"r/{t}") for t in "abc"] + [snip("g", "go", "r/g")]
    probs = np.array([[0.6, 0.4], [0.99, 0.01], [0.1, 0.9], [0.5, 0.5]])
    _, issues = dg.confident_learning(train, probs, ["go", "python"])
    assert [i.snippet.text for i in issues] == ["b", "a"]


def test_shortcut_probe_counts_repos():
    train = []
    for r in range(6):  # "func" in every go repo; "zqx" only in go repo 0
        extra = "zqx = 1\n" if r == 0 else ""
        for i in range(2):
            train.append(snip(f"func F{r}{i}() {{}}\n{extra}", "go", f"go/r{r}", i))
            train.append(snip(f"def f{r}{i}(): pass\n", "python", f"py/r{r}", i))
    probe = dg.shortcut_probe(train, n_features=5000, top_k=5000)  # keep every n-gram
    go = {f.token: f for f in probe["go"]}
    python = {f.token: f for f in probe["python"]}
    assert set(probe) == {"go", "python"}
    assert go["zqx"].class_repos == 1 and go["zqx"].shortcut_suspect
    assert go["fun"].class_repos == 6 and not go["fun"].shortcut_suspect
    assert go["fun"].other_share == 0.0
    # binary problem: sklearn stores one coefficient row; python's is its negation
    assert go["fun"].weight > 0
    assert python["fun"].weight == pytest.approx(-go["fun"].weight)


def test_show_token():
    assert dg.show_token("a b") == "`a␠b`"
    assert dg.show_token(":\n") == "`:\\n`"
    assert dg.show_token("`x") == "`` `x ``"


def test_cli_end_to_end(processed, tmp_path, capsys):
    out = tmp_path / "diag.md"
    args = ["diagnose", "--data", str(processed), "--out", str(out), "--features", "60",
            "--top", "5"]  # fmt: skip
    assert main(args) == 0
    printed = capsys.readouterr().out
    train, _ = load(processed)
    assert f"label-issue candidates: 0 of {len(train)}" in printed  # clean synthetic data
    md = out.read_text(encoding="utf-8")
    assert "## 1. Label-issue candidates" in md and "### python" in md
    assert main(["diagnose", "--data", str(tmp_path / "none")]) == 1
