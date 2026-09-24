import json

import numpy as np
import pytest

from codelangtm import baselines as bl
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.data import Snippet, load_snippets, save_snippets
from codelangtm.features import Binarizer

LANGS = ("python", "go")
FAST = ["naive_bayes", "decision_tree", "logistic_regression"]


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j}\n}}\n" for j in range(8))
    return "package main\n\n" + body


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    raw = root / "raw"
    raw.mkdir()
    items = []
    for lang, make, ext in (("python", py_text, "py"), ("go", go_text, "go")):
        for r in range(10):
            for i in range(3):
                text = make(f"{lang}{r}x{i}")
                items.append(Snippet(text, lang, "github", f"{lang}-org/r{r}", "sha",
                                     f"src/f{i}.{ext}", "MIT", 1, text.count("\n")))  # fmt: skip
    save_snippets(items, raw / "all.jsonl")
    build_dataset([raw], root / "processed", languages=LANGS)
    return root / "processed"


def run(processed, models=FAST, repeats=2):
    return bl.run_baselines(processed, models, n_features=60, languages=LANGS, repeats=repeats)


def test_results_shape(processed):
    results, meta = run(processed)
    assert [r.name for r in results] == FAST
    for r in results:
        assert len(r.cv_f1) == 5
        assert 0 <= r.test_f1 <= 1 and 0 <= r.test_accuracy <= 1
        assert set(r.per_language_f1) == set(LANGS)
        assert np.array(r.confusion).sum() == meta["n_test"]
        assert r.latency_ms > 0 and r.size_kb > 0 and r.wild_f1 is None
        assert len(r.repeat_f1) == 2 and 0 <= r.repeat_mean <= 1
    assert meta["n_features"] == 60 and set(meta["dataset_files"]) >= {"train.jsonl"}
    assert meta["repeats"] == 2 and meta["split"]["split"] == "stable-hash"


def test_repeated_splits_differ_and_never_leak(processed):
    pool = [*load_snippets(processed / "train.jsonl"), *load_snippets(processed / "test.jsonl")]
    splits = [
        bl.stable_split(pool, 0.2, 2, salt=f"{bl.SPLIT_SALT}:repeat:{k}").test_repos
        for k in range(3)
    ]
    assert len(set(splits)) == 3  # independent draws
    scores = bl.repeated_split_f1(bl.make_models()["naive_bayes"], pool, Binarizer(60), 3)
    assert len(scores) == 3


def test_vocabulary_never_sees_held_out_text(processed, monkeypatch):
    seen: list[set[str]] = []
    original = Binarizer.fit

    def spy(self, snippets, y=None):
        snippets = list(snippets)
        seen.append(set(snippets))
        return original(self, snippets, y)

    monkeypatch.setattr(Binarizer, "fit", spy)
    bl.run_baselines(processed, ["naive_bayes"], n_features=60, languages=LANGS, repeats=0)

    test_texts = {s.text for s in load_snippets(processed / "test.jsonl")}
    train = list(load_snippets(processed / "train.jsonl"))
    folds = json.loads((processed / "folds.json").read_text(encoding="utf-8"))["folds"]
    assert len(seen) == 6  # 5 folds + final fit
    for fitted in seen:
        assert not fitted & test_texts
    for k, fitted in enumerate(seen[:5]):
        held_out = {s.text for s, f in zip(train, folds, strict=True) if f == k}
        assert not fitted & held_out


def test_deterministic_except_timings(processed):
    a, _ = run(processed)
    b, _ = run(processed)
    for x, y in zip(a, b, strict=True):
        assert (x.cv_f1, x.test_f1, x.confusion) == (y.cv_f1, y.test_f1, y.confusion)


def result(name, cv, test):
    return bl.ModelResult(name, cv, test, test, {"python": test, "go": test},
                          [[1, 0], [0, 1]], 0.1, 0.01, 1.0)  # fmt: skip


def test_best_selected_by_cv_not_test():
    a = result("a", [0.9, 0.9], 0.70)
    b = result("b", [0.8, 0.8], 0.99)
    assert bl.best_by_cv([a, b]) is a


def test_render_results_md():
    meta = {
        "generated": "2026-09-24 21:00:00", "data_dir": "d",
        "dataset_files": {"train.jsonl": "abc"},
        "n_train": 10, "n_test": 2, "n_wild": 0, "n_folds": 5, "n_features": 60, "seed": 0,
        "versions": {"scikit-learn": "x"}, "cpu": "cpu",
    }  # fmt: skip
    md = bl.render_results_md([result("a", [0.9], 0.8), result("b", [0.5], 0.9)], meta, LANGS)
    assert "| **a** | 0.900" in md and "Best by CV: **a**" in md
    assert "## Confusion matrix: a (test)" in md and "Most frequent confusions: none" in md


def test_unknown_model_rejected(processed):
    with pytest.raises(ValueError, match="unknown model"):
        bl.run_baselines(processed, ["gpt"], languages=LANGS)


def test_folds_mismatch_rejected(processed, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    for f in ("train.jsonl", "test.jsonl"):
        (bad / f).write_bytes((processed / f).read_bytes())
    (bad / "folds.json").write_text(json.dumps({"folds": [0, 1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="folds.json"):
        bl.run_baselines(bad, ["naive_bayes"], languages=LANGS)


def test_cli(processed, tmp_path, capsys):
    out = tmp_path / "results.md"
    args = ["baselines", "--data", str(processed), "--model", "naive_bayes",
            "--features", "60", "--repeats", "2", "--out", str(out)]  # fmt: skip
    assert main(args) == 0
    assert "best by CV: naive_bayes" in capsys.readouterr().out
    assert out.read_text(encoding="utf-8").startswith("# Results")
    assert main(["baselines", "--data", str(tmp_path / "none")]) == 1
