import json
import math

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import f1_score

from codelangtm import tm_results as tr
from codelangtm.audit import load_dataset
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.config import parse_tm
from codelangtm.data import Snippet, save_snippets
from codelangtm.model import load_model

LANGS = ("python", "go")

# ------------------------------------------------------------------ statistics


def test_uncorrected_case_equals_paired_t_test():
    rng = np.random.default_rng(0)
    a, b = rng.random(10), rng.random(10)
    res = tr.corrected_ttest(a - b, test_train_ratio=0.0)
    ref = stats.ttest_rel(a, b)
    assert res["t"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
    assert res["ci_low"] < res["mean"] < res["ci_high"] and res["df"] == 9


def test_correction_widens_the_interval():
    d = [0.01, 0.02, -0.005, 0.015, 0.01, 0.0, 0.02, 0.012, 0.008, 0.011]
    plain = tr.corrected_ttest(d, 0.0)
    corrected = tr.corrected_ttest(d, 0.25)
    assert corrected["mean"] == plain["mean"]
    assert corrected["ci_high"] - corrected["ci_low"] > plain["ci_high"] - plain["ci_low"]
    assert corrected["p"] > plain["p"]
    # by hand: se = sqrt((1/J + ratio) * var)
    se = math.sqrt((1 / 10 + 0.25) * np.var(d, ddof=1))
    assert corrected["t"] == pytest.approx(np.mean(d) / se)


def test_ttest_edge_cases():
    assert tr.corrected_ttest([0.0, 0.0, 0.0], 0.25)["p"] == 1.0
    zero_var = tr.corrected_ttest([0.01, 0.01, 0.01], 0.25)
    assert zero_var["p"] == 0.0 and zero_var["t"] == math.inf
    with pytest.raises(ValueError, match="at least 2"):
        tr.corrected_ttest([0.1], 0.25)


# ------------------------------------------------------------------ runs (TMU)


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


def small_cfg(processed, **over):
    raw = {
        "data": processed.as_posix(), "seeds": [1, 2], "repeats": 2,
        "baselines": ["naive_bayes"], "features": {"n_features": 40},
        "tm": {"small": {"n_clauses": 10, "T": 5, "s": 3.0, "epochs": 2}},
    }  # fmt: skip
    raw.update(over)
    return parse_tm(raw)


@pytest.fixture(scope="module")
def run(processed, tmp_path_factory):
    pytest.importorskip("tmu")
    models = tmp_path_factory.mktemp("models")
    baselines, entries, meta = tr.run_tm_results(small_cfg(processed), languages=LANGS,
                                                 models_dir=models)  # fmt: skip
    return baselines, entries, meta, models


def test_protocol_shapes(run):
    baselines, entries, meta, _ = run
    assert [b.name for b in baselines] == ["naive_bayes"]
    (entry,) = entries
    assert entry.setting == "small" and [r.seed for r in entry.runs] == [1, 2]
    for r in entry.runs:
        assert len(r.result.cv_f1) == 5 and len(r.result.repeat_f1) == 2
        assert r.result.pipeline is None  # freed after use
        assert r.n_clauses_total == 2 * 10 and r.model_json_kb > 0 and r.tmu_latency_ms > 0
    assert entry.cv_by_fold.shape == (5,) and entry.repeat_by_split.shape == (2,)
    assert meta["cv_test_train_ratio"] == pytest.approx(0.25)
    assert 0.1 < meta["repeated_test_train_ratio"] < 0.5 and meta["seeds"] == [1, 2]


def test_saved_models_reproduce_the_test_score(run, processed):
    _, entries, _, models = run
    test = load_dataset(processed)["test"]
    y = np.asarray([s.language for s in test])
    for r in entries[0].runs:
        loaded = load_model(models / f"small_seed{r.seed}.json")
        assert loaded.meta["seed"] == r.seed
        pred = loaded.predict([s.text for s in test])
        present = sorted(set(y))
        f1 = f1_score(y, pred, labels=present, average="macro", zero_division=0)
        assert f1 == pytest.approx(r.result.test_f1)


def test_significance_rows(run):
    baselines, entries, meta, _ = run
    rows = tr.significance(baselines, entries, meta)
    assert [(r["tm"], r["baseline"], r["evaluation"]) for r in rows] == [
        ("small", "naive_bayes", "CV folds"), ("small", "naive_bayes", "repeated test splits"),
    ]  # fmt: skip
    cv = rows[0]
    expected = entries[0].cv_by_fold - np.asarray(baselines[0].cv_f1)
    assert cv["mean"] == pytest.approx(expected.mean()) and cv["n_splits"] == 5


def test_json_and_markdown(run, processed):
    baselines, entries, meta, _ = run
    data = json.loads(json.dumps(tr.tm_results_json(small_cfg(processed), baselines, entries,
                                                    meta, LANGS), default=float))  # fmt: skip
    assert data["schema"] == tr.TM_RESULTS_SCHEMA
    small = data["tm"]["small"]
    assert len(small["seeds"]) == 2 and small["seeds"][0]["model_path"].endswith("seed1.json")
    agg = small["aggregate"]
    assert agg["test_mean"] == pytest.approx(np.mean([s["test_f1"] for s in small["seeds"]]))
    assert np.sum(agg["confusion_sum"]) == 2 * meta["n_test"]
    md = tr.render_tm_results_md(data)
    assert md.startswith("# TM results") and "| **small** (10 clauses, 2 epochs) |" in md
    assert "## Significance: TM minus baseline" in md and "Nadeau & Bengio" in md
    assert "summed over 2 seeds" in md and "tm-curves.md" in md


def test_cli_tm_results(processed, tmp_path, capsys):
    pytest.importorskip("tmu")
    cfg = tmp_path / "tm.yaml"
    cfg.write_text(
        f"data: {processed.as_posix()}\nout: {(tmp_path / 'r.md').as_posix()}\n"
        "seeds: [1]\nrepeats: 2\nbaselines: [naive_bayes]\nfeatures: {n_features: 30}\n"
        "tm:\n  a: {n_clauses: 6, T: 3, epochs: 1}\n",
        encoding="utf-8",
    )
    models = tmp_path / "m"
    assert main(["tm-results", "--config", str(cfg), "--models-dir", str(models),
                 "--no-save-models"]) == 0  # fmt: skip
    assert "a - naive_bayes (CV folds)" in capsys.readouterr().out
    assert (tmp_path / "r.json").exists() and not models.exists()
    assert main(["tm-results", "--config", str(cfg), "--setting", "zz"]) == 1
    assert "unknown TM setting" in capsys.readouterr().err
