from dataclasses import replace

import pytest

from codelangtm import ablations as ab
from codelangtm import baselines as bl
from codelangtm.audit import load_dataset
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.config import FeatureConfig, parse_ablations
from codelangtm.data import Snippet, load_snippets, save_snippets
from codelangtm.features import Binarizer

LANGS = ("python", "go")
BASE = {"n_features": 60, "ngram_sizes": [2, 3], "use_delimiters": True}


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


def make_cfg(processed, studies=None, models=("naive_bayes", "logistic_regression")):
    return parse_ablations({
        "data": processed.as_posix(), "models": list(models), "base": BASE,
        "studies": studies or {
            "vocab": {"n_features": [30, 60]},
            "delims": {"use_delimiters": [True, False]},
        },
    })  # fmt: skip


def test_shared_binarizer_equals_pipeline(processed):
    # The ablation shortcut (one binarizer per fold, shared by models) must give exactly the
    # CV scores of the baselines Pipeline.
    dataset = load_dataset(processed)
    folds = bl.load_folds(processed)
    train = dataset["train"]
    models = {k: v for k, v in bl.make_models().items() if k in ("naive_bayes", "decision_tree")}
    feats = FeatureConfig(60, (2, 3), True)
    runs = ab.evaluate_features(
        feats, [s.text for s in train], [s.language for s in train], folds, models, LANGS
    )
    for name, model in models.items():
        expected = bl.evaluate_model(name, model, dataset, folds, feats.binarizer(), LANGS)
        assert runs[name].cv_f1 == expected.cv_f1


def test_run_ablations_evaluates_each_setting_once(processed):
    cfg = make_cfg(processed)
    seen = []
    runs, meta = ab.run_ablations(cfg, LANGS, progress=lambda i, n, f: seen.append((i, n, f)))
    # base (60, delimiters on) appears in both studies but is evaluated once
    assert [f.n_features for _, _, f in seen] == [60, 30, 60]
    assert [f.use_delimiters for _, _, f in seen] == [True, True, False]
    assert all(n == 3 for _, n, _ in seen)
    assert len(runs) == 3 * 2 and meta["n_settings"] == 3
    for run in runs.values():
        assert len(run.cv_f1) == 5 and set(run.per_language_f1) == set(LANGS)
        assert run.vocab_size <= run.features.n_features and run.binarize_ms > 0
    base = runs[cfg.base, "naive_bayes"]
    assert ab.paired_delta(base, base) == (0.0, 0.0)


def test_vocab_size_reports_actual_vocabulary(processed):
    cfg = make_cfg(processed, {"big": {"n_features": [100_000]}}, models=["naive_bayes"])
    runs, _ = ab.run_ablations(cfg, LANGS)
    big = runs[replace(cfg.base, n_features=100_000), "naive_bayes"]
    assert big.vocab_size < 100_000  # capped by n-grams that exist


def test_never_fits_on_test_snippets(processed, monkeypatch):
    fitted: list[set[str]] = []
    original = Binarizer.fit

    def spy(self, snippets, y=None):
        snippets = list(snippets)
        fitted.append(set(snippets))
        return original(self, snippets, y)

    monkeypatch.setattr(Binarizer, "fit", spy)
    ab.run_ablations(make_cfg(processed, models=["naive_bayes"]), LANGS)
    test_texts = {s.text for s in load_snippets(processed / "test.jsonl")}
    assert len(fitted) == 3 * 5  # settings x folds, no final fit on all of train
    assert all(not f & test_texts for f in fitted)


def test_select_studies(processed):
    cfg = make_cfg(processed)
    assert ab.select_studies(cfg, None) is cfg
    only = ab.select_studies(cfg, ["delims"])
    assert [s.name for s in only.studies] == ["delims"]
    assert len(only.feature_settings()) == 2
    with pytest.raises(ValueError, match="unknown study"):
        ab.select_studies(cfg, ["nope"])


def test_unknown_model_rejected(processed):
    with pytest.raises(ValueError, match="unknown model"):
        ab.run_ablations(make_cfg(processed, models=["gpt"]), LANGS)


def test_render_markdown(processed):
    cfg = make_cfg(processed)
    runs, meta = ab.run_ablations(cfg, LANGS)
    meta["config_path"] = "configs/x.yaml"
    md = ab.render_ablations_md(cfg, runs, meta, LANGS)
    assert md.startswith("# Feature ablations")
    assert "## vocab" in md and "## delims" in md and "## Best setting by CV" in md
    assert "**M=60** (base)" in md and "| M=30 |" in md and "| delimiters off |" in md
    assert "`configs/x.yaml`" in md and "test set is not used" in md
    assert "setting" in ab.summary_table(cfg, runs)


def test_cli(processed, tmp_path, capsys):
    out = tmp_path / "ablations.md"
    cfg = tmp_path / "abl.yaml"
    cfg.write_text(
        f"data: {processed.as_posix()}\nmodels: [naive_bayes]\n"
        "base: {n_features: 60}\n"
        "studies:\n  vocab: {n_features: [30, 60]}\n  delims: {use_delimiters: [false]}\n",
        encoding="utf-8",
    )
    assert main(["ablate", "--config", str(cfg), "--study", "vocab", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "[1/2]" in printed and f"wrote {out}" in printed
    md = out.read_text(encoding="utf-8")
    assert "## vocab" in md and "## delims" not in md

    assert main(["ablate", "--config", str(cfg), "--study", "nope"]) == 1
    assert "unknown study" in capsys.readouterr().err
    assert main(["ablate", "--config", str(tmp_path / "missing.yaml")]) == 1
