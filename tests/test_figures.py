import json
import subprocess
import sys

import numpy as np
import pytest

from codelangtm import baselines as bl
from codelangtm import figures as fg

LANGS = ("python", "go", "sql")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def results_sidecar():
    def result(name, cv, test):
        r = bl.ModelResult(
            name, cv, test, test, {"python": test, "go": 1.0, "sql": test - 0.1},
            [[5, 1, 0], [0, 6, 0], [1, 0, 4]], 0.4, 0.03, 35.0,
            repeat_f1=[test - 0.01, test + 0.01], fit_cpu_seconds=0.5,
            peak_binarize_mb=70.0, peak_model_mb=1.2, vocab_kb=5.0,
        )  # fmt: skip
        return r

    meta = {
        "n_train": 40, "n_test": 17,
        "composition": {
            "train": {lang: {"snippets": 13, "repos": 4} for lang in LANGS},
            "test": {lang: {"snippets": 6, "repos": 1} for lang in LANGS},
            "wild": {},
        },
    }  # fmt: skip
    models = [result("naive_bayes", [0.95, 0.97], 0.96), result("random_forest", [0.9], 0.93)]
    models[1].size_kb = 7000.0  # log-scale panel
    return json.loads(json.dumps(bl._json_ready(bl.results_json(models, meta, LANGS))))


def setting(i, base, **features):
    full = {"n_features": 500, "ngram_sizes": [2, 3], "use_delimiters": True,
            "selection": "frequency", "min_df": 1, "word_tokens": False, **features}  # fmt: skip
    score = 0.8 + 0.01 * i
    run = {"cv_f1": [score, score], "cv_mean": score, "cv_std": 0.01, "delta_mean": 0.01 * i,
           "delta_std": 0.005, "per_language_f1": dict.fromkeys(LANGS, score),
           "fit_seconds": 0.1}  # fmt: skip
    return {"id": i, "label": f"setting {i}", "is_base": base, "features": full,
            "vocab_size": 400, "binarize_ms": 0.1,
            "models": {"logistic_regression": run, "naive_bayes": run}}  # fmt: skip


def ablations_sidecar():
    settings = [
        setting(0, True),
        setting(1, False, n_features=100),
        setting(2, False, selection="chi2"),
        setting(3, False, selection="chi2", n_features=2000),
        setting(4, False, selection="class_balanced", ngram_sizes=[2]),
        setting(5, False, ngram_sizes=[2]),
    ]
    return {
        "schema": "codelangtm.ablations/1",
        "meta": {},
        "languages": list(LANGS),
        "models": ["logistic_regression", "naive_bayes"],
        "base": 0,
        "studies": [
            {"name": "vocabulary_size", "varied": ["n_features"], "settings": [1, 0]},
            {"name": "selection_x_vocabulary", "varied": ["selection", "n_features"],
             "settings": [0, 2, 3]},
            {"name": "selection_x_ngrams", "varied": ["selection", "ngram_sizes"],
             "settings": [5, 0, 2, 4]},
        ],  # fmt: skip
        "settings": settings,
        "best_by_cv": {"logistic_regression": 5, "naive_bayes": 5},
    }


def curves_sidecar(epochs=12):
    rng = np.random.default_rng(0)

    def summary(level):
        val = (level - 0.3 * np.exp(-np.arange(epochs) / 3) + rng.normal(0, 0.01, epochs)).tolist()
        return {
            "chosen_epoch": 6, "smoothed_at_chosen": level - 0.01, "smoothed_max": level,
            "smoothed_max_epoch": 9, "tail_mean": level, "tail_epochs": 5,
            "val_mean": val, "val_std_runs": [0.02] * epochs, "val_std_folds": [0.02] * epochs,
            "val_smoothed": val, "train_mean": [min(1.0, v + 0.05) for v in val],
            "epoch_seconds_median": [0.03] * epochs,
            "included_mean": [1000.0 + 50 * e for e in range(epochs)],
            "nonempty_mean": [100.0] * epochs,
            "changed_mean": [1000.0 / (e + 1) for e in range(epochs)],
        }  # fmt: skip

    return {
        "schema": "codelangtm.tm-curves/1",
        "meta": {},
        "settings": {
            "tm_400": {"params": {"n_clauses": 400, "T": 100, "s": 5.0}, "summary": summary(0.96)},
            "tm_100": {"params": {"n_clauses": 100, "T": 30, "s": 3.5}, "summary": summary(0.93)},
        },
    }


def tm_results_sidecar():
    def base(name, cv, test):
        return {"name": name, "cv_mean": cv, "cv_std": 0.01, "repeat_mean": cv + 0.005,
                "repeat_std": 0.01, "test_f1": test,
                "per_language_f1": {"python": test, "go": 1.0, "sql": test - 0.05}}  # fmt: skip

    def tm(cv, test):
        return {"params": {"n_clauses": 400, "epochs": 120}, "seeds": [{}, {}, {}],
                "aggregate": {"cv_mean": cv, "cv_std_folds": 0.02, "repeat_mean": cv + 0.01,
                              "repeat_std_splits": 0.015, "test_mean": test,
                              "test_min": test - 0.01, "test_max": test + 0.01,
                              "per_language_f1": {"python": test, "go": 0.98,
                                                  "sql": 0.9}}}  # fmt: skip

    return {
        "schema": "codelangtm.tm-results/1",
        "languages": list(LANGS),
        "baselines": [base("naive_bayes", 0.963, 0.959),
                      base("logistic_regression", 0.953, 0.971)],
        "tm": {"tm_400": tm(0.95, 0.94), "tm_100": tm(0.93, 0.92)},
    }


def resources_sidecar():
    def row(name, wall, total):
        return {"name": name, "fit": {"wall_s": wall, "added_mb": 76.0},
                "predict": {"binarize_ms_median": 0.04, "predict_ms_median": total - 0.04,
                            "total_ms_p95": total * 1.3}}  # fmt: skip

    return {"schema": "codelangtm.resources/1",
            "models": [row("naive_bayes", 0.35, 0.2), row("tm_400 (seed 4)", 7.7, 0.34)]}


def clauses_sidecar(formation=True):
    def clause(i, polarity, n, fires):
        weight = (1 + i % 7) * (1 if polarity == "for" else -1)
        return {"index": i, "polarity": polarity, "weight": weight, "n_literals": n,
                "fires_own": fires, "fires_other": i % 3}  # fmt: skip

    clauses = [clause(i, ("for", "against")[i % 2], (i * 7) % 60, (i * 5) % 40) for i in range(90)]
    data = {
        "schema": "codelangtm.clauses/1",
        "languages": list(LANGS),
        "signature_matrix": {"features": ['"def"', '":="', '"SEL"'],
                             "owner": list(LANGS), "ids": [0, 1, 2],
                             "usage": [[0.12, 0.0, 0.004], [0.01, 0.1, 0.0],
                                       [0.0, 0.0, 0.03]]},  # fmt: skip
        "overlap": [[1.0, 0.2, 0.25], [0.2, 1.0, 0.18], [0.25, 0.18, 1.0]],
        "clauses": clauses,
        "formation": None,
    }
    if formation:
        settled = {"python": {'"def"': 1, '"):"': 12}, "go": {'":="': 3}, "sql": {'"SEL"': None}}
        data["formation"] = {"epochs": 30, "jaccard_with_final": [i / 30 for i in range(1, 31)],
                             "settled_epoch": settled}  # fmt: skip
    return data


def errors_sidecar():
    def flows(counts):
        total = sum(counts)
        return {g: {"count": c, "share": c / total} for g, c in zip(LANGS, counts, strict=True)}

    return {
        "schema": "codelangtm.errors/1", "setting": "tm_400", "seeds": [1, 2, 3],
        "baselines": ["naive_bayes"], "languages": list(LANGS),
        "models": {"tm": {"confusion": [[90, 3, 0], [2, 95, 1], [0, 6, 80]]}},
        "inflow": {"tm": flows([2, 9, 1]), "naive_bayes": flows([1, 2, 1])},
        "embedded": {"host": "html", "guest": "javascript",
                     "points": [[0.0, 0], [0.7, 3], [0.9, 0], [0.2, 0]]},
    }  # fmt: skip


ALL_FIGURES = (set(fg.RESULTS_FIGURES) | set(fg.ABLATION_FIGURES) | set(fg.CURVE_FIGURES)
               | set(fg.TM_RESULT_FIGURES) | set(fg.RESOURCE_FIGURES)
               | set(fg.CLAUSE_FIGURES) | set(fg.ERROR_FIGURES))  # fmt: skip


@pytest.fixture
def sidecars(tmp_path):
    pytest.importorskip("matplotlib")
    results, ablations = tmp_path / "results.json", tmp_path / "ablations.json"
    bl.write_json(results_sidecar(), results)
    bl.write_json(ablations_sidecar(), ablations)
    curves = bl.write_json(curves_sidecar(), tmp_path / "tm-curves.json")
    tm_results = bl.write_json(tm_results_sidecar(), tmp_path / "tm-results.json")
    res = bl.write_json(resources_sidecar(), tmp_path / "resources.json")
    clauses = bl.write_json(clauses_sidecar(), tmp_path / "clauses.json")
    errors = bl.write_json(errors_sidecar(), tmp_path / "errors.json")
    return results, ablations, curves, tm_results, res, clauses, errors


def test_render_all_writes_every_figure(sidecars, tmp_path):
    written = fg.render_all(sidecars[0], sidecars[1], tmp_path / "figs", *sidecars[2:])
    names = {p.stem for p in written}
    assert names == ALL_FIGURES
    for path in written:
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE) and len(data) > 5_000
        assert b"matplotlib" not in data and b"Software" not in data  # no metadata stamps


def test_figures_are_byte_identical_across_runs(sidecars, tmp_path):
    a = fg.render_all(sidecars[0], sidecars[1], tmp_path / "a", *sidecars[2:])
    b = fg.render_all(sidecars[0], sidecars[1], tmp_path / "b", *sidecars[2:])
    for x, y in zip(a, b, strict=True):
        assert x.read_bytes() == y.read_bytes(), x.name


def test_only_given_sidecars_are_drawn(sidecars, tmp_path):
    written = fg.render_all(sidecars[0], None, tmp_path / "r")
    assert {p.stem for p in written} == set(fg.RESULTS_FIGURES)


def test_sidecar_errors(sidecars, tmp_path):
    with pytest.raises(FileNotFoundError, match="sidecar not found"):
        fg.render_all(tmp_path / "missing.json", None, tmp_path)
    with pytest.raises(ValueError, match="expected schema codelangtm.results/"):
        fg.render_all(sidecars[1], None, tmp_path)  # ablations file passed as results


def test_clause_figures_without_formation(sidecars, tmp_path):
    path = bl.write_json(clauses_sidecar(formation=False), tmp_path / "c.json")
    written = fg.render_all(None, None, tmp_path / "c", clauses_path=path)
    assert {p.stem for p in written} == set(fg.CLAUSE_FIGURES) - {"clause_formation_replay"}


def test_error_figures_skip_embedded_without_points(sidecars, tmp_path):
    data = {**errors_sidecar(), "embedded": None}
    path = bl.write_json(data, tmp_path / "e.json")
    written = fg.render_all(None, None, tmp_path / "e", errors_path=path)
    assert {p.stem for p in written} == set(fg.ERROR_FIGURES) - {"errors_embedded"}


def test_errors_confusion_greys_the_diagonal(sidecars):
    ax = fg.errors_confusion(errors_sidecar()).axes[0]
    assert ax.images[0].get_array()[0][0] == 0  # diagonal carries no colour
    assert ax.images[0].get_array()[2][1] == pytest.approx(6 / 86)
    colours = {t.get_text(): t.get_color() for t in ax.texts}
    assert colours["90"] == fg.MUTED


def test_clause_signature_cells_show_small_shares(sidecars):
    ax = fg.clause_signatures(clauses_sidecar()).axes[0]
    shown = sorted(t.get_text() for t in ax.texts)
    assert shown == sorted(["12", "<1", "1", "10", "3"])  # percent; exact zeros left blank


def test_confusion_defaults_to_best_and_colours_by_row_share(sidecars):
    fig = fg.confusion(results_sidecar())
    ax = fig.axes[0]
    assert "naive bayes" in ax.get_title(loc="left")
    shown = sorted(t.get_text() for t in ax.texts)
    assert shown == sorted(["5", "1", "6", "1", "4"])  # counts, zero cells left blank
    assert ax.images[0].get_array()[0][0] == pytest.approx(5 / 6)


def test_selection_methods_keep_their_colour(sidecars):
    ax = fg.ablation_vocabulary(ablations_sidecar()).axes[0]
    colours = {line.get_label(): line.get_color() for line in ax.get_lines()
               if not line.get_label().startswith("_")}  # fmt: skip
    assert colours == {"frequency": fg.SERIES[0], "chi2": fg.SERIES[1]}  # no class_balanced here
    freq = next(line for line in ax.get_lines() if line.get_label() == "frequency")
    assert list(freq.get_xdata()) == [100, 500]  # sorted by M, base shared by two studies


def test_diff_label():
    base = {"n_features": 500, "ngram_sizes": [2, 3], "use_delimiters": True,
            "selection": "frequency", "min_df": 1, "word_tokens": False}  # fmt: skip
    assert fg.diff_label(base, base) == "base"
    changed = {**base, "n_features": 2000, "selection": "class_balanced", "word_tokens": True}
    assert fg.diff_label(changed, base) == "M=2000, class balanced, words on"
    assert fg.diff_label({**base, "ngram_sizes": [2]}, base) == "n=2"


def test_number_format():
    assert fg._num(6995.2) == "6,995" and fg._num(0.03214) == "0.0321" and fg._num(35.25) == "35.2"


def test_core_imports_without_matplotlib():
    # Block matplotlib in a fresh interpreter: the package and CLI still import, and drawing
    # fails with the install hint instead of a bare ModuleNotFoundError.
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import codelangtm.cli, codelangtm.figures as fg, codelangtm.baselines\n"
        "try:\n"
        "    fg.render_all(None, None, '.')\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert fg.INSTALL_HINT in out.stdout


def test_cli_report(sidecars, tmp_path, capsys):
    from codelangtm.cli import main

    out = tmp_path / "figs"
    args = ["report", "--results", str(sidecars[0]), "--ablations", str(sidecars[1]),
            "--out", str(out)]  # fmt: skip
    missing = ["--curves", str(tmp_path / "no.json"), "--tm-results", str(tmp_path / "no2.json"),
               "--resources", str(tmp_path / "no3.json"), "--clauses", str(tmp_path / "no4.json"),
               "--errors", str(tmp_path / "no5.json")]
    assert main([*args, *missing]) == 0
    printed = capsys.readouterr().out
    assert "wrote" in printed and "skipping TM curve figures" in printed
    assert "skipping TM result figures" in printed
    assert {p.stem for p in out.glob("*.png")} == set(fg.RESULTS_FIGURES) | set(
        fg.ABLATION_FIGURES
    )
    assert main([*args, "--curves", str(sidecars[2]), "--tm-results", str(sidecars[3]),
                 "--resources", str(sidecars[4]), "--clauses", str(sidecars[5]),
                 "--errors", str(sidecars[6])]) == 0
    assert {p.stem for p in out.glob("*.png")} == ALL_FIGURES
    assert main(["report", "--results", str(tmp_path / "none.json"), "--out", str(out)]) == 1
    assert "sidecar not found" in capsys.readouterr().err
