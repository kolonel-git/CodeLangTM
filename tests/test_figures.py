import json
import subprocess
import sys

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


@pytest.fixture
def sidecars(tmp_path):
    pytest.importorskip("matplotlib")
    results, ablations = tmp_path / "results.json", tmp_path / "ablations.json"
    bl.write_json(results_sidecar(), results)
    bl.write_json(ablations_sidecar(), ablations)
    return results, ablations


def test_render_all_writes_every_figure(sidecars, tmp_path):
    written = fg.render_all(*sidecars, tmp_path / "figs")
    names = {p.stem for p in written}
    assert names == set(fg.RESULTS_FIGURES) | set(fg.ABLATION_FIGURES)
    for path in written:
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE) and len(data) > 5_000
        assert b"matplotlib" not in data and b"Software" not in data  # no metadata stamps


def test_figures_are_byte_identical_across_runs(sidecars, tmp_path):
    a = fg.render_all(*sidecars, tmp_path / "a")
    b = fg.render_all(*sidecars, tmp_path / "b")
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
