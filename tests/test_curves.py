import json
from pathlib import Path

import numpy as np
import pytest

from codelangtm import curves as cv
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.config import load_baselines_config, load_tm_config, parse_tm
from codelangtm.data import Snippet, load_snippets, save_snippets
from codelangtm.features import Binarizer

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("python", "go")


# ------------------------------------------------------------------ pure functions


def test_moving_average_shrinks_symmetrically_at_edges():
    assert cv.moving_average([1, 2, 3, 4, 5], 3).tolist() == [1, 2, 3, 4, 5]  # linear stays
    assert cv.moving_average([0, 9, 0, 9, 0], 3).tolist() == [0, 3, 6, 3, 0]
    assert cv.moving_average([0.6, 0.9, 0.9, 0.9, 0.9], 5).tolist()[0] == 0.6  # no future leak
    assert cv.moving_average([1, 2, 3], 1).tolist() == [1, 2, 3]


def test_choose_epoch_takes_start_of_plateau():
    curve = [0.5, 0.7, 0.85, 0.9, 0.92, 0.93, 0.93, 0.93, 0.94, 0.93]
    epoch, smooth = cv.choose_epoch(curve, window=1, tolerance=0.015)
    assert epoch == 6 and smooth.tolist() == curve  # target 0.94 - 0.015 = 0.925: first 0.93
    epoch, _ = cv.choose_epoch(curve, window=1, tolerance=0.0)
    assert epoch == 9  # zero tolerance = the maximum itself


def test_choose_epoch_ignores_a_single_spike():
    curve = [0.80] * 10 + [0.99] + [0.80] * 10  # one lucky epoch
    epoch_raw, _ = cv.choose_epoch(curve, window=1, tolerance=0.005)
    epoch_smooth, _ = cv.choose_epoch(curve, window=11, tolerance=0.005)
    assert epoch_raw == 11 and epoch_smooth < 11  # smoothing stops the spike deciding


# ------------------------------------------------------------------------- config


def tm_raw(**over):
    raw = {
        "data": "d", "seeds": [1, 2], "curves": {"max_epochs": 3, "smooth_window": 1},
        "tm": {"small": {"n_clauses": 10, "T": 5, "s": 3.0, "epochs": 2}},
    }  # fmt: skip
    raw.update(over)
    return raw


def test_parse_tm_valid():
    cfg = parse_tm(tm_raw())
    assert cfg.seeds == (1, 2) and cfg.curves.max_epochs == 3
    assert cfg.setting("small").kwargs() == {"n_clauses": 10, "T": 5, "s": 3.0, "epochs": 2}
    with pytest.raises(ValueError, match="unknown TM setting"):
        cfg.setting("big")


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"seeds": [0, 1]}, "integer >= 1"),
        ({"seeds": [1, 1]}, "duplicate"),
        ({"tm": {"x": {"n_clauses": 9}}}, "must be even"),
        ({"tm": {"x": {"seed": 3}}}, "unknown key"),
        ({"tm": {"x": {"s": 0.5}}}, "s: expected a number >= 1"),
        ({"tm": {}}, "config.tm"),
        ({"curves": {"smooth_window": 4}}, "must be odd"),
        ({"curves": {"plateau_tolerance": -1}}, ">= 0"),
        ({"epochz": 3}, "unknown key"),
    ],
)
def test_parse_tm_rejects(over, message):
    with pytest.raises(ValueError, match=message):
        parse_tm(tm_raw(**over))


def test_repo_tm_config_matches_frozen_features():
    tm = load_tm_config(ROOT / "configs" / "tm.yaml")
    assert tm.features == load_baselines_config(ROOT / "configs" / "baselines.yaml").features
    assert all(s >= 1 for s in tm.seeds) and len(tm.seeds) >= 5
    assert {s.name for s in tm.settings} >= {"tm_400", "tm_100"}


# ------------------------------------------------------------ runs (TMU required)


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


def small_cfg(processed, tmp_path=None):
    over = {"data": processed.as_posix(), "features": {"n_features": 40}}
    if tmp_path is not None:
        over["curves_out"] = (tmp_path / "curves.md").as_posix()
    return parse_tm(tm_raw(**over))


@pytest.fixture(scope="module")
def curve_runs(processed):
    pytest.importorskip("tmu")
    return cv.run_curves(small_cfg(processed), languages=LANGS)


def test_run_curves_shape(curve_runs):
    runs, meta = curve_runs
    assert list(runs) == ["small"] and len(runs["small"]) == 5 * 2  # folds x seeds
    for r in runs["small"]:
        for series in (r.val_f1, r.train_f1, r.epoch_seconds, r.included, r.nonempty, r.changed):
            assert len(series) == 3
        assert all(0 <= f <= 1 for f in r.val_f1 + r.train_f1)
        assert r.changed[0] == r.included[0]  # first epoch: every include decision is new
    assert meta["seeds"] == [1, 2] and "tmu" in meta["versions"]


def test_curves_are_deterministic(processed, curve_runs):
    again, _ = cv.run_curves(small_cfg(processed), languages=LANGS)
    first, _ = curve_runs
    assert [r.val_f1 for r in again["small"]] == [r.val_f1 for r in first["small"]]


def test_curves_never_see_test_or_held_out_text(processed, monkeypatch):
    pytest.importorskip("tmu")
    seen: list[set[str]] = []
    original = Binarizer.fit

    def spy(self, snippets, y=None):
        snippets = list(snippets)
        seen.append(set(snippets))
        return original(self, snippets, y)

    monkeypatch.setattr(Binarizer, "fit", spy)
    cv.run_curves(parse_tm(tm_raw(data=processed.as_posix(), seeds=[1],
                                  curves={"max_epochs": 1})), languages=LANGS)  # fmt: skip
    test_texts = {s.text for s in load_snippets(processed / "test.jsonl")}
    train = list(load_snippets(processed / "train.jsonl"))
    folds = json.loads((processed / "folds.json").read_text(encoding="utf-8"))["folds"]
    assert len(seen) == 5  # one binarizer per fold
    for k, fitted in enumerate(seen):
        held_out = {s.text for s, f in zip(train, folds, strict=True) if f == k}
        assert not fitted & test_texts and not fitted & held_out


def test_json_and_markdown(processed, curve_runs):
    runs, meta = curve_runs
    data = cv.curves_json(small_cfg(processed), runs, meta)
    assert data["schema"] == cv.CURVES_SCHEMA
    s = data["settings"]["small"]
    assert len(s["runs"]) == 10 and s["n_clauses_total"] == 10 * 2
    sm = s["summary"]
    assert 1 <= sm["chosen_epoch"] <= 3 and len(sm["val_mean"]) == 3
    mean = np.mean([r["val_f1"] for r in s["runs"]], axis=0)
    assert np.allclose(sm["val_mean"], mean)
    md = cv.render_curves_md(json.loads(json.dumps(data)))
    assert md.startswith("# TM training curves") and "## Recommendation" in md
    assert "| **small** |" in md and "| 1 |" in md and "| 5 |" not in md  # only epochs <= 3
    assert "test set is not used" in md


def test_cli_tm_curve(processed, tmp_path, capsys):
    pytest.importorskip("tmu")
    cfg = tmp_path / "tm.yaml"
    cfg.write_text(
        f"data: {processed.as_posix()}\ncurves_out: {(tmp_path / 'c.md').as_posix()}\n"
        "seeds: [1]\ncurves: {max_epochs: 2, smooth_window: 1}\nfeatures: {n_features: 30}\n"
        "tm:\n  a: {n_clauses: 10, T: 5}\n  b: {n_clauses: 6, T: 3}\n",
        encoding="utf-8",
    )
    assert main(["tm-curve", "--config", str(cfg), "--setting", "b"]) == 0
    out = capsys.readouterr().out
    assert "[5/5] b fold 4 seed 1" in out and "set `epochs`" in out
    data = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert list(data["settings"]) == ["b"]
    assert main(["tm-curve", "--config", str(cfg), "--setting", "zz"]) == 1
    assert "unknown TM setting" in capsys.readouterr().err
