"""Training curves (M3 step B4): how the TM learns, epoch by epoch, on CV folds of train only.

For every TM setting, fold and seed, the TM is trained one epoch at a time. After each epoch we
record macro-F1 on the held-out fold and on the fold's training part, the epoch time, and how the
clauses look (included literals, non-empty clauses, include decisions changed since the previous
epoch). The held-out curve, averaged over folds and seeds and smoothed, picks the epoch count for
the full protocol run (B3). The test set is never used here.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import LANGUAGES
from .audit import load_dataset
from .baselines import _macro_f1, dataset_meta, load_folds
from .config import CurveConfig, TMConfig, TMSetting, config_dict, config_hash
from .model import TMLanguageClassifier

CURVES_SCHEMA = "codelangtm.tm-curves/1"
CHECKPOINTS = (1, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 300)
TAIL = 20  # "last epochs" window for the end-of-scan average


@dataclass
class CurveRun:
    setting: str
    fold: int
    seed: int
    val_f1: list[float]
    train_f1: list[float]
    epoch_seconds: list[float]  # one training epoch incl. copying the clauses out of TMU
    included: list[int]  # included literals, all clauses
    nonempty: list[int]  # clauses with at least one literal
    changed: list[int]  # include decisions that differ from the previous epoch


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Centred moving average. Near the ends the window shrinks symmetrically (epoch 1 is its own
    value), so an edge point is never pulled towards later or earlier epochs only."""
    values = np.asarray(values, dtype=float)
    n, half = len(values), window // 2
    out = np.empty_like(values)
    for i in range(n):
        r = min(half, i, n - 1 - i)
        out[i] = values[i - r : i + r + 1].mean()
    return out


def choose_epoch(
    mean_curve: Sequence[float], window: int, tolerance: float, rule: str = "plateau"
) -> tuple[int, np.ndarray]:
    """(epoch, smoothed curve), 1-based. Rule "plateau": the first epoch whose smoothed F1 is
    within `tolerance` of the smoothed maximum (where the curve levels off). Rule "max": the
    epoch of the smoothed maximum itself (first one on ties)."""
    smooth = moving_average(mean_curve, window)
    if rule == "max":
        return int(np.argmax(smooth)) + 1, smooth
    if rule != "plateau":
        raise ValueError(f"unknown epoch rule {rule!r}")
    target = smooth.max() - tolerance
    return int(np.argmax(smooth >= target)) + 1, smooth


def train_curve(setting: TMSetting, seed: int, fold: int, x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray, classes: Sequence[str],
                max_epochs: int) -> CurveRun:  # fmt: skip
    model = TMLanguageClassifier(**setting.kwargs(), seed=seed)
    run = CurveRun(setting.name, fold, seed, [], [], [], [], [], [])
    previous = None
    for _ in range(max_epochs):
        start = time.perf_counter()
        model.partial_fit(x_train, y_train, classes=classes)
        run.epoch_seconds.append(time.perf_counter() - start)
        state = model.state_
        run.val_f1.append(_macro_f1(y_val, state.predict(x_val)))
        run.train_f1.append(_macro_f1(y_train, state.predict(x_train)))
        counts = state.literal_counts()
        run.included.append(int(counts.sum()))
        run.nonempty.append(int((counts > 0).sum()))
        include = state.include_matrix()
        run.changed.append(int(include.sum() if previous is None else (include != previous).sum()))
        previous = include
    return run


def run_curves(
    cfg: TMConfig,
    settings: Sequence[str] | None = None,
    languages: Sequence[str] = LANGUAGES,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, list[CurveRun]], dict]:
    """Curves for the named settings (default: all), every fold x seed."""
    chosen = [cfg.setting(n) for n in settings] if settings else list(cfg.settings)
    data_dir = Path(cfg.data)
    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir)
    train = dataset["train"]
    if len(folds) != len(train):
        raise ValueError("folds.json does not match train.jsonl; rebuild the dataset")
    texts = np.asarray([s.text for s in train], dtype=object)
    labels = np.asarray([s.language for s in train])
    classes = sorted(set(labels))

    runs: dict[str, list[CurveRun]] = {s.name: [] for s in chosen}
    fold_ids = sorted(set(folds.tolist()))
    total, done = len(fold_ids) * len(chosen) * len(cfg.seeds), 0
    for k in fold_ids:
        tr, va = folds != k, folds == k
        # One binarizer per fold, fit on that fold's training part only; shared by all runs.
        binarizer = cfg.features.binarizer().fit(list(texts[tr]), labels[tr])
        x_tr, x_va = binarizer.transform(list(texts[tr])), binarizer.transform(list(texts[va]))
        for setting in chosen:
            for seed in cfg.seeds:
                done += 1
                if progress:
                    progress(f"[{done}/{total}] {setting.name} fold {k} seed {seed}")
                run = train_curve(setting, seed, k, x_tr, labels[tr], x_va, labels[va], classes,
                                  cfg.curves.max_epochs)  # fmt: skip
                runs[setting.name].append(run)

    try:
        from importlib.metadata import version

        tmu_version = version("tmu")
    except Exception:  # pragma: no cover - metadata missing
        tmu_version = "unknown"
    meta = {
        **dataset_meta(data_dir, dataset, folds),
        "config_hash": config_hash(cfg),
        "seeds": list(cfg.seeds),
        "tmu": tmu_version,
    }
    meta["versions"]["tmu"] = tmu_version
    return runs, meta


def summarise(runs: list[CurveRun], curves: CurveConfig) -> dict:
    """Mean curves over all fold x seed runs, the chosen epoch and clause statistics."""
    val = np.asarray([r.val_f1 for r in runs])
    train = np.asarray([r.train_f1 for r in runs])
    mean = val.mean(axis=0)
    epoch, smooth = choose_epoch(mean, curves.smooth_window, curves.plateau_tolerance,
                                 curves.epoch_rule)  # fmt: skip
    plateau, _ = choose_epoch(mean, curves.smooth_window, curves.plateau_tolerance, "plateau")
    peak = int(smooth.argmax()) + 1
    tail = min(TAIL, val.shape[1])
    # Spread across folds: average each fold's seeds first, then take the std over folds.
    by_fold: dict[int, list[np.ndarray]] = {}
    for r in runs:
        by_fold.setdefault(r.fold, []).append(np.asarray(r.val_f1))
    fold_means = np.asarray([np.mean(v, axis=0) for _, v in sorted(by_fold.items())])
    return {
        "epoch_rule": curves.epoch_rule,
        "chosen_epoch": epoch,
        "plateau_epoch": plateau,
        "smoothed_at_plateau": float(smooth[plateau - 1]),
        "smoothed_at_chosen": float(smooth[epoch - 1]),
        "smoothed_max": float(smooth.max()),
        "smoothed_max_epoch": peak,
        "tail_mean": float(val[:, -tail:].mean()),
        "tail_epochs": tail,
        "val_mean": val.mean(axis=0).tolist(),
        "val_std_runs": val.std(axis=0).tolist(),
        "val_std_folds": fold_means.std(axis=0).tolist(),
        "val_smoothed": smooth.tolist(),
        "train_mean": train.mean(axis=0).tolist(),
        "epoch_seconds_median": [
            statistics.median(v) for v in zip(*(r.epoch_seconds for r in runs), strict=True)
        ],
        "included_mean": np.mean([r.included for r in runs], axis=0).tolist(),
        "nonempty_mean": np.mean([r.nonempty for r in runs], axis=0).tolist(),
        "changed_mean": np.mean([r.changed for r in runs], axis=0).tolist(),
    }


def curves_json(cfg: TMConfig, runs: dict[str, list[CurveRun]], meta: dict) -> dict:
    """Machine-readable twin of the markdown: summaries plus every raw run (research record)."""
    settings = {}
    for name, setting_runs in runs.items():
        params = cfg.setting(name).kwargs()
        n_classes = len(meta.get("composition", {}).get("train", {})) or None
        settings[name] = {
            "params": params,
            "n_clauses_total": params["n_clauses"] * n_classes if n_classes else None,
            "summary": summarise(setting_runs, cfg.curves),
            "runs": [
                {"fold": r.fold, "seed": r.seed, "val_f1": r.val_f1, "train_f1": r.train_f1}
                for r in setting_runs
            ],
        }
    return {
        "schema": CURVES_SCHEMA,
        "meta": meta,
        "curves": config_dict(cfg.curves),
        "features": config_dict(cfg.features),
        "settings": settings,
    }


def _fmt_params(params: dict) -> str:
    keys = ("n_clauses", "T", "s", "weighted_clauses")
    return ", ".join(f"{k}={params[k]}" for k in keys if k in params)


def render_curves_md(data: dict) -> str:
    meta, curves = data["meta"], data["curves"]
    files = ", ".join(f"`{k}` {v}" for k, v in meta["dataset_files"].items()) or "n/a"
    md = [
        "# TM training curves",
        "",
        "Generated by `codelangtm tm-curve`; do not edit by hand. Re-run to refresh.",
        "",
        "## Setup",
        "",
        f"- Generated: {meta['generated']}; config hash `{meta['config_hash']}`",
        f"- Dataset: `{meta['data_dir']}` (train {meta['n_train']}); SHA-256 prefixes: {files}",
        f"- Protocol: {meta['n_folds']}-fold repo-grouped CV on train only (the test set is not "
        f"used); binarizer refit per fold; seeds {', '.join(map(str, meta['seeds']))}; "
        f"up to {curves['max_epochs']} epochs, macro-F1 on the held-out fold after every epoch",
        f"- Epoch rule: mean held-out F1 over all folds x seeds, centred moving average of "
        f"{curves['smooth_window']} epochs; " + _rule_text(curves),
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
        + f"; CPU: {meta['cpu']}",
        "",
        "## Recommendation",
        "",
        "| Setting | Parameters | Chosen epoch | Smoothed F1 there | Plateau epoch (F1) "
        f"| Smoothed max (epoch) | Mean of last {TAIL} epochs | Epoch time (s, median) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, s in data["settings"].items():
        sm = s["summary"]
        e = sm["chosen_epoch"]
        md.append(
            f"| **{name}** | {_fmt_params(s['params'])} | **{e}** | "
            f"{sm['smoothed_at_chosen']:.3f} | "
            f"{sm['plateau_epoch']} ({sm['smoothed_at_plateau']:.3f}) | "
            f"{sm['smoothed_max']:.3f} ({sm['smoothed_max_epoch']}) | {sm['tail_mean']:.3f} | "
            f"{statistics.median(sm['epoch_seconds_median']):.3f} |"
        )
    md += ["", "![Training curves](figures/tm_curves.png)", "",
           "![Clause formation](figures/tm_clause_formation.png)"]  # fmt: skip
    for name, s in data["settings"].items():
        sm = s["summary"]
        n = len(sm["val_mean"])
        md += [
            "",
            f"## {name}",
            "",
            f"{_fmt_params(s['params'])}; {len(s['runs'])} runs (folds x seeds). Held-out spread "
            "is the std across folds (each fold averaged over seeds). *Changed* counts include "
            "decisions that flipped since the previous epoch.",
            "",
            "| Epoch | Held-out F1 (mean ± std over folds) | Smoothed | Train F1 "
            "| Included literals | Non-empty clauses | Changed |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for e in [c for c in CHECKPOINTS if c <= n]:
            i = e - 1
            md.append(
                f"| {e} | {sm['val_mean'][i]:.3f} ± {sm['val_std_folds'][i]:.3f} | "
                f"{sm['val_smoothed'][i]:.3f} | {sm['train_mean'][i]:.3f} | "
                f"{sm['included_mean'][i]:,.0f} | {sm['nonempty_mean'][i]:,.0f} | "
                f"{sm['changed_mean'][i]:,.0f} |"
            )
    return "\n".join(md) + "\n"


def _rule_text(curves: dict) -> str:
    plateau = (f"plateau epoch = first within {curves['plateau_tolerance']} of the smoothed "
               "maximum")  # fmt: skip
    if curves.get("epoch_rule", "plateau") == "max":
        return f"chosen epoch = the smoothed maximum (rule `max`); {plateau}, shown for reference"
    return f"chosen epoch = {plateau} (rule `plateau`, where the curve levels off)"


def summary_table(data: dict) -> str:
    rows = [f"{'setting':<12}{'epoch':>7}{'smoothed':>10}{'max (ep)':>14}{'last-20':>9}"]
    for name, s in data["settings"].items():
        sm = s["summary"]
        rows.append(
            f"{name:<12}{sm['chosen_epoch']:>7}{sm['smoothed_at_chosen']:>10.3f}"
            f"{sm['smoothed_max']:>9.3f} ({sm['smoothed_max_epoch']:>3}){sm['tail_mean']:>9.3f}"
        )
    return "\n".join(rows)
