"""Feature ablations: repo-grouped CV of binarizer settings, compared fold-by-fold to a base.

Test data is never used here: ablations choose the feature design, so only CV folds on train
may inform them. For each setting the binarizer is fit once per fold on that fold's training
part and its output shared by all models, which is equivalent to one Pipeline per model.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score

from . import LANGUAGES
from .audit import load_dataset
from .baselines import _macro_f1, dataset_meta, load_folds, make_models
from .config import AblationConfig, FeatureConfig, Study, config_hash


@dataclass
class AblationRun:
    features: FeatureConfig
    model: str
    cv_f1: list[float]  # per fold, in fold order
    per_language_f1: dict[str, float]  # on pooled out-of-fold predictions
    vocab_size: int  # mean vocabulary size actually used (< M if too few n-grams exist)
    binarize_ms: float  # transform time per validation snippet
    fit_seconds: float  # model fitting only, summed over folds

    @property
    def cv_mean(self) -> float:
        return statistics.fmean(self.cv_f1)

    @property
    def cv_std(self) -> float:
        return statistics.pstdev(self.cv_f1)


def evaluate_features(
    features: FeatureConfig,
    texts: Sequence[str],
    labels: Sequence[str],
    folds: np.ndarray,
    models: dict[str, object],
    languages: Sequence[str] = LANGUAGES,
) -> dict[str, AblationRun]:
    x_all = np.asarray(texts, dtype=object)
    y_all = np.asarray(labels)
    scores: dict[str, list[float]] = {name: [] for name in models}
    oof: dict[str, np.ndarray] = {name: np.empty(len(y_all), dtype=y_all.dtype) for name in models}
    fit_time = dict.fromkeys(models, 0.0)
    vocab_sizes, transform_seconds = [], 0.0

    for k in sorted(set(folds.tolist())):
        train, val = folds != k, folds == k
        binarizer = features.binarizer().fit(list(x_all[train]))
        vocab_sizes.append(len(binarizer.vocabulary_))
        x_train = binarizer.transform(list(x_all[train]))
        start = time.perf_counter()
        x_val = binarizer.transform(list(x_all[val]))
        transform_seconds += time.perf_counter() - start
        for name, model in models.items():
            fitted = clone(model)
            start = time.perf_counter()
            fitted.fit(x_train, y_all[train])
            fit_time[name] += time.perf_counter() - start
            pred = fitted.predict(x_val)
            oof[name][val] = pred
            scores[name].append(_macro_f1(y_all[val], pred))

    labels_list = list(languages)
    runs = {}
    for name in models:
        per_lang = f1_score(y_all, oof[name], labels=labels_list, average=None, zero_division=0)
        runs[name] = AblationRun(
            features=features,
            model=name,
            cv_f1=scores[name],
            per_language_f1={
                lang: float(f) for lang, f in zip(labels_list, per_lang, strict=True)
            },
            vocab_size=round(statistics.fmean(vocab_sizes)),
            binarize_ms=transform_seconds / max(len(y_all), 1) * 1000,
            fit_seconds=fit_time[name],
        )
    return runs


def paired_delta(run: AblationRun, base: AblationRun) -> tuple[float, float]:
    """Mean and std of per-fold F1 differences vs base. Same folds, so fold difficulty cancels."""
    diffs = [a - b for a, b in zip(run.cv_f1, base.cv_f1, strict=True)]
    return statistics.fmean(diffs), statistics.pstdev(diffs)


Runs = dict[tuple[FeatureConfig, str], AblationRun]


def select_studies(cfg: AblationConfig, names: Sequence[str] | None) -> AblationConfig:
    """Keep only the named studies (config order); `None` or empty keeps all."""
    if not names:
        return cfg
    known = {s.name for s in cfg.studies}
    missing = set(names) - known
    if missing:
        raise ValueError(f"unknown study(s): {sorted(missing)}; available: {sorted(known)}")
    return replace(cfg, studies=tuple(s for s in cfg.studies if s.name in names))


def run_ablations(
    cfg: AblationConfig, languages: Sequence[str] = LANGUAGES, progress: object = None
) -> tuple[Runs, dict]:
    """Evaluate every unique feature setting once (base always included).

    `progress(i, n, features)`, if given, is called before each setting.
    """
    available = make_models(cfg.seed)
    unknown = set(cfg.models) - set(available)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")

    data_dir = Path(cfg.data)
    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir)
    train = dataset["train"]
    if len(folds) != len(train):
        raise ValueError("folds.json does not match train.jsonl; rebuild the dataset")
    texts = [s.text for s in train]
    labels = [s.language for s in train]
    models = {name: available[name] for name in cfg.models}

    runs: Runs = {}
    settings = cfg.feature_settings()
    for i, features in enumerate(settings):
        if callable(progress):
            progress(i, len(settings), features)
        for name, run in evaluate_features(
            features, texts, labels, folds, models, languages
        ).items():
            runs[features, name] = run

    meta = {
        **dataset_meta(data_dir, dataset, folds),
        "seed": cfg.seed,
        "config_hash": config_hash(cfg),
        "studies": [s.name for s in cfg.studies],
        "n_settings": len(settings),
    }
    return runs, meta


def describe(features: FeatureConfig) -> str:
    sizes = "+".join(str(n) for n in features.ngram_sizes)
    delims = "on" if features.use_delimiters else "off"
    return f"M={features.n_features}, n={sizes}, delimiters {delims}"


def _setting_label(features: FeatureConfig, varied: Sequence[str]) -> str:
    parts = []
    for key in varied:
        value = getattr(features, key)
        if key == "ngram_sizes":
            parts.append("n=" + "+".join(str(n) for n in value))
        elif key == "use_delimiters":
            parts.append("delimiters " + ("on" if value else "off"))
        else:
            parts.append(f"M={value}")
    return ", ".join(parts)


def _delta_cell(run: AblationRun, base: AblationRun) -> str:
    if run is base:
        return "base"
    mean, std = paired_delta(run, base)
    return f"{mean:+.3f} ± {std:.3f}"


def best_setting(runs: Runs, model: str) -> AblationRun:
    return max((r for (_, m), r in runs.items() if m == model), key=lambda r: r.cv_mean)


def study_table(
    study: Study, runs: Runs, base: FeatureConfig, models: Sequence[str]
) -> list[str]:
    head = "| Setting | " + " | ".join(
        f"{m} CV macro-F1 | Δ vs base (paired)" for m in models
    ) + " | Vocab used | Binarize (ms/snippet) |"
    md = [head, "| --- |" + " --- | --- |" * len(models) + " --- | --- |"]
    for features in study.settings:
        label = _setting_label(features, study.varied)
        if features == base:
            label = f"**{label}** (base)"
        cells = []
        for m in models:
            run = runs[features, m]
            cells += [f"{run.cv_mean:.3f} ± {run.cv_std:.3f}", _delta_cell(run, runs[base, m])]
        first = runs[features, models[0]]
        md.append(
            f"| {label} | " + " | ".join(cells)
            + f" | {first.vocab_size} | {first.binarize_ms:.3f} |"
        )
    return md


def language_table(
    study: Study, runs: Runs, base: FeatureConfig, model: str, languages: Sequence[str]
) -> list[str]:
    md = [
        "| Setting | " + " | ".join(languages) + " |",
        "| --- |" + " --- |" * len(languages),
    ]
    for features in study.settings:
        label = _setting_label(features, study.varied)
        if features == base:
            label = f"**{label}**"
        f1 = runs[features, model].per_language_f1
        md.append(f"| {label} | " + " | ".join(f"{f1[lang]:.2f}" for lang in languages) + " |")
    return md


def render_ablations_md(
    cfg: AblationConfig, runs: Runs, meta: dict, languages: Sequence[str] = LANGUAGES
) -> str:
    files = ", ".join(f"`{k}` {v}" for k, v in meta["dataset_files"].items()) or "n/a"
    primary = cfg.models[0]
    config_path = meta.get("config_path")
    md = [
        "# Feature ablations",
        "",
        "Generated by `codelangtm ablate`; do not edit by hand. Re-run to refresh.",
        "",
        "## Setup",
        "",
        f"- Generated: {meta['generated']}",
        f"- Config: {f'`{config_path}`' if config_path else 'n/a'} "
        f"(settings hash `{meta['config_hash']}`)",
        f"- Dataset: `{meta['data_dir']}` (train {meta['n_train']}); SHA-256 prefixes: {files}",
        f"- Base features: {describe(cfg.base)}",
        f"- Protocol: {meta['n_folds']}-fold repo-grouped CV on train only; binarizer refit per "
        "fold. The test set is not used: ablations select the feature design.",
        "- Δ vs base: mean ± std of per-fold macro-F1 differences on the same folds. A change "
        "whose mean is smaller than its std is within fold noise.",
        f"- Models: {', '.join(cfg.models)}; seed {cfg.seed}; "
        + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
        + f"; CPU: {meta['cpu']}",
        "- Binarize time: transform only, per validation snippet, single run (approximate).",
    ]
    for study in cfg.studies:
        md += ["", f"## {study.name}", "", f"Varied: {', '.join(study.varied)}.", ""]
        md += study_table(study, runs, cfg.base, cfg.models)
        md += ["", f"Per-language F1 ({primary}, out-of-fold):", ""]
        md += language_table(study, runs, cfg.base, primary, languages)

    md += ["", "## Best setting by CV", ""]
    for m in cfg.models:
        best = best_setting(runs, m)
        mean, std = paired_delta(best, runs[cfg.base, m])
        md.append(
            f"- {m}: {describe(best.features)} (CV {best.cv_mean:.3f}; "
            f"Δ vs base {mean:+.3f} ± {std:.3f})"
        )
    return "\n".join(md) + "\n"


def summary_table(cfg: AblationConfig, runs: Runs) -> str:
    # ASCII only for Windows consoles.
    rows = [f"{'setting':<40}" + "".join(f"{m[:20]:>22}" for m in cfg.models)]
    for features in cfg.feature_settings():
        cells = []
        for m in cfg.models:
            run = runs[features, m]
            mean, _ = paired_delta(run, runs[cfg.base, m])
            cells.append(f"{run.cv_mean:.3f} ({mean:+.3f})")
        rows.append(f"{describe(features):<40}" + "".join(f"{c:>22}" for c in cells))
    return "\n".join(rows)
