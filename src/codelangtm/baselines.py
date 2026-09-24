"""Classical baselines on binarized n-gram features, with repo-grouped CV and a held-out test.

Every model is `Pipeline([Binarizer, classifier])`, cloned fresh per fold, so the n-gram
vocabulary is always learned from training snippets only. The best model is selected by CV
macro-F1; the test set is scored once per model and never used for selection.
"""

from __future__ import annotations

import json
import pickle
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import LANGUAGES
from .audit import load_dataset
from .features import Binarizer

LATENCY_REPEATS = 5


def make_models(seed: int = 0) -> dict[str, object]:
    # BernoulliNB, not MultinomialNB: features are presence/absence flags, not counts.
    return {
        "naive_bayes": BernoulliNB(),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=5000),
        "linear_svm": LinearSVC(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }


MODEL_NAMES = tuple(make_models())


@dataclass
class ModelResult:
    name: str
    cv_f1: list[float]
    test_f1: float
    test_accuracy: float
    per_language_f1: dict[str, float]
    confusion: list[list[int]]
    fit_seconds: float
    latency_ms: float
    size_kb: float
    wild_f1: float | None = None

    @property
    def cv_mean(self) -> float:
        return statistics.fmean(self.cv_f1)

    @property
    def cv_std(self) -> float:
        return statistics.pstdev(self.cv_f1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    present = sorted(set(y_true))  # languages absent from y_true would score an undefined 0
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def _pipeline(model: object, binarizer: Binarizer) -> Pipeline:
    return Pipeline([("binarize", clone(binarizer)), ("model", clone(model))])


def load_folds(data_dir: str | Path) -> np.ndarray:
    data = json.loads((Path(data_dir) / "folds.json").read_text(encoding="utf-8"))
    return np.asarray(data["folds"])


def evaluate_model(
    name: str,
    model: object,
    dataset: dict[str, list],
    folds: np.ndarray,
    binarizer: Binarizer,
    languages: Sequence[str] = LANGUAGES,
) -> ModelResult:
    train, test = dataset["train"], dataset["test"]
    x_train = np.asarray([s.text for s in train], dtype=object)
    y_train = np.asarray([s.language for s in train])
    if len(folds) != len(train):
        raise ValueError("folds.json does not match train.jsonl; rebuild the dataset")

    cv = []
    for k in sorted(set(folds.tolist())):
        fold_pipe = _pipeline(model, binarizer)
        fold_pipe.fit(list(x_train[folds != k]), y_train[folds != k])
        pred = fold_pipe.predict(list(x_train[folds == k]))
        cv.append(_macro_f1(y_train[folds == k], pred))

    pipe = _pipeline(model, binarizer)
    start = time.perf_counter()
    pipe.fit(list(x_train), y_train)
    fit_seconds = time.perf_counter() - start

    x_test = [s.text for s in test]
    y_test = np.asarray([s.language for s in test])
    timings = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        pred = pipe.predict(x_test)
        timings.append(time.perf_counter() - start)
    latency_ms = statistics.median(timings) / max(len(x_test), 1) * 1000

    labels = list(languages)
    per_lang = f1_score(y_test, pred, labels=labels, average=None, zero_division=0)
    wild = dataset.get("wild") or []
    wild_f1 = None
    if wild:
        y_wild = np.asarray([s.language for s in wild])
        wild_f1 = _macro_f1(y_wild, pipe.predict([s.text for s in wild]))

    return ModelResult(
        name=name,
        cv_f1=cv,
        test_f1=_macro_f1(y_test, pred),
        test_accuracy=float(accuracy_score(y_test, pred)),
        per_language_f1={lang: float(f) for lang, f in zip(labels, per_lang, strict=True)},
        confusion=confusion_matrix(y_test, pred, labels=labels).tolist(),
        fit_seconds=fit_seconds,
        latency_ms=latency_ms,
        size_kb=len(pickle.dumps(pipe)) / 1024,
        wild_f1=wild_f1,
    )


def run_baselines(
    data_dir: str | Path,
    models: Sequence[str] | None = None,
    n_features: int = 500,
    seed: int = 0,
    languages: Sequence[str] = LANGUAGES,
) -> tuple[list[ModelResult], dict]:
    data_dir = Path(data_dir)
    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir)
    available = make_models(seed)
    unknown = set(models or ()) - set(available)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")
    binarizer = Binarizer(n_features=n_features)
    results = [
        evaluate_model(name, available[name], dataset, folds, binarizer, languages)
        for name in (models or available)
    ]

    manifest_path = data_dir / "dataset.json"
    files = {}
    if manifest_path.exists():
        files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    meta = {
        "generated": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "data_dir": str(data_dir),
        "dataset_files": {k: v[:12] for k, v in files.items()},
        "n_train": len(dataset["train"]),
        "n_test": len(dataset["test"]),
        "n_wild": len(dataset.get("wild", [])),
        "n_folds": len(set(folds.tolist())),
        "n_features": n_features,
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "cpu": platform.processor() or platform.machine(),
    }
    return results, meta


def best_by_cv(results: Sequence[ModelResult]) -> ModelResult:
    return max(results, key=lambda r: r.cv_mean)


def summary_table(results: Sequence[ModelResult]) -> str:
    # ASCII only: Windows consoles (cp1252) cannot print "±".
    rows = [f"{'model':<21}{'CV macro-F1':>18}{'test F1':>9}{'acc':>7}{'ms/snip':>9}{'KB':>8}"]
    for r in results:
        rows.append(
            f"{r.name:<21}{r.cv_mean:>9.3f} +/- {r.cv_std:.3f}{r.test_f1:>9.3f}"
            f"{r.test_accuracy:>7.3f}{r.latency_ms:>9.3f}{r.size_kb:>8.0f}"
        )
    return "\n".join(rows)


def _top_confusions(r: ModelResult, languages: Sequence[str], n: int = 5) -> list[str]:
    pairs = [
        (r.confusion[i][j], languages[i], languages[j])
        for i in range(len(languages))
        for j in range(len(languages))
        if i != j and r.confusion[i][j]
    ]
    return [f"{t} → {p}: {c}" for c, t, p in sorted(pairs, reverse=True)[:n]]


def render_results_md(
    results: Sequence[ModelResult], meta: dict, languages: Sequence[str] = LANGUAGES
) -> str:
    best = best_by_cv(results)
    files = ", ".join(f"`{k}` {v}" for k, v in meta["dataset_files"].items()) or "n/a"
    md = [
        "# Results",
        "",
        "Generated by `codelangtm baselines`; do not edit by hand. Re-run to refresh.",
        "",
        "## Setup",
        "",
        f"- Generated: {meta['generated']}",
        f"- Dataset: `{meta['data_dir']}` (train {meta['n_train']}, test {meta['n_test']}, "
        f"wild {meta['n_wild']}); SHA-256 prefixes: {files}",
        f"- Features: `Binarizer(n_features={meta['n_features']}, ngram_sizes=(2, 3))`, "
        "refit inside every fold",
        f"- Protocol: {meta['n_folds']}-fold repo-grouped CV on train (model selection), then "
        "refit on all of train and score test once",
        f"- Seed {meta['seed']}; " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
        + f"; CPU: {meta['cpu']}",
        "- Latency: end-to-end (binarize + predict), median of "
        f"{LATENCY_REPEATS} passes over the test set, divided by snippets",
        "",
        "## Baselines",
        "",
        "| Model | CV macro-F1 | Test macro-F1 | Test accuracy | Fit (s) | Latency (ms/snippet) "
        "| Size (KB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        name = f"**{r.name}**" if r is best else r.name
        md.append(
            f"| {name} | {r.cv_mean:.3f} ± {r.cv_std:.3f} | {r.test_f1:.3f} | "
            f"{r.test_accuracy:.3f} | {r.fit_seconds:.2f} | {r.latency_ms:.3f} | {r.size_kb:.0f} |"
        )
    md += [
        "",
        f"Best by CV: **{best.name}** (CV {best.cv_mean:.3f}, test {best.test_f1:.3f}).",
        "",
        "## Test F1 per language",
        "",
        "| Model | " + " | ".join(languages) + " |",
        "| --- |" + " --- |" * len(languages),
    ]
    for r in results:
        md.append(
            f"| {r.name} | " + " | ".join(f"{r.per_language_f1[lang]:.2f}" for lang in languages)
            + " |"
        )
    md += [
        "",
        f"## Confusion matrix: {best.name} (test)",
        "",
        "Rows are true languages, columns predicted.",
        "",
        "| true \\ pred | " + " | ".join(languages) + " |",
        "| --- |" + " --- |" * len(languages),
    ]
    for lang, row in zip(languages, best.confusion, strict=True):
        md.append(f"| {lang} | " + " | ".join(str(v) for v in row) + " |")
    confusions = _top_confusions(best, languages)
    md += ["", "Most frequent confusions: " + ("; ".join(confusions) if confusions else "none")]
    return "\n".join(md) + "\n"
