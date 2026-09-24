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
from dataclasses import dataclass, field
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
from .splits import SPLIT_SALT, check_no_leakage, stable_split

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
    repeat_f1: list[float] = field(default_factory=list)  # test F1 over extra repo-level splits

    @property
    def cv_mean(self) -> float:
        return statistics.fmean(self.cv_f1)

    @property
    def cv_std(self) -> float:
        return statistics.pstdev(self.cv_f1)

    @property
    def repeat_mean(self) -> float | None:
        return statistics.fmean(self.repeat_f1) if self.repeat_f1 else None

    @property
    def repeat_std(self) -> float | None:
        return statistics.pstdev(self.repeat_f1) if self.repeat_f1 else None


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    present = sorted(set(y_true))  # languages absent from y_true would score an undefined 0
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def _pipeline(model: object, binarizer: Binarizer) -> Pipeline:
    return Pipeline([("binarize", clone(binarizer)), ("model", clone(model))])


def repeated_split_f1(
    model: object,
    pool: Sequence,
    binarizer: Binarizer,
    repeats: int,
    test_size: float = 0.2,
) -> list[float]:
    """Test macro-F1 over `repeats` independent balanced repo-level splits of `pool`.

    Reports how much the test score depends on which repos land in test. Not used for model
    selection (that is CV's job).
    """
    scores = []
    for k in range(repeats):
        split = stable_split(pool, test_size, n_folds=2, salt=f"{SPLIT_SALT}:repeat:{k}")
        train, test = split.split(pool)
        check_no_leakage(train, test)
        pipe = _pipeline(model, binarizer)
        pipe.fit([s.text for s in train], [s.language for s in train])
        pred = pipe.predict([s.text for s in test])
        scores.append(_macro_f1(np.asarray([s.language for s in test]), pred))
    return scores


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
    repeats: int = 0,
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
        repeat_f1=repeated_split_f1(model, [*train, *test], binarizer, repeats),
    )


def run_baselines(
    data_dir: str | Path,
    models: Sequence[str] | None = None,
    n_features: int = 500,
    seed: int = 0,
    languages: Sequence[str] = LANGUAGES,
    repeats: int = 10,
    binarizer: Binarizer | None = None,
) -> tuple[list[ModelResult], dict]:
    """`binarizer` (unfitted) sets all feature options; if omitted, `Binarizer(n_features)`."""
    data_dir = Path(data_dir)
    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir)
    available = make_models(seed)
    unknown = set(models or ()) - set(available)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")
    binarizer = binarizer or Binarizer(n_features=n_features)
    results = [
        evaluate_model(name, available[name], dataset, folds, binarizer, languages, repeats)
        for name in (models or available)
    ]
    meta = {
        **dataset_meta(data_dir, dataset, folds),
        "n_features": binarizer.n_features,
        "features": {
            "n_features": binarizer.n_features,
            "ngram_sizes": list(binarizer.ngram_sizes),
            "use_delimiters": binarizer.use_delimiters,
        },
        "seed": seed,
        "repeats": repeats,
    }
    return results, meta


def dataset_meta(data_dir: Path, dataset: dict[str, list], folds: np.ndarray) -> dict:
    """Provenance shared by generated reports: dataset hashes, sizes, split, versions, CPU."""
    manifest_path = data_dir / "dataset.json"
    files, split_params = {}, {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files, split_params = manifest.get("files", {}), manifest.get("params", {})
    return {
        "generated": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "data_dir": str(data_dir),
        "dataset_files": {k: v[:12] for k, v in files.items()},
        "n_train": len(dataset["train"]),
        "n_test": len(dataset["test"]),
        "n_wild": len(dataset.get("wild", [])),
        "n_folds": len(set(folds.tolist())),
        "split": split_params,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "cpu": platform.processor() or platform.machine(),
    }


def best_by_cv(results: Sequence[ModelResult]) -> ModelResult:
    return max(results, key=lambda r: r.cv_mean)


def summary_table(results: Sequence[ModelResult]) -> str:
    # ASCII only: Windows consoles (cp1252) cannot print "±".
    rows = [
        f"{'model':<21}{'CV macro-F1':>18}{'test F1':>9}{'repeated test':>19}"
        f"{'ms/snip':>9}{'KB':>8}"
    ]
    for r in results:
        rep = f"{r.repeat_mean:.3f} +/- {r.repeat_std:.3f}" if r.repeat_f1 else "-"
        rows.append(
            f"{r.name:<21}{r.cv_mean:>9.3f} +/- {r.cv_std:.3f}{r.test_f1:>9.3f}{rep:>19}"
            f"{r.latency_ms:>9.3f}{r.size_kb:>8.0f}"
        )
    return "\n".join(rows)


def _repeat_cell(r: ModelResult) -> str:
    if not r.repeat_f1:
        return "-"
    return (
        f"{r.repeat_mean:.3f} ± {r.repeat_std:.3f} "
        f"({min(r.repeat_f1):.3f}-{max(r.repeat_f1):.3f})"
    )


def _top_confusions(r: ModelResult, languages: Sequence[str], n: int = 5) -> list[str]:
    pairs = [
        (r.confusion[i][j], languages[i], languages[j])
        for i in range(len(languages))
        for j in range(len(languages))
        if i != j and r.confusion[i][j]
    ]
    return [f"{t} → {p}: {c}" for c, t, p in sorted(pairs, reverse=True)[:n]]


def _split_text(params: dict) -> str:
    if params.get("split") == "stable-hash":
        return (
            f"stable hash-based per-language repo split (salt `{params.get('salt')}`, "
            f"test share {params.get('test_size')}), repos never shared between train and test"
        )
    return "repo-grouped (StratifiedGroupKFold)" if params else "n/a"


def _features_text(meta: dict) -> str:
    f = meta.get("features") or {"n_features": meta["n_features"], "ngram_sizes": [2, 3]}
    sizes = ", ".join(str(n) for n in f["ngram_sizes"])
    delims = "" if f.get("use_delimiters", True) else ", use_delimiters=False"
    return f"`Binarizer(n_features={f['n_features']}, ngram_sizes=({sizes}){delims})`"


def _config_text(meta: dict) -> str:
    cfg = meta.get("config")
    if not cfg:
        return "n/a"
    source = f"`{cfg['path']}`" if cfg.get("path") else "defaults"
    if cfg.get("overrides"):
        source += " + CLI overrides " + ", ".join(f"`{k}`" for k in cfg["overrides"])
    return f"{source} (settings hash `{cfg['hash']}`)"


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
        f"- Config: {_config_text(meta)}",
        f"- Dataset: `{meta['data_dir']}` (train {meta['n_train']}, test {meta['n_test']}, "
        f"wild {meta['n_wild']}); SHA-256 prefixes: {files}",
        f"- Features: {_features_text(meta)}, refit inside every fold",
        f"- Split: {_split_text(meta.get('split', {}))}",
        f"- Protocol: {meta['n_folds']}-fold repo-grouped CV on train (model selection), then "
        "refit on all of train and score test once",
        f"- Repeated test: {meta.get('repeats', 0)} extra balanced repo-level train/test splits "
        "of train + test (salts `…:repeat:k`), reported as mean ± std (min-max); not used for "
        "model selection",
        f"- Seed {meta['seed']}; " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
        + f"; CPU: {meta['cpu']}",
        "- Latency: end-to-end (binarize + predict), median of "
        f"{LATENCY_REPEATS} passes over the test set, divided by snippets",
        "",
        "## Baselines",
        "",
        "| Model | CV macro-F1 | Test macro-F1 | Repeated test macro-F1 | Test accuracy "
        "| Fit (s) | Latency (ms/snippet) | Size (KB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        name = f"**{r.name}**" if r is best else r.name
        md.append(
            f"| {name} | {r.cv_mean:.3f} ± {r.cv_std:.3f} | {r.test_f1:.3f} | "
            f"{_repeat_cell(r)} | {r.test_accuracy:.3f} | {r.fit_seconds:.2f} | "
            f"{r.latency_ms:.3f} | {r.size_kb:.0f} |"
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
