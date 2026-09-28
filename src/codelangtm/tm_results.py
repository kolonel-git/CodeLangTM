"""Full evaluation of the TM beside the baselines (M3 step B3), research-ready.

Every TM setting runs the exact protocol of the baselines (`baselines.evaluate_model`): 5-fold
repo-grouped CV on train, one fit on all of train scored once on test, and the repeated
repo-level splits; the TM does this once per seed. Baselines named in the config run once
(they are deterministic). The TM is compared with each baseline by a paired, corrected
resampled t-test (Nadeau & Bengio 2003) on the same CV folds and on the same repeated splits,
using per-split TM scores averaged over seeds.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline

from . import LANGUAGES
from .audit import load_dataset
from .baselines import (
    SPLIT_SALT,
    ModelResult,
    _macro_f1,
    dataset_meta,
    evaluate_model,
    load_folds,
    make_models,
    result_dict,
)
from .config import TMConfig, config_dict, config_hash
from .model import TMLanguageClassifier, model_dict, save_pipeline
from .splits import stable_split

TM_RESULTS_SCHEMA = "codelangtm.tm-results/1"
TEST_SIZE = 0.2  # repeated splits use the same share as `repeated_split_f1`
LATENCY_REPEATS = 5


# ------------------------------------------------------------------ statistics


def corrected_ttest(
    diffs: Sequence[float], test_train_ratio: float, confidence: float = 0.95
) -> dict:
    """Nadeau & Bengio's corrected resampled t-test on paired per-split differences.

    Repeated splits share most of their training data, so the plain variance of the differences
    understates the uncertainty. The correction replaces 1/J with 1/J + n_test/n_train.
    Returns mean, confidence interval, t, two-sided p and the number of splits.
    """
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    if j < 2:
        raise ValueError("need at least 2 paired splits")
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    se = math.sqrt((1 / j + test_train_ratio) * var)
    df = j - 1
    crit = float(stats.t.ppf(0.5 + confidence / 2, df))
    if se == 0:
        t_stat, p = (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    else:
        t_stat = mean / se
        p = float(2 * stats.t.sf(abs(t_stat), df))
    return {
        "mean": mean,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "t": t_stat,
        "p": p,
        "df": df,
        "n_splits": j,
        "test_train_ratio": test_train_ratio,
    }


def repeated_test_train_ratio(pool: Sequence, repeats: int) -> float:
    """Mean n_test / n_train over the repeated splits (same salts as `repeated_split_f1`)."""
    ratios = []
    for k in range(repeats):
        split = stable_split(pool, TEST_SIZE, n_folds=2, salt=f"{SPLIT_SALT}:repeat:{k}")
        train, test = split.split(pool)
        ratios.append(len(test) / len(train))
    return statistics.fmean(ratios)


# ------------------------------------------------------------------ TM runs


@dataclass
class TMSeedRun:
    seed: int
    result: ModelResult
    n_clauses_total: int
    included_literals: int
    empty_clauses: int
    model_json_kb: float
    tmu_latency_ms: float
    model_path: str | None = None


@dataclass
class TMEntry:
    setting: str
    params: dict
    runs: list[TMSeedRun] = field(default_factory=list)

    def seed_values(self, attr: str) -> list[float]:
        return [getattr(r.result, attr) for r in self.runs]

    @property
    def cv_by_fold(self) -> np.ndarray:
        """Per-fold CV F1 averaged over seeds."""
        return np.mean([r.result.cv_f1 for r in self.runs], axis=0)

    @property
    def repeat_by_split(self) -> np.ndarray:
        """Per-split repeated-test F1 averaged over seeds."""
        return np.mean([r.result.repeat_f1 for r in self.runs], axis=0)


def _tmu_latency_ms(pipe: object, texts: Sequence[str]) -> float:
    """Classifier-only TMU prediction time per snippet (binarization excluded), median of runs."""
    x = pipe.named_steps["binarize"].transform(texts)
    model = pipe.named_steps["model"]
    timings = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        model.predict_tmu(x)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) / max(len(texts), 1) * 1000


def run_tm_results(
    cfg: TMConfig,
    settings: Sequence[str] | None = None,
    languages: Sequence[str] = LANGUAGES,
    models_dir: str | Path | None = "models",
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ModelResult], list[TMEntry], dict]:
    """Baselines once, then every TM setting x seed through the identical protocol."""
    say = progress or (lambda _m: None)
    chosen = [cfg.setting(n) for n in settings] if settings else list(cfg.settings)
    data_dir = Path(cfg.data)
    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir)
    binarizer = cfg.features.binarizer()
    meta = {
        **dataset_meta(data_dir, dataset, folds),
        "config_hash": config_hash(cfg),
        "seeds": list(cfg.seeds),
        "repeats": cfg.repeats,
        "features": config_dict(cfg.features),
    }
    try:
        from importlib.metadata import version

        meta["versions"]["tmu"] = version("tmu")
    except Exception:  # pragma: no cover - metadata missing
        meta["versions"]["tmu"] = "unknown"

    available = make_models(0)
    unknown = set(cfg.baselines) - set(available)
    if unknown:
        raise ValueError(f"unknown baseline(s): {sorted(unknown)}")
    baselines = []
    for name in cfg.baselines:
        say(f"baseline {name}")
        baselines.append(evaluate_model(name, available[name], dataset, folds, binarizer,
                                        languages, cfg.repeats))  # fmt: skip

    test_texts = [s.text for s in dataset["test"]]
    entries = []
    for setting in chosen:
        entry = TMEntry(setting.name, setting.kwargs())
        for seed in cfg.seeds:
            say(f"{setting.name} seed {seed}")
            model = TMLanguageClassifier(**setting.kwargs(), seed=seed)
            result = evaluate_model(setting.name, model, dataset, folds, binarizer, languages,
                                    cfg.repeats, keep_pipeline=True)  # fmt: skip
            pipe = result.pipeline
            state = pipe.named_steps["model"].state_
            model_meta = {
                "setting": setting.name, "seed": seed, "config_hash": meta["config_hash"],
                "dataset_files": meta["dataset_files"], "trained_on": "train (all folds)",
            }  # fmt: skip
            path = None
            if models_dir is not None:
                path = save_pipeline(Path(models_dir) / f"{setting.name}_seed{seed}.json", pipe,
                                     model_meta).as_posix()  # fmt: skip
            text = json.dumps(model_dict(pipe.named_steps["binarize"], state, model_meta),
                              ensure_ascii=False, separators=(",", ":"))  # fmt: skip
            json_kb = len(text.encode("utf-8")) / 1024
            counts = state.literal_counts()
            entry.runs.append(TMSeedRun(
                seed=seed, result=result, n_clauses_total=state.n_clauses,
                included_literals=int(counts.sum()), empty_clauses=int((counts == 0).sum()),
                model_json_kb=json_kb, tmu_latency_ms=_tmu_latency_ms(pipe, test_texts),
                model_path=path,
            ))  # fmt: skip
            result.pipeline = None  # free the TMU object
        entries.append(entry)

    pool = [*dataset["train"], *dataset["test"]]
    meta["repeated_test_train_ratio"] = repeated_test_train_ratio(pool, cfg.repeats)
    meta["cv_test_train_ratio"] = 1 / (meta["n_folds"] - 1)
    return baselines, entries, meta


def significance(baselines: Sequence[ModelResult], entries: Sequence[TMEntry], meta: dict) -> list:
    """TM (seed-averaged per split) minus each baseline, on CV folds and on repeated splits."""
    rows = []
    for entry in entries:
        for base in baselines:
            cv_diff = entry.cv_by_fold - np.asarray(base.cv_f1)
            rows.append({"tm": entry.setting, "baseline": base.name, "evaluation": "CV folds",
                         **corrected_ttest(cv_diff, meta["cv_test_train_ratio"])})  # fmt: skip
            if base.repeat_f1 and entry.runs and entry.runs[0].result.repeat_f1:
                rep_diff = entry.repeat_by_split - np.asarray(base.repeat_f1)
                test = corrected_ttest(rep_diff, meta["repeated_test_train_ratio"])
                rows.append({"tm": entry.setting, "baseline": base.name,
                             "evaluation": "repeated test splits", **test})  # fmt: skip
    return rows


# ------------------------------------------------------------------ final models


CANONICAL_RULE = "median CV macro-F1 over seeds (no test data used)"


def canonical_seed(data: dict, setting: str) -> int:
    """The seed whose CV macro-F1 is the median of the setting's seeds (lower median if even,
    lower seed on ties): a typical model, chosen without looking at the test set."""
    seeds = data["tm"][setting]["seeds"]
    ranked = sorted(seeds, key=lambda s: (s["cv_mean"], s["seed"]))
    return ranked[(len(ranked) - 1) // 2]["seed"]


def select_canonical(data: dict, setting: str, models_dir: str | Path, out: str | Path) -> dict:
    """Copy `<models_dir>/<setting>_seed<k>.json` for the canonical seed to `out`, recording the
    rule and every seed's CV score in the model's metadata. Returns that metadata."""
    seed = canonical_seed(data, setting)
    src = Path(models_dir) / f"{setting}_seed{seed}.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found: rerun `codelangtm tm-results` to save models")
    model = json.loads(src.read_text(encoding="utf-8"))
    model["meta"]["canonical"] = {
        "rule": CANONICAL_RULE,
        "setting": setting,
        "seed": seed,
        "cv_by_seed": {str(s["seed"]): s["cv_mean"] for s in data["tm"][setting]["seeds"]},
        "source": src.as_posix(),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")) + "\n",
                   encoding="utf-8")  # fmt: skip
    return model["meta"]


def train_final(cfg: TMConfig, setting: str, seed: int, out: str | Path) -> dict:
    """Train one TM on all of train (no CV, no repeated splits) and save it as a model file.
    Returns a summary; the test score is printed for information only, not used for choices."""
    s = cfg.setting(setting)
    data_dir = Path(cfg.data)
    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir)
    meta = dataset_meta(data_dir, dataset, folds)
    train, test = dataset["train"], dataset["test"]
    binarizer = cfg.features.binarizer()
    model = TMLanguageClassifier(**s.kwargs(), seed=seed)
    start = time.perf_counter()
    x_train = binarizer.fit([t.text for t in train], [t.language for t in train]).transform(
        [t.text for t in train]
    )
    model.fit(x_train, [t.language for t in train])
    seconds = time.perf_counter() - start
    pipe = Pipeline([("binarize", binarizer), ("model", model)])
    model_meta = {
        "setting": setting, "seed": seed, "config_hash": config_hash(cfg),
        "dataset_files": meta["dataset_files"], "trained_on": "train (all folds)",
    }  # fmt: skip
    path = save_pipeline(out, pipe, model_meta)
    y_test = np.asarray([t.language for t in test])
    return {
        "path": path.as_posix(),
        "fit_seconds": seconds,
        "train_f1": _macro_f1(np.asarray([t.language for t in train]), model.predict(x_train)),
        "test_f1": _macro_f1(y_test, pipe.predict([t.text for t in test])),
        "epochs": model.epochs_trained_,
    }


# ------------------------------------------------------------------ reports


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def aggregate(entry: TMEntry, languages: Sequence[str]) -> dict:
    """Seed-level summary of one TM setting, comparable to a baseline row."""
    cv_seed = [r.result.cv_mean for r in entry.runs]
    test_seed = [r.result.test_f1 for r in entry.runs]
    rep_seed = [r.result.repeat_mean for r in entry.runs if r.result.repeat_f1]
    fold_means = entry.cv_by_fold
    out = {
        "cv_mean": statistics.fmean(cv_seed),
        "cv_std_folds": float(np.std(fold_means)),  # like the baselines' "± std over folds"
        "cv_std_seeds": _mean_std(cv_seed)[1],
        "test_mean": statistics.fmean(test_seed),
        "test_std_seeds": _mean_std(test_seed)[1],
        "test_min": min(test_seed),
        "test_max": max(test_seed),
        "test_accuracy_mean": statistics.fmean(r.result.test_accuracy for r in entry.runs),
        "per_language_f1": {
            lang: statistics.fmean(r.result.per_language_f1[lang] for r in entry.runs)
            for lang in languages
        },
        "confusion_sum": np.sum([r.result.confusion for r in entry.runs], axis=0).tolist(),
        "fit_seconds": statistics.fmean(r.result.fit_seconds for r in entry.runs),
        "fit_cpu_seconds": statistics.fmean(r.result.fit_cpu_seconds or 0 for r in entry.runs),
        "latency_ms": statistics.fmean(r.result.latency_ms for r in entry.runs),
        "tmu_latency_ms": statistics.fmean(r.tmu_latency_ms for r in entry.runs),
        "size_kb": statistics.fmean(r.result.size_kb for r in entry.runs),
        "model_json_kb": statistics.fmean(r.model_json_kb for r in entry.runs),
        "n_clauses_total": entry.runs[0].n_clauses_total if entry.runs else 0,
        "included_literals": statistics.fmean(r.included_literals for r in entry.runs),
        "empty_clauses": statistics.fmean(r.empty_clauses for r in entry.runs),
        "peak_model_mb": statistics.fmean(r.result.peak_model_mb or 0 for r in entry.runs),
    }
    if rep_seed:
        by_split = entry.repeat_by_split
        out.update(repeat_mean=float(by_split.mean()), repeat_std_splits=float(by_split.std()),
                   repeat_min=float(by_split.min()), repeat_max=float(by_split.max()),
                   repeat_std_seeds=_mean_std(rep_seed)[1])  # fmt: skip
    return out


def tm_results_json(cfg: TMConfig, baselines, entries, meta, languages=LANGUAGES) -> dict:
    tm = {}
    for e in entries:
        tm[e.setting] = {
            "params": e.params,
            "aggregate": aggregate(e, languages),
            "seeds": [
                {**result_dict(r.result), "seed": r.seed, "n_clauses_total": r.n_clauses_total,
                 "included_literals": r.included_literals, "empty_clauses": r.empty_clauses,
                 "model_json_kb": r.model_json_kb, "tmu_latency_ms": r.tmu_latency_ms,
                 "model_path": r.model_path}  # fmt: skip
                for r in e.runs
            ],
        }
    return {
        "schema": TM_RESULTS_SCHEMA,
        "meta": meta,
        "languages": list(languages),
        "baselines": [result_dict(b) for b in baselines],
        "tm": tm,
        "significance": significance(baselines, entries, meta),
    }


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def render_tm_results_md(data: dict) -> str:
    meta, langs = data["meta"], data["languages"]
    files = ", ".join(f"`{k}` {v}" for k, v in meta["dataset_files"].items()) or "n/a"
    seeds = ", ".join(map(str, meta["seeds"]))
    md = [
        "# TM results",
        "",
        "Generated by `codelangtm tm-results`; do not edit by hand. Re-run to refresh.",
        "",
        "## Setup",
        "",
        f"- Generated: {meta['generated']}; config `{meta.get('config_path', 'n/a')}` "
        f"(settings hash `{meta['config_hash']}`)",
        f"- Dataset: `{meta['data_dir']}` (train {meta['n_train']}, test {meta['n_test']}); "
        f"SHA-256 prefixes: {files}",
        "- Protocol (identical code for every model, `baselines.evaluate_model`): "
        f"{meta['n_folds']}-fold repo-grouped CV on train; one fit on all of train scored once on "
        f"test; {meta['repeats']} repeated balanced repo-level splits of train + test. "
        "Binarizer refit inside every fit.",
        f"- TM: every setting runs once per seed ({seeds}); TM rows are means over seeds. Epoch "
        "counts come from the training curves ([tm-curves.md](tm-curves.md)), chosen on CV only.",
        "- Significance: TM minus baseline per fold / per split (TM averaged over seeds), "
        "corrected resampled t-test (Nadeau & Bengio 2003), two-sided, 95% interval; test/train "
        f"ratio {meta['cv_test_train_ratio']:.3f} for CV folds, "
        f"{meta['repeated_test_train_ratio']:.3f} for the repeated splits.",
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
        + f"; CPU: {meta['cpu']}",
        "",
        "## Accuracy",
        "",
        "Baselines: mean ± std over folds (CV) and over splits (repeated test), as in "
        "[results.md](results.md). TM: the same, averaged over seeds, plus the spread over seeds.",
        "",
        "| Model | CV macro-F1 | Test macro-F1 | Repeated test macro-F1 | Test accuracy "
        "| Seed spread (std: CV / test / repeated) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for b in data["baselines"]:
        rep = (f"{b['repeat_mean']:.3f} ± {b['repeat_std']:.3f}" if b["repeat_f1"] else "-")
        md.append(f"| {b['name']} | {b['cv_mean']:.3f} ± {b['cv_std']:.3f} | {b['test_f1']:.3f} "
                  f"| {rep} | {b['test_accuracy']:.3f} | - |")  # fmt: skip
    for name, t in data["tm"].items():
        a = t["aggregate"]
        rep = (f"{a['repeat_mean']:.3f} ± {a['repeat_std_splits']:.3f}" if "repeat_mean" in a
               else "-")  # fmt: skip
        spread = (f"{a['cv_std_seeds']:.3f} / {a['test_std_seeds']:.3f} / "
                  f"{a.get('repeat_std_seeds', 0):.3f}")  # fmt: skip
        md.append(
            f"| **{name}** ({t['params']['n_clauses']} clauses, {t['params']['epochs']} epochs) "
            f"| {a['cv_mean']:.3f} ± {a['cv_std_folds']:.3f} | {a['test_mean']:.3f} "
            f"({a['test_min']:.3f}-{a['test_max']:.3f}) | {rep} | {a['test_accuracy_mean']:.3f} "
            f"| {spread} |"
        )
    md += [
        "",
        "![TM vs baselines](figures/tm_comparison.png)",
        "",
        "## Significance: TM minus baseline",
        "",
        "Positive Δ means the TM scores higher. p < 0.05 with an interval that excludes 0 means "
        "the difference is unlikely to be split-to-split chance.",
        "",
        "| TM | Baseline | Evaluation | Mean Δ macro-F1 | 95% interval | t | p | Splits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in data["significance"]:
        md.append(f"| {s['tm']} | {s['baseline']} | {s['evaluation']} | {s['mean']:+.3f} | "
                  f"{s['ci_low']:+.3f} to {s['ci_high']:+.3f} | {s['t']:.2f} | {_p(s['p'])} | "
                  f"{s['n_splits']} |")  # fmt: skip
    md += [
        "",
        "## Test F1 per language",
        "",
        "| Model | " + " | ".join(langs) + " |",
        "| --- |" + " --- |" * len(langs),
    ]
    for b in data["baselines"]:
        md.append(f"| {b['name']} | "
                  + " | ".join(f"{b['per_language_f1'][g]:.2f}" for g in langs) + " |")  # fmt: skip
    for name, t in data["tm"].items():
        f1 = t["aggregate"]["per_language_f1"]
        md.append(f"| {name} (mean of seeds) | " + " | ".join(f"{f1[g]:.2f}" for g in langs)
                  + " |")  # fmt: skip
    md += ["", "![Per-language test F1](figures/tm_per_language.png)"]
    for name, t in data["tm"].items():
        conf = t["aggregate"]["confusion_sum"]
        n_seeds = len(t["seeds"])
        md += [
            "",
            f"## Confusion matrix: {name} (test, summed over {n_seeds} seeds)",
            "",
            "Rows are true languages, columns predicted; each seed classifies the "
            f"{meta['n_test']} test snippets once.",
            "",
            "| true \\ pred | " + " | ".join(langs) + " |",
            "| --- |" + " --- |" * len(langs),
        ]
        md += [f"| {g} | " + " | ".join(str(v) for v in row) + " |"
               for g, row in zip(langs, conf, strict=True)]  # fmt: skip
    md += [
        "",
        "## Model size and speed",
        "",
        "Latency is end to end (binarize + predict with the NumPy `TMState`), median of passes "
        "over the test set; TMU latency is TMU's own `predict` on already-binarized input. Model "
        "JSON is the one-file model format (`codelangtm.tm/1`, compact). Timings on this machine "
        "vary with background load; B5 measures resources in isolated processes.",
        "",
        "| TM | Clauses | Included literals (per clause) | Empty clauses | Model JSON (KB) "
        "| Pickled (KB) | Fit wall / CPU (s) | Latency (ms/snippet) | TMU predict (ms/snippet) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, t in data["tm"].items():
        a = t["aggregate"]
        per = a["included_literals"] / max(a["n_clauses_total"] - a["empty_clauses"], 1)
        md.append(f"| {name} | {a['n_clauses_total']:,} | {a['included_literals']:,.0f} "
                  f"({per:.1f}) | {a['empty_clauses']:.0f} | {a['model_json_kb']:.0f} | "
                  f"{a['size_kb']:.0f} | {a['fit_seconds']:.1f} / {a['fit_cpu_seconds']:.1f} | "
                  f"{a['latency_ms']:.3f} | {a['tmu_latency_ms']:.3f} |")  # fmt: skip
    base_rows = [f"{b['name']} {b['resources']['latency_ms']:.3f} ms, "
                 f"{b['resources']['size_kb']:.0f} KB" for b in data["baselines"]]  # fmt: skip
    md += ["", "Baselines in the same run: " + "; ".join(base_rows) + "."]
    paths = [s["model_path"] for t in data["tm"].values() for s in t["seeds"] if s["model_path"]]
    if paths:
        md += ["", f"Final models (one per seed, trained on all of train) saved locally, not "
               f"committed: {', '.join(f'`{p}`' for p in paths)}."]  # fmt: skip
    return "\n".join(md) + "\n"


def summary_table(data: dict) -> str:
    rows = [f"{'model':<22}{'CV':>8}{'test':>8}{'repeated':>10}"]
    for b in data["baselines"]:
        rows.append(f"{b['name']:<22}{b['cv_mean']:>8.3f}{b['test_f1']:>8.3f}"
                    f"{b['repeat_mean'] or 0:>10.3f}")  # fmt: skip
    for name, t in data["tm"].items():
        a = t["aggregate"]
        rows.append(f"{name:<22}{a['cv_mean']:>8.3f}{a['test_mean']:>8.3f}"
                    f"{a.get('repeat_mean', 0):>10.3f}")  # fmt: skip
    for s in data["significance"]:
        rows.append(f"  {s['tm']} - {s['baseline']} ({s['evaluation']}): {s['mean']:+.3f} "
                    f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] p={_p(s['p'])}")  # fmt: skip
    return "\n".join(rows)
