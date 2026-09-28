"""Process-level resource measurement (M3 step B5), the same protocol for every model.

Each measurement runs in a fresh Python process that does one job and reports JSON:
- `fit`: load the data, then train the pipeline (binarizer + classifier) on all of train;
  records wall and CPU time, the process memory before training and its peak during training.
- `predict`: load a saved model, then classify every test snippet one at a time, timing feature
  extraction (binarize) and classification (predict) separately; also a batch pass.
Process memory is the operating system's view (peak working set on Windows, max RSS elsewhere),
so memory allocated inside C extensions such as TMU is included, unlike `tracemalloc`.
Each job is repeated and the median kept, because timings on a shared machine swing with load.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

RESOURCES_SCHEMA = "codelangtm.resources/1"


# ------------------------------------------------------------------ process memory


def memory_mb() -> tuple[float, float]:
    """(current, peak) resident memory of this process in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                               wintypes.DWORD]  # fmt: skip
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                          counters.cb):  # fmt: skip
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 1024  # bytes vs KB
    current_mb = peak_mb
    status = Path("/proc/self/status")
    if status.exists():
        # Linux: read current (VmRSS) and peak (VmHWM) from the same snapshot. ru_maxrss is
        # updated lazily by the kernel and can lag behind VmRSS (issues-and-fixes W7).
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_mb = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak_mb = int(line.split()[1]) / 1024
    return current_mb, max(peak_mb, current_mb)


# ------------------------------------------------------------------ worker jobs (child process)


def _pctl(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _make_pipeline(spec: str, config: Path):
    """spec: `baseline:<name>` or `tm:<setting>:<seed>`."""
    from sklearn.pipeline import Pipeline

    from .baselines import make_models
    from .config import load_tm_config
    from .model import TMLanguageClassifier

    cfg = load_tm_config(config)
    kind, _, rest = spec.partition(":")
    if kind == "baseline":
        model = make_models(0)[rest]
    elif kind == "tm":
        setting, _, seed = rest.partition(":")
        model = TMLanguageClassifier(**cfg.setting(setting).kwargs(), seed=int(seed))
    else:
        raise ValueError(f"unknown model spec {spec!r}")
    return cfg, Pipeline([("binarize", cfg.features.binarizer()), ("model", model)])


def job_fit(spec: str, config: Path) -> dict:
    from .audit import load_dataset

    cfg, pipe = _make_pipeline(spec, config)
    train = load_dataset(cfg.data)["train"]
    texts, labels = [s.text for s in train], [s.language for s in train]
    if spec.startswith("tm:"):  # import TMU before the "before" reading, like the baselines' libs
        from .model import _tm_classifier_class

        _tm_classifier_class()
    before, _ = memory_mb()
    wall, cpu = time.perf_counter(), time.process_time()
    pipe.fit(texts, labels)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    _, peak = memory_mb()
    return {"wall_s": wall, "cpu_s": cpu, "rss_before_mb": before, "peak_rss_mb": peak,
            "added_mb": peak - before}  # fmt: skip


def job_predict(model_file: Path, data: Path, passes: int) -> dict:
    start = time.perf_counter()
    if model_file.suffix == ".json":
        from .model import load_model

        loaded = load_model(model_file)
        binarizer, predict = loaded.binarizer, loaded.state.predict
    else:
        with model_file.open("rb") as fh:
            pipe = pickle.load(fh)
        binarizer, predict = pipe.named_steps["binarize"], pipe.named_steps["model"].predict
    load_s = time.perf_counter() - start
    from .audit import load_dataset

    texts = [s.text for s in load_dataset(data)["test"]]
    before, _ = memory_mb()
    predict(binarizer.transform(texts[:1]))  # warm-up: builds lookup tables and caches
    binarize, classify = [], []
    for _ in range(passes):
        for text in texts:
            t0 = time.perf_counter()
            x = binarizer.transform([text])
            t1 = time.perf_counter()
            predict(x)
            t2 = time.perf_counter()
            binarize.append((t1 - t0) * 1000)
            classify.append((t2 - t1) * 1000)
    t0 = time.perf_counter()
    predict(binarizer.transform(texts))
    batch_s = time.perf_counter() - t0
    _, peak = memory_mb()
    total = [a + b for a, b in zip(binarize, classify, strict=True)]
    return {
        "load_s": load_s, "rss_before_mb": before, "peak_rss_mb": peak,
        "binarize_ms_median": statistics.median(binarize), "binarize_ms_p95": _pctl(binarize, 95),
        "predict_ms_median": statistics.median(classify), "predict_ms_p95": _pctl(classify, 95),
        "total_ms_median": statistics.median(total), "total_ms_p95": _pctl(total, 95),
        "batch_ms_per_snippet": batch_s / len(texts) * 1000,
        "batch_snippets_per_s": len(texts) / batch_s, "n_snippets": len(texts),
    }  # fmt: skip


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m codelangtm.resources")
    parser.add_argument("job", choices=("fit", "predict"))
    parser.add_argument("--spec")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args(argv)
    if args.job == "fit":
        out = job_fit(args.spec, args.config)
    else:
        out = job_predict(args.model_file, args.data, args.passes)
    print("RESULT " + json.dumps(out))
    return 0


# ------------------------------------------------------------------ parent side


def run_job(args: Sequence[str]) -> dict:
    """Run one job in a fresh interpreter and parse its RESULT line."""
    proc = subprocess.run([sys.executable, "-m", "codelangtm.resources", *args],
                          capture_output=True, text=True)  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(f"resource job failed ({' '.join(args)}):\n{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"resource job printed no result ({' '.join(args)})")


def _median_runs(runs: list[dict]) -> dict:
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


def model_specs(cfg, tm_results: dict, models_dir: Path) -> list[dict]:
    """Baselines from the config + each TM setting's canonical (median-CV) seed from B3."""
    from .tm_results import canonical_seed

    specs = [{"name": b, "spec": f"baseline:{b}", "model_file": None} for b in cfg.baselines]
    for s in cfg.settings:
        if s.name not in tm_results["tm"]:
            continue
        seed = canonical_seed(tm_results, s.name)
        path = Path(models_dir) / f"{s.name}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: rerun `codelangtm tm-results`")
        specs.append({"name": f"{s.name} (seed {seed})", "spec": f"tm:{s.name}:{seed}",
                      "model_file": path})  # fmt: skip
    return specs


def measure_all(
    config: Path,
    tm_results: dict,
    models_dir: Path,
    repeats: int = 3,
    passes: int = 3,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    from .audit import load_dataset
    from .baselines import dataset_meta, load_folds
    from .config import config_hash, load_tm_config

    say = progress or (lambda _m: None)
    cfg = load_tm_config(config)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in model_specs(cfg, tm_results, models_dir):
            fits = []
            for k in range(repeats):
                say(f"{spec['name']}: fit {k + 1}/{repeats}")
                fits.append(run_job(["fit", "--spec", spec["spec"], "--config", str(config)]))
            model_file = spec["model_file"]
            if model_file is None:  # baselines: fit here once and pickle for the predict job
                _, pipe = _make_pipeline(spec["spec"], config)
                train = load_dataset(cfg.data)["train"]
                pipe.fit([s.text for s in train], [s.language for s in train])
                model_file = Path(tmp) / f"{spec['name']}.pkl"
                model_file.write_bytes(pickle.dumps(pipe))
            preds = []
            for k in range(repeats):
                say(f"{spec['name']}: predict {k + 1}/{repeats}")
                job = ["predict", "--model-file", str(model_file), "--data", str(cfg.data),
                       "--passes", str(passes)]  # fmt: skip
                preds.append(run_job(job))
            rows.append({
                "name": spec["name"], "spec": spec["spec"],
                "model_kind": "TM model file (JSON)" if model_file.suffix == ".json"
                else "pickled pipeline",
                "model_kb": model_file.stat().st_size / 1024,
                "fit": _median_runs(fits), "predict": _median_runs(preds),
                "fit_runs": fits, "predict_runs": preds,
            })  # fmt: skip
    dataset = load_dataset(cfg.data)
    meta = {
        **dataset_meta(Path(cfg.data), dataset, load_folds(cfg.data)),
        "config_hash": config_hash(cfg),
        "repeats": repeats,
        "passes": passes,
        "platform": sys.platform,
        "memory_source": "peak working set (Windows)" if sys.platform == "win32"
        else "max RSS (getrusage)",
    }  # fmt: skip
    return rows, meta


def resources_json(rows: list[dict], meta: dict) -> dict:
    return {"schema": RESOURCES_SCHEMA, "meta": meta, "models": rows}


def render_resources_md(data: dict) -> str:
    meta = data["meta"]
    md = [
        "# Resources (process level)",
        "",
        "Generated by `codelangtm resources`; do not edit by hand. Re-run to refresh.",
        "",
        "## Setup",
        "",
        f"- Generated: {meta['generated']}; config hash `{meta['config_hash']}`; "
        f"dataset train {meta['n_train']}, test {meta['n_test']}",
        f"- Every number comes from a fresh Python process doing one job; each job ran "
        f"{meta['repeats']} times and the median is shown. Memory: {meta['memory_source']}, "
        "which includes memory allocated in C extensions (TMU, liblinear).",
        "- **Training:** load the data, then fit binarizer + classifier on all of train. "
        "*Added* = peak process memory during training minus the memory before it (after "
        "imports and data loading).",
        f"- **Prediction:** load the saved model (TM: the JSON model file, NumPy only, no TMU; "
        f"baselines: the pickled pipeline), then classify each of the {meta['n_test']} test "
        f"snippets one at a time, {meta['passes']} passes; binarize and predict timed "
        "separately (median and 95th percentile per snippet); plus one batch pass.",
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
        + f"; CPU: {meta['cpu']}; platform {meta['platform']}",
        "",
        "## Training",
        "",
        "| Model | Wall (s) | CPU (s) | Memory before (MB) | Peak (MB) | Added by training (MB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in data["models"]:
        f = r["fit"]
        md.append(f"| {r['name']} | {f['wall_s']:.2f} | {f['cpu_s']:.2f} | "
                  f"{f['rss_before_mb']:.0f} | {f['peak_rss_mb']:.0f} | "
                  f"{f['added_mb']:.0f} |")  # fmt: skip
    md += [
        "",
        "## Prediction, one snippet at a time",
        "",
        "| Model | Model file (KB) | Load (s) | Binarize ms (median / p95) | Predict ms "
        "(median / p95) | Total ms (median / p95) | Batch ms/snippet | Peak memory (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in data["models"]:
        p = r["predict"]
        md.append(
            f"| {r['name']} | {r['model_kb']:.0f} ({r['model_kind']}) | {p['load_s']:.2f} | "
            f"{p['binarize_ms_median']:.3f} / {p['binarize_ms_p95']:.3f} | "
            f"{p['predict_ms_median']:.3f} / {p['predict_ms_p95']:.3f} | "
            f"{p['total_ms_median']:.3f} / {p['total_ms_p95']:.3f} | "
            f"{p['batch_ms_per_snippet']:.3f} | {p['peak_rss_mb']:.0f} |"
        )
    md += ["", "![Process-level resources](figures/process_resources.png)"]
    return "\n".join(md) + "\n"


def summary_table(data: dict) -> str:
    rows = [f"{'model':<22}{'fit s':>8}{'fit +MB':>9}{'1-snip ms':>11}{'p95 ms':>9}{'KB':>7}"]
    for r in data["models"]:
        f, p = r["fit"], r["predict"]
        rows.append(f"{r['name']:<22}{f['wall_s']:>8.2f}{f['added_mb']:>9.0f}"
                    f"{p['total_ms_median']:>11.3f}{p['total_ms_p95']:>9.3f}"
                    f"{r['model_kb']:>7.0f}")  # fmt: skip
    return "\n".join(rows)


if __name__ == "__main__":  # worker entry point: python -m codelangtm.resources <job> ...
    sys.exit(_worker_main(sys.argv[1:]))
