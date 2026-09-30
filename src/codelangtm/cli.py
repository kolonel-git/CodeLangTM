"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

from . import ALL_LANGUAGES, __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codelangtm", description="Identify the programming language of a code snippet."
    )
    parser.add_argument("--version", action="version", version=f"codelangtm {__version__}")
    sub = parser.add_subparsers(dest="command")

    collect = sub.add_parser("collect", help="collect real-world snippets")
    sources = collect.add_subparsers(dest="source", required=True)
    gh = sources.add_parser("github", help="permissively licensed GitHub repos")
    gh.add_argument(
        "--language", action="append", choices=ALL_LANGUAGES,
        help="language to collect (repeatable; default: all 14 languages)",
    )  # fmt: skip
    gh.add_argument("--repos", type=int, default=25, help="repos per language")
    gh.add_argument("--per-repo", type=int, default=5, help="max snippets per repo")
    gh.add_argument("--min-stars", type=int, default=50)
    gh.add_argument(
        "--pages", type=int, default=1,
        help="search result pages (100 repos each) read per star band; raise it for languages "
        "with few permissive repos (default 1, the setting used for dataset v5)",
    )  # fmt: skip
    gh.add_argument("--seed", type=int, default=0)
    gh.add_argument("--out", type=Path, default=Path("data/raw/github"))
    gh.add_argument("--cache", type=Path, default=Path("data/cache/github"))
    gh.add_argument(
        "--hard-out", type=Path, default=Path("data/hard/github"),
        help="where mostly-embedded windows (hard examples) go; never under --out, so "
        "`data build` cannot train on them",
    )  # fmt: skip

    data = sub.add_parser("data", help="dataset tools")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="merge sources into train/test/wild splits")
    build.add_argument(
        "--source", action="append", type=Path,
        help="directory (searched recursively) or .jsonl file; repeatable (default: data/raw)",
    )  # fmt: skip
    build.add_argument(
        "--hard-source", action="append", type=Path,
        help="directory or .jsonl file of hard examples (mostly-embedded windows, evaluation "
        "only); repeatable (default: data/hard if it exists)",
    )  # fmt: skip
    build.add_argument("--out", type=Path, default=Path("data/processed"))
    build.add_argument("--test-size", type=float, default=0.2)
    build.add_argument("--folds", type=int, default=5)
    build.add_argument(
        "--salt", default="codelangtm-v1",
        help="split salt; changing it gives a different (still stable, balanced) split",
    )  # fmt: skip

    audit = data_sub.add_parser("audit", help="per-language stats, flags and review samples")
    audit.add_argument("--data", type=Path, default=Path("data/processed"))
    audit.add_argument("--out", type=Path, default=Path("data/processed/audit.md"))
    audit.add_argument("--samples", type=int, default=10, help="samples per language")
    audit.add_argument("--seed", type=int, default=0)

    # Flags default to None so only flags actually given override the config file.
    base = sub.add_parser("baselines", help="train and evaluate classical baselines")
    base.add_argument(
        "--config", type=Path,
        help="YAML experiment config, e.g. configs/baselines.yaml (flags below override it)",
    )  # fmt: skip
    base.add_argument("--data", type=Path, help="default: data/processed")
    base.add_argument(
        "--model", action="append",
        help="naive_bayes, decision_tree, logistic_regression, linear_svm or random_forest "
        "(repeatable; default: all)",
    )  # fmt: skip
    base.add_argument("--features", type=int, help="binary features M (default: 500)")
    base.add_argument("--seed", type=int, help="default: 0")
    base.add_argument(
        "--repeats", type=int,
        help="extra repo-level train/test splits for test-score spread (0 = off; default: 10)",
    )  # fmt: skip
    base.add_argument("--out", type=Path, help="default: docs/results.md")

    abl = sub.add_parser("ablate", help="feature ablations with repo-grouped CV (train only)")
    abl.add_argument("--config", type=Path, default=Path("configs/ablations.yaml"))
    abl.add_argument(
        "--study", action="append", help="run only this study (repeatable; default: all)"
    )
    abl.add_argument("--out", type=Path, help="default: the config's `out` (docs/ablations.md)")

    rep = sub.add_parser(
        "report", help="draw the report figures (docs/figures/) from the JSON sidecars"
    )
    rep.add_argument("--results", type=Path, default=Path("docs/results.json"))
    rep.add_argument("--ablations", type=Path, default=Path("docs/ablations.json"))
    rep.add_argument(
        "--curves", type=Path, default=Path("docs/tm-curves.json"),
        help="TM training curves sidecar (skipped if the file does not exist)",
    )  # fmt: skip
    rep.add_argument(
        "--tm-results", type=Path, default=Path("docs/tm-results.json"),
        help="TM results sidecar (skipped if the file does not exist)",
    )  # fmt: skip
    rep.add_argument(
        "--resources", type=Path, default=Path("docs/resources.json"),
        help="process-level resources sidecar (skipped if the file does not exist)",
    )  # fmt: skip
    rep.add_argument(
        "--clauses", type=Path, default=Path("docs/clauses.json"),
        help="clause inspector sidecar (skipped if the file does not exist)",
    )  # fmt: skip
    rep.add_argument(
        "--errors", type=Path, default=Path("docs/errors.json"),
        help="error analysis sidecar (skipped if the file does not exist)",
    )  # fmt: skip
    rep.add_argument("--out", type=Path, default=Path("docs/figures"))

    res = sub.add_parser(
        "resources", help="process-level fit/predict resources, each job in a fresh process"
    )
    res.add_argument("--config", type=Path, default=Path("configs/tm.yaml"))
    res.add_argument(
        "--results", type=Path, default=Path("docs/tm-results.json"),
        help="tm-results sidecar (picks each TM setting's median-CV seed)",
    )  # fmt: skip
    res.add_argument("--models-dir", type=Path, default=Path("models"))
    res.add_argument("--repeats", type=int, default=3, help="runs per job (median kept)")
    res.add_argument("--passes", type=int, default=3, help="passes over the test set per run")
    res.add_argument("--out", type=Path, default=Path("docs/resources.md"))

    tmr = sub.add_parser(
        "tm-results", help="TM vs baselines: full protocol per seed + significance tests"
    )
    tmr.add_argument("--config", type=Path, default=Path("configs/tm.yaml"))
    tmr.add_argument(
        "--setting", action="append", help="TM setting from the config (repeatable; default: all)"
    )
    tmr.add_argument("--out", type=Path, help="default: the config's `out`")
    tmr.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="where the final models (one JSON per setting x seed) are saved (gitignored)",
    )  # fmt: skip
    tmr.add_argument("--no-save-models", action="store_true")

    train = sub.add_parser("tm-train", help="train one TM on all of train and save the model file")
    train.add_argument("--config", type=Path, default=Path("configs/tm.yaml"))
    train.add_argument("--setting", help="TM setting from the config (default: the first)")
    train.add_argument("--seed", type=int, default=1, help="TM seed (>= 1; default: 1)")
    train.add_argument("--out", type=Path, help="default: models/<setting>_seed<seed>.json")

    select = sub.add_parser(
        "tm-select", help="copy the median-CV seed's model (from tm-results) to the official path"
    )
    select.add_argument("--results", type=Path, default=Path("docs/tm-results.json"))
    select.add_argument("--setting", default="tm_400")
    select.add_argument("--models-dir", type=Path, default=Path("models"))
    select.add_argument("--out", type=Path, default=Path("models/tm.json"))

    cl = sub.add_parser(
        "clauses", help="clause inspector: every clause of a model file as a readable rule"
    )
    cl.add_argument("--model", type=Path, default=Path("models/tm.json"))
    cl.add_argument(
        "--data", type=Path, default=Path("data/processed"),
        help="dataset the model was trained on (clause statistics use its train split only)",
    )  # fmt: skip
    cl.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="other seeds of the same setting (<setting>_seed<k>.json) for the stability check",
    )  # fmt: skip
    cl.add_argument("--no-formation", action="store_true", help="skip the formation replay")
    cl.add_argument("--out", type=Path, default=Path("docs/clauses.md"))

    er = sub.add_parser(
        "errors", help="error analysis on out-of-fold CV predictions (train only; TM vs baselines)"
    )
    er.add_argument("--config", type=Path, default=Path("configs/tm.yaml"))
    er.add_argument("--setting", default="tm_400")
    er.add_argument(
        "--tm-results", type=Path, default=Path("docs/tm-results.json"),
        help="consistency check: per-fold scores must equal this run's (skipped if missing)",
    )  # fmt: skip
    er.add_argument(
        "--clauses", type=Path, default=Path("docs/clauses.json"),
        help="language overlap from the clause inspector (skipped if missing)",
    )  # fmt: skip
    er.add_argument(
        "--model", type=Path, default=Path("models/tm.json"),
        help="official model, for its signature n-grams (skipped if missing)",
    )  # fmt: skip
    er.add_argument("--out", type=Path, default=Path("docs/errors.md"))
    er.add_argument(
        "--review", type=Path, default=Path("data/processed/errors-review.md"),
        help="local review file with the snippets' code (keep it out of git)",
    )  # fmt: skip

    ex = sub.add_parser("explain", help="predict one snippet and show the clauses that fired")
    ex.add_argument("--model", type=Path, default=Path("models/tm.json"))
    ex.add_argument("--file", required=True, help="snippet file, or - for standard input")
    ex.add_argument("--top", type=int, default=8, help="clauses shown per language")

    curve = sub.add_parser(
        "tm-curve", help="TM training curves on CV folds (train only); recommends the epoch count"
    )
    curve.add_argument("--config", type=Path, default=Path("configs/tm.yaml"))
    curve.add_argument(
        "--setting", action="append", help="TM setting from the config (repeatable; default: all)"
    )
    curve.add_argument("--out", type=Path, help="default: the config's `curves_out`")

    diag = sub.add_parser(
        "diagnose", help="label-issue candidates and shortcut features (training data only)"
    )
    diag.add_argument("--data", type=Path, default=Path("data/processed"))
    diag.add_argument("--out", type=Path, default=Path("data/processed/diagnostics.md"))
    diag.add_argument("--features", type=int, default=500)
    diag.add_argument("--top", type=int, default=15, help="features shown per language")
    diag.add_argument("--seed", type=int, default=0)
    return parser


def _diagnose(args: argparse.Namespace) -> int:
    from .diagnostics import run_diagnostics

    try:
        result = run_diagnostics(args.data, args.out, args.features, args.seed, args.top)
    except (FileNotFoundError, ValueError) as e:
        print(f"diagnose failed: {e}", file=sys.stderr)
        return 1
    print(result.summary())
    print(f"\nreview file: {args.out}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from .figures import render_all

    optional = {}
    for key, label in (("curves", "TM curve"), ("tm_results", "TM result"),
                       ("resources", "resource"), ("clauses", "clause"),
                       ("errors", "error analysis")):  # fmt: skip
        path = getattr(args, key)
        optional[key] = path if path and path.exists() else None
        if path and optional[key] is None:
            print(f"note: {path.as_posix()} not found, skipping {label} figures")
    try:
        written = render_all(args.results, args.ablations, args.out, optional["curves"],
                             optional["tm_results"], optional["resources"],
                             optional["clauses"], optional["errors"])  # fmt: skip
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"report failed: {e}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path.as_posix()}")
    print("narrative: docs/report.md (hand-written; update its numbers if the results changed)")
    return 0


def _tm_results(args: argparse.Namespace) -> int:
    from .baselines import sidecar_path, write_json
    from .config import load_tm_config
    from .tm_results import render_tm_results_md, run_tm_results, summary_table, tm_results_json

    try:
        cfg = load_tm_config(args.config)
        models_dir = None if args.no_save_models else args.models_dir
        baselines, entries, meta = run_tm_results(
            cfg, args.setting, models_dir=models_dir, progress=lambda m: print(m, flush=True)
        )
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"tm-results failed: {e}", file=sys.stderr)
        return 1
    meta["config_path"] = args.config.as_posix()
    data = tm_results_json(cfg, baselines, entries, meta)
    out = args.out or cfg.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tm_results_md(data), encoding="utf-8")
    sidecar = write_json(data, sidecar_path(out))
    print()
    print(summary_table(data))
    print(f"\nwrote {out} and {sidecar}")
    return 0


def _resources(args: argparse.Namespace) -> int:
    from .baselines import sidecar_path, write_json
    from .figures import load_sidecar
    from .resources import measure_all, render_resources_md, resources_json, summary_table

    try:
        tm_results = load_sidecar(args.results, "codelangtm.tm-results/")
        rows, meta = measure_all(args.config, tm_results, args.models_dir, args.repeats,
                                 args.passes, progress=lambda m: print(m, flush=True))  # fmt: skip
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"resources failed: {e}", file=sys.stderr)
        return 1
    meta["config_path"] = args.config.as_posix()
    data = resources_json(rows, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_resources_md(data), encoding="utf-8")
    sidecar = write_json(data, sidecar_path(args.out))
    print()
    print(summary_table(data))
    print(f"\nwrote {args.out} and {sidecar}")
    return 0


def _tm_train(args: argparse.Namespace) -> int:
    from .config import load_tm_config
    from .tm_results import train_final

    try:
        cfg = load_tm_config(args.config)
        setting = args.setting or cfg.settings[0].name
        out = args.out or Path("models") / f"{setting}_seed{args.seed}.json"
        summary = train_final(cfg, setting, args.seed, out)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"tm-train failed: {e}", file=sys.stderr)
        return 1
    print(f"trained {setting} seed {args.seed}: {summary['epochs']} epochs in "
          f"{summary['fit_seconds']:.1f} s; train macro-F1 {summary['train_f1']:.3f}, "
          f"test macro-F1 {summary['test_f1']:.3f} (information only)")  # fmt: skip
    print(f"wrote {summary['path']}")
    return 0


def _tm_select(args: argparse.Namespace) -> int:
    from .figures import load_sidecar
    from .tm_results import select_canonical

    try:
        data = load_sidecar(args.results, "codelangtm.tm-results/")
        if args.setting not in data["tm"]:
            raise ValueError(f"setting {args.setting!r} not in {args.results}")
        meta = select_canonical(data, args.setting, args.models_dir, args.out)
    except (FileNotFoundError, ValueError) as e:
        print(f"tm-select failed: {e}", file=sys.stderr)
        return 1
    c = meta["canonical"]
    scores = ", ".join(f"seed {k}: {v:.4f}" for k, v in c["cv_by_seed"].items())
    print(f"canonical {c['setting']} model: seed {c['seed']} ({c['rule']}); CV {scores}")
    print(f"wrote {args.out.as_posix()} (from {c['source']})")
    return 0


def _seed_models(models_dir: Path, meta: dict) -> dict:
    """Other seeds of the model's setting found in `models_dir` (not the model's own seed)."""
    from .model import load_model

    setting, seed = meta.get("setting"), meta.get("seed")
    found = {}
    for path in sorted(models_dir.glob(f"{setting}_seed*.json")):
        match = re.fullmatch(rf"{re.escape(str(setting))}_seed(\d+)", path.stem)
        if match and int(match.group(1)) != seed:
            found[f"seed {match.group(1)}"] = load_model(path)
    return found


def _clauses(args: argparse.Namespace) -> int:
    from .audit import load_dataset
    from .baselines import sidecar_path
    from .model import load_model
    from .rules import clauses_json_text, inspect_model, render_clauses_md, summary_table

    try:
        model = load_model(args.model)
        train = load_dataset(args.data)["train"]
        manifest = args.data / "dataset.json"
        expected = model.meta.get("dataset_files")
        if expected and manifest.exists():
            files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
            if {k: v[:12] for k, v in files.items()} != expected:
                raise ValueError(f"{args.data} is not the dataset {args.model} was trained on")
        others = _seed_models(args.models_dir, model.meta)
        data = inspect_model(model, [s.text for s in train], [s.language for s in train], others,
                             formation=not args.no_formation,
                             progress=lambda m: print(m, flush=True))  # fmt: skip
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"clauses failed: {e}", file=sys.stderr)
        return 1
    data["model"]["path"] = args.model.as_posix()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_clauses_md(data), encoding="utf-8")
    sidecar = sidecar_path(args.out)
    sidecar.write_text(clauses_json_text(data), encoding="utf-8")
    print()
    print(summary_table(data))
    if not data["checks"]["rules_match_model"]:
        print("WARNING: rules do not reproduce the model's clause outputs", file=sys.stderr)
    print(f"\nwrote {args.out} and {sidecar}")
    return 0


def _errors(args: argparse.Namespace) -> int:
    from .audit import load_dataset
    from .baselines import sidecar_path
    from .config import load_tm_config
    from .errors import (
        analyse,
        render_errors_md,
        render_review_md,
        run_out_of_fold,
        signature_document_frequency,
        summary_table,
    )
    from .figures import load_sidecar
    from .rules import extract_rules, rows_json_text, signature_features

    try:
        cfg = load_tm_config(args.config)
        cfg.setting(args.setting)  # fail early on an unknown setting
        train = load_dataset(cfg.data)["train"]
        tm_results = overlap = signature_df = None
        if args.tm_results.exists():
            tm_results = load_sidecar(args.tm_results, "codelangtm.tm-results/")
        if args.clauses.exists():
            clauses = load_sidecar(args.clauses, "codelangtm.clauses/")
            langs = clauses["languages"]
            overlap = {a: dict(zip(langs, row, strict=True))
                       for a, row in zip(langs, clauses["overlap"], strict=True)}  # fmt: skip
        if args.model.exists():
            from .model import load_model

            model = load_model(args.model)
            vocabulary = model.binarizer.vocabulary_
            rules = extract_rules(model.state, vocabulary)
            sigs = signature_features(rules, list(model.classes), vocabulary)
            signature_df = signature_document_frequency(sigs, vocabulary, train)
        for path, what in ((args.tm_results, "consistency check"), (args.clauses, "overlap"),
                           (args.model, "signature n-grams (H6)")):  # fmt: skip
            if not path.exists():
                print(f"note: {path.as_posix()} not found, skipping the {what}")
        oof = run_out_of_fold(cfg, args.setting, progress=lambda m: print(m, flush=True))
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"errors failed: {e}", file=sys.stderr)
        return 1
    texts = {i: s.text for i, s in enumerate(train)}
    data = analyse(oof, tm_results, overlap, signature_df, texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_errors_md(data), encoding="utf-8")
    sidecar = sidecar_path(args.out)
    sidecar.write_text(rows_json_text(data, "snippets"), encoding="utf-8")
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(render_review_md(data, texts), encoding="utf-8")
    print()
    print(summary_table(data))
    c = data["consistency"]
    if c.get("checked") and not c.get("matches"):
        print(f"WARNING: per-fold scores differ from {args.tm_results} "
              f"(max {c['max_abs_diff']:.2g})", file=sys.stderr)  # fmt: skip
    print(f"\nwrote {args.out} and {sidecar}; review file (local only): {args.review}")
    return 0


def _explain(args: argparse.Namespace) -> int:
    from .model import load_model
    from .rules import explain_snippet, format_explanation

    try:
        model = load_model(args.model)
        if args.file == "-":
            text = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        else:
            text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ValueError) as e:
        print(f"explain failed: {e}", file=sys.stderr)
        return 1
    print(format_explanation(explain_snippet(model, text), top=args.top))
    return 0


def _tm_curve(args: argparse.Namespace) -> int:
    from .baselines import sidecar_path, write_json
    from .config import load_tm_config
    from .curves import curves_json, render_curves_md, run_curves, summary_table

    try:
        cfg = load_tm_config(args.config)
        runs, meta = run_curves(cfg, args.setting, progress=lambda m: print(m, flush=True))
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"tm-curve failed: {e}", file=sys.stderr)
        return 1
    meta["config_path"] = args.config.as_posix()
    data = curves_json(cfg, runs, meta)
    out = args.out or cfg.curves_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_curves_md(data), encoding="utf-8")
    sidecar = write_json(data, sidecar_path(out))
    print()
    print(summary_table(data))
    print(f"\nwrote {out} and {sidecar}")
    print(f"set `epochs` per setting in {args.config.as_posix()} to the chosen epoch, then "
          "run `codelangtm report` for the figures")  # fmt: skip
    return 0


def _ablate(args: argparse.Namespace) -> int:
    from . import ablations as ab
    from .baselines import sidecar_path, write_json
    from .config import load_ablation_config

    def progress(i: int, n: int, features: object) -> None:
        print(f"[{i + 1}/{n}] {ab.describe(features)}", flush=True)

    try:
        cfg = ab.select_studies(load_ablation_config(args.config), args.study)
        runs, meta = ab.run_ablations(cfg, progress=progress)
    except (FileNotFoundError, ValueError) as e:
        print(f"ablate failed: {e}", file=sys.stderr)
        return 1
    meta["config_path"] = args.config.as_posix()
    out = args.out or cfg.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ab.render_ablations_md(cfg, runs, meta), encoding="utf-8")
    sidecar = write_json(ab.ablations_json(cfg, runs, meta), sidecar_path(out))
    print()
    print(ab.summary_table(cfg, runs))
    print(f"\nwrote {out} and {sidecar}")
    return 0


def _baselines(args: argparse.Namespace) -> int:
    from .baselines import (
        best_by_cv,
        render_results_md,
        results_json,
        run_baselines,
        sidecar_path,
        summary_table,
        write_json,
    )
    from .config import BaselinesConfig, config_hash, load_baselines_config

    overrides = {
        "data": args.data, "models": args.model, "n_features": args.features,
        "seed": args.seed, "repeats": args.repeats, "out": args.out,
    }  # fmt: skip
    try:
        cfg = load_baselines_config(args.config) if args.config else BaselinesConfig()
        cfg = cfg.with_overrides(**overrides)
        results, meta = run_baselines(
            cfg.data, cfg.models, seed=cfg.seed, repeats=cfg.repeats,
            binarizer=cfg.features.binarizer(),
        )  # fmt: skip
    except (FileNotFoundError, ValueError) as e:
        print(f"baselines failed: {e}", file=sys.stderr)
        return 1
    meta["config"] = {
        "path": args.config.as_posix() if args.config else None,
        "overrides": sorted(k for k, v in overrides.items() if v is not None),
        "hash": config_hash(cfg),
    }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(render_results_md(results, meta), encoding="utf-8")
    sidecar = write_json(results_json(results, meta), sidecar_path(cfg.out))
    print(summary_table(results))
    print(f"\nbest by CV: {best_by_cv(results).name}\nwrote {cfg.out} and {sidecar}")
    return 0


def _data_audit(args: argparse.Namespace) -> int:
    from .audit import run_audit, summary_table

    try:
        stats, flags = run_audit(args.data, args.out, args.samples, args.seed)
    except (FileNotFoundError, ValueError) as e:
        print(f"data audit failed: {e}", file=sys.stderr)
        return 1
    print(summary_table(stats))
    print("\nflags:" if flags else "\nflags: none")
    for f in flags:
        print(f"  - {f}")
    print(f"\nreview file: {args.out}")
    return 0


def _data_build(args: argparse.Namespace) -> int:
    from .build import build_dataset

    try:
        report = build_dataset(
            args.source or [Path("data/raw")], args.out, args.test_size, args.folds, args.salt,
            hard_sources=args.hard_source
            if args.hard_source is not None
            else ([Path("data/hard")] if Path("data/hard").exists() else []),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"data build failed: {e}", file=sys.stderr)
        return 1
    print(report.summary())
    print(f"wrote {args.out}")
    return 0


def _collect_github(args: argparse.Namespace) -> int:
    try:
        from . import github
    except ImportError:
        print("Collector dependencies missing: run `uv sync --extra collect`", file=sys.stderr)
        return 1
    from .data import save_snippets

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        client = github.GitHubClient(cache_dir=args.cache)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    params = {
        "repos": args.repos,
        "per_repo": args.per_repo,
        "min_stars": args.min_stars,
        "pages": args.pages,
        "seed": args.seed,
    }
    with client:
        for language in args.language or ALL_LANGUAGES:
            snippets, report = github.collect_language(
                client, language, args.repos, args.per_repo, args.min_stars, args.seed,
                pages=args.pages,
            )
            save_snippets(snippets, args.out / f"{language}.jsonl")
            args.hard_out.mkdir(parents=True, exist_ok=True)
            save_snippets(report.hard_snippets, args.hard_out / f"{language}.jsonl")
            manifest = {
                **report.to_dict(),
                "source": "github",
                "collected_at": date.today().isoformat(),
                "search_languages": list(github.search_qualifiers(language)),
                "params": params,
            }
            (args.out / f"{language}.manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "collect" and args.source == "github":
        return _collect_github(args)
    if args.command == "data" and args.data_command == "build":
        return _data_build(args)
    if args.command == "data" and args.data_command == "audit":
        return _data_audit(args)
    if args.command == "baselines":
        return _baselines(args)
    if args.command == "ablate":
        return _ablate(args)
    if args.command == "report":
        return _report(args)
    if args.command == "tm-curve":
        return _tm_curve(args)
    if args.command == "tm-results":
        return _tm_results(args)
    if args.command == "tm-train":
        return _tm_train(args)
    if args.command == "resources":
        return _resources(args)
    if args.command == "tm-select":
        return _tm_select(args)
    if args.command == "clauses":
        return _clauses(args)
    if args.command == "errors":
        return _errors(args)
    if args.command == "explain":
        return _explain(args)
    if args.command == "diagnose":
        return _diagnose(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
