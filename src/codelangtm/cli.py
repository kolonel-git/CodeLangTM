"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from . import LANGUAGES, __version__


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
        "--language", action="append", choices=LANGUAGES,
        help="language to collect (repeatable; default: all core languages)",
    )  # fmt: skip
    gh.add_argument("--repos", type=int, default=25, help="repos per language")
    gh.add_argument("--per-repo", type=int, default=5, help="max snippets per repo")
    gh.add_argument("--min-stars", type=int, default=50)
    gh.add_argument("--seed", type=int, default=0)
    gh.add_argument("--out", type=Path, default=Path("data/raw/github"))
    gh.add_argument("--cache", type=Path, default=Path("data/cache/github"))

    data = sub.add_parser("data", help="dataset tools")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="merge sources into train/test/wild splits")
    build.add_argument(
        "--source", action="append", type=Path,
        help="directory (searched recursively) or .jsonl file; repeatable (default: data/raw)",
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


def _baselines(args: argparse.Namespace) -> int:
    from .baselines import best_by_cv, render_results_md, run_baselines, summary_table
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
    print(summary_table(results))
    print(f"\nbest by CV: {best_by_cv(results).name}\nwrote {cfg.out}")
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
            args.source or [Path("data/raw")], args.out, args.test_size, args.folds, args.salt
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
        "seed": args.seed,
    }
    with client:
        for language in args.language or LANGUAGES:
            snippets, report = github.collect_language(
                client, language, args.repos, args.per_repo, args.min_stars, args.seed
            )
            save_snippets(snippets, args.out / f"{language}.jsonl")
            manifest = {
                **report.to_dict(),
                "source": "github",
                "collected_at": date.today().isoformat(),
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
    if args.command == "diagnose":
        return _diagnose(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
