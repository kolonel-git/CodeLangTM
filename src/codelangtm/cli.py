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
    return parser


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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
