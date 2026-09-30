"""Assemble collected sources into leak-free train / test / wild files with CV folds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import ALL_LANGUAGES
from .data import Snippet, languages_present, load_snippets, save_snippets
from .dedup import dedup
from .labels import check_label, filter_labels
from .splits import SPLIT_SALT, check_no_leakage, separate_wild, stable_split

MIN_TRAIN_REPOS = 5  # fewer repos per language makes stratified group CV unreliable
SPLITS = ("train", "test", "wild")


@dataclass
class BuildReport:
    loaded: int = 0
    label_dropped: Counter[str] = field(default_factory=Counter)
    duplicates_removed: int = 0
    counts: dict[str, Counter[str]] = field(default_factory=dict)  # split -> language counts
    repos: dict[str, int] = field(default_factory=dict)  # split -> distinct repos
    warnings: list[str] = field(default_factory=list)
    # hard examples: total, per language, and where their repo sits in the split
    hard: dict = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # file name -> sha256

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "label_dropped": dict(self.label_dropped.most_common()),
            "duplicates_removed": self.duplicates_removed,
            "counts": {k: dict(sorted(v.items())) for k, v in self.counts.items()},
            "repos": self.repos,
            "warnings": self.warnings,
            "hard": self.hard,
            "files": self.files,
        }

    def summary(self, languages: Sequence[str] | None = None) -> str:
        if languages is None:  # every language that occurs in any split
            languages = [
                lang for lang in ALL_LANGUAGES if any(self.counts[s][lang] for s in SPLITS)
            ]
        rows = [f"{'language':<12}{'train':>7}{'test':>7}{'wild':>7}"]
        for lang in languages:
            rows.append(f"{lang:<12}" + "".join(f"{self.counts[s][lang]:>7}" for s in SPLITS))
        totals = "".join(f"{sum(self.counts[s].values()):>7}" for s in SPLITS)
        rows.append(f"{'total':<12}{totals}")
        rows.append(
            f"loaded {self.loaded}, label-dropped {sum(self.label_dropped.values())}, "
            f"duplicates removed {self.duplicates_removed}"
        )
        if self.hard.get("count"):
            where = ", ".join(f"{k} {v}" for k, v in self.hard["repo_in"].items() if v)
            rows.append(f"hard examples (evaluation only): {self.hard['count']} ({where})")
        rows += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(rows)


def load_sources(
    paths: Sequence[str | Path], languages: Sequence[str] = ALL_LANGUAGES
) -> list[Snippet]:
    """Load every .jsonl file (directories searched recursively, in sorted order)."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.rglob("*.jsonl"))
        elif p.is_file() and p.suffix == ".jsonl":
            files.append(p)
        else:
            raise FileNotFoundError(f"not a directory or .jsonl file: {p}")
    if not files:
        raise FileNotFoundError(f"no .jsonl files found in: {', '.join(map(str, paths))}")
    out: list[Snippet] = []
    for f in files:
        out.extend(load_snippets(f, tuple(languages)))
    return out


def load_hard(
    paths: Sequence[str | Path], languages: Sequence[str] = ALL_LANGUAGES
) -> list[Snippet]:
    """Hard examples: the same JSONL format, but no files at all is fine."""
    try:
        return load_sources(paths, languages) if paths else []
    except FileNotFoundError as e:
        if "no .jsonl files" in str(e):
            return []
        raise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(
    sources: Sequence[str | Path],
    out: str | Path,
    test_size: float = 0.2,
    n_folds: int = 5,
    salt: str = SPLIT_SALT,
    languages: Sequence[str] | None = None,
    hard_sources: Sequence[str | Path] = (),
) -> BuildReport:
    """Build train/test/wild + folds. `languages` restricts which labels are accepted (default:
    all 14); the checks and the manifest cover the languages that actually occur.

    `hard_sources`: mostly-embedded windows set aside by the collector. They are written to
    `hard.jsonl` for evaluation only, are never part of train/test/folds, and a window that
    duplicates a training, test or wild snippet is dropped from them."""
    report = BuildReport()
    snippets = load_sources(sources, languages or ALL_LANGUAGES)
    languages = languages or languages_present(snippets)
    report.loaded = len(snippets)

    hard_found: list[Snippet] = []  # mostly-embedded windows found among the main sources
    snippets, report.label_dropped = filter_labels(snippets, hard_found)

    # Training candidates first: when a wild snippet duplicates one of them, the wild copy goes.
    main, wild = separate_wild(snippets)
    kept, stats = dedup([*main, *wild])
    report.duplicates_removed = stats.exact_removed + stats.near_removed
    main, wild = separate_wild(kept)

    hard_loaded = [*hard_found, *load_hard(hard_sources, languages or ALL_LANGUAGES)]
    hard_raw = [s for s in hard_loaded if s.source != "wild" and check_label(s).hard]
    if len(hard_raw) < len(hard_loaded):
        report.warnings.append(
            f"{len(hard_loaded) - len(hard_raw)} hard example(s) ignored: no longer "
            "mostly-embedded (or failing another label check)"
        )
    hard_kept, _ = dedup([*main, *wild, *hard_raw])
    hard_ids = {id(s) for s in hard_raw}
    hard = [s for s in hard_kept if id(s) in hard_ids]

    split = stable_split(main, test_size, n_folds, salt)
    train, test = split.split(main)
    check_no_leakage(train, test)
    check_no_leakage(main, wild)
    folds = split.folds(train)

    parts = {"train": train, "test": test, "wild": wild}
    where = {r: "train" for r in {s.repo for s in train}} | {s.repo: "test" for s in test}
    report.hard = {
        "count": len(hard),
        "by_language": dict(Counter(s.language for s in hard)),
        "repo_in": dict(Counter(where.get(s.repo, "neither") for s in hard)),
    }
    report.counts = {name: Counter(s.language for s in part) for name, part in parts.items()}
    report.repos = {name: len({s.repo for s in part}) for name, part in parts.items()}
    for lang in languages:
        n_repos = len({s.repo for s in train if s.language == lang})
        if n_repos == 0:
            report.warnings.append(f"{lang}: no training data")
        elif n_repos < MIN_TRAIN_REPOS:
            report.warnings.append(
                f"{lang}: only {n_repos} training repo(s); CV will be unreliable"
            )
        if n_repos and not report.counts["test"][lang]:
            report.warnings.append(f"{lang}: no test data")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        save_snippets(part, out / f"{name}.jsonl")
    save_snippets(hard, out / "hard.jsonl")
    (out / "folds.json").write_text(
        json.dumps({"n_folds": n_folds, "salt": salt, "folds": folds}), encoding="utf-8"
    )
    written = [f"{name}.jsonl" for name in parts] + ["hard.jsonl", "folds.json"]
    report.files = {name: _sha256(out / name) for name in written}
    manifest = {
        **report.to_dict(),
        "params": {
            "split": "stable-hash",
            "test_size": test_size,
            "n_folds": n_folds,
            "salt": salt,
        },
        "sources": [str(p) for p in sources],
        "languages": list(languages),
    }
    (out / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report
