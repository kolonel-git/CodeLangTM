"""Assemble collected sources into leak-free train / test / wild files with CV folds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import LANGUAGES
from .data import Snippet, load_snippets, save_snippets
from .dedup import dedup
from .labels import filter_labels
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
    files: dict[str, str] = field(default_factory=dict)  # file name -> sha256

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "label_dropped": dict(self.label_dropped.most_common()),
            "duplicates_removed": self.duplicates_removed,
            "counts": {k: dict(sorted(v.items())) for k, v in self.counts.items()},
            "repos": self.repos,
            "warnings": self.warnings,
            "files": self.files,
        }

    def summary(self, languages: Sequence[str] = LANGUAGES) -> str:
        rows = [f"{'language':<12}{'train':>7}{'test':>7}{'wild':>7}"]
        for lang in languages:
            rows.append(f"{lang:<12}" + "".join(f"{self.counts[s][lang]:>7}" for s in SPLITS))
        totals = "".join(f"{sum(self.counts[s].values()):>7}" for s in SPLITS)
        rows.append(f"{'total':<12}{totals}")
        rows.append(
            f"loaded {self.loaded}, label-dropped {sum(self.label_dropped.values())}, "
            f"duplicates removed {self.duplicates_removed}"
        )
        rows += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(rows)


def load_sources(
    paths: Sequence[str | Path], languages: Sequence[str] = LANGUAGES
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(
    sources: Sequence[str | Path],
    out: str | Path,
    test_size: float = 0.2,
    n_folds: int = 5,
    salt: str = SPLIT_SALT,
    languages: Sequence[str] = LANGUAGES,
) -> BuildReport:
    report = BuildReport()
    snippets = load_sources(sources, languages)
    report.loaded = len(snippets)

    snippets, report.label_dropped = filter_labels(snippets)

    # Training candidates first: when a wild snippet duplicates one of them, the wild copy goes.
    main, wild = separate_wild(snippets)
    kept, stats = dedup([*main, *wild])
    report.duplicates_removed = stats.exact_removed + stats.near_removed
    main, wild = separate_wild(kept)

    split = stable_split(main, test_size, n_folds, salt)
    train, test = split.split(main)
    check_no_leakage(train, test)
    check_no_leakage(main, wild)
    folds = split.folds(train)

    parts = {"train": train, "test": test, "wild": wild}
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
    (out / "folds.json").write_text(
        json.dumps({"n_folds": n_folds, "salt": salt, "folds": folds}), encoding="utf-8"
    )
    written = [f"{name}.jsonl" for name in parts] + ["folds.json"]
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
