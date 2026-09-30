"""Snippet record schema and JSONL I/O. See docs/data-sources.md."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import ALL_LANGUAGES, CORE_LANGUAGES, LANGUAGES

SOURCES = ("github", "the-stack", "codesearchnet", "wild")


@dataclass(frozen=True)
class Snippet:
    text: str
    language: str
    source: str
    repo: str
    commit: str
    path: str
    license: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text is empty")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source: {self.source!r}")
        if not self.repo:
            raise ValueError("repo is required (used as split group)")
        if not self.license:
            raise ValueError("license is required")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("invalid line range")

    @property
    def group_key(self) -> str:
        """Splits are grouped by source repo to prevent leakage."""
        return self.repo

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1


def languages_present(snippets: Iterable[Snippet]) -> tuple[str, ...]:
    """The languages that occur in `snippets`, in the project's fixed order (core, then stretch)."""
    seen = {s.language for s in snippets}
    return tuple(lang for lang in ALL_LANGUAGES if lang in seen)


LANGUAGE_SETS = {"core": CORE_LANGUAGES, "all": ALL_LANGUAGES}
LANGUAGE_SPEC_HELP = "auto (the languages in the data), core (the original 8), all (14) or a list"


def parse_language_spec(text: str) -> str | tuple[str, ...]:
    """Command-line/YAML text -> spec: `auto`, `core`, `all` or a tuple of language names."""
    text = text.strip().lower()
    if text in ("auto", *LANGUAGE_SETS):
        return text
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names:
        raise ValueError(f"empty language list; use {LANGUAGE_SPEC_HELP}")
    return names


def resolve_languages(
    spec: str | Sequence[str] | None, present: Iterable[str]
) -> tuple[str, ...]:
    """The languages an experiment evaluates, in the project's fixed order.

    `spec`: None or "auto" = every language that occurs in the data; "core" / "all" = a fixed
    set; otherwise a list of names. Asking for a language that has no data is an error: a
    missing language must never silently drop out of a report."""
    present = set(present)
    if spec is None or spec == "auto":
        requested = present
    elif isinstance(spec, str):
        requested = set(LANGUAGE_SETS[spec]) if spec in LANGUAGE_SETS else set(
            parse_language_spec(spec)
        )
    else:
        requested = set(spec)
    unknown = sorted(requested - set(ALL_LANGUAGES))
    if unknown:
        raise ValueError(f"unknown language(s) {unknown}; known: {list(ALL_LANGUAGES)}")
    missing = sorted(requested - present)
    if missing:
        raise ValueError(
            f"no data for language(s) {missing}; the dataset has {sorted(present)}. "
            f"Choose --languages {LANGUAGE_SPEC_HELP}, or collect the missing languages"
        )
    if not requested:
        raise ValueError("no languages to evaluate: the dataset is empty")
    return tuple(lang for lang in ALL_LANGUAGES if lang in requested)


def restrict_dataset(dataset: dict[str, list], folds, languages: Sequence[str]):
    """Keep only snippets of `languages`; `folds` (one id per train snippet) is filtered too,
    so every remaining snippet keeps its original CV fold."""
    keep = set(languages)
    mask = [s.language in keep for s in dataset["train"]]
    restricted = {
        split: [s for s in part if s.language in keep] for split, part in dataset.items()
    }
    return restricted, folds[mask]


def save_snippets(snippets: Iterable[Snippet], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in snippets:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_snippets(path: str | Path, languages: tuple[str, ...] | None = None) -> Iterator[Snippet]:
    """Yield snippets from JSONL. Pass `languages` to reject unexpected labels."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            s = Snippet(**json.loads(line))
            if languages is not None and s.language not in languages:
                raise ValueError(f"{path}:{lineno}: unknown language {s.language!r}")
            yield s


__all__ = [
    "LANGUAGES",
    "SOURCES",
    "Snippet",
    "languages_present",
    "load_snippets",
    "save_snippets",
]
