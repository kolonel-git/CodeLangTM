"""Snippet record schema and JSONL I/O. See docs/data-sources.md."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from . import LANGUAGES

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


__all__ = ["LANGUAGES", "SOURCES", "Snippet", "load_snippets", "save_snippets"]
