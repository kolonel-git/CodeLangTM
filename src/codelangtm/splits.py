"""Leak-free splits: every repo lands wholly in train or wholly in test.

Snippets from one repo share identifiers, style and boilerplate. A random snippet-level split
lets the model memorise a repo and inflates test scores. Grouping by `Snippet.group_key`
(the repo) prevents that; stratification keeps language proportions similar across folds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import Snippet


def separate_wild(snippets: Sequence[Snippet]) -> tuple[list[Snippet], list[Snippet]]:
    """Split off the held-out wild set (source == 'wild'); it is never used for training."""
    main = [s for s in snippets if s.source != "wild"]
    wild = [s for s in snippets if s.source == "wild"]
    return main, wild


def _folds(
    snippets: Sequence[Snippet], n_splits: int, seed: int
) -> Iterator[tuple[list[Snippet], list[Snippet]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    y = np.array([s.language for s in snippets])
    groups = np.array([s.group_key for s in snippets])
    if len(set(groups)) < n_splits:
        raise ValueError(f"need at least {n_splits} distinct repos, got {len(set(groups))}")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(np.zeros(len(snippets)), y, groups):
        yield [snippets[i] for i in train_idx], [snippets[i] for i in test_idx]


def group_kfold(
    snippets: Sequence[Snippet], n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[list[Snippet], list[Snippet]]]:
    """Stratified group k-fold CV: yields (train, test) per fold; each snippet is tested once."""
    return _folds(snippets, n_splits, seed)


def group_split(
    snippets: Sequence[Snippet], test_size: float = 0.2, seed: int = 0
) -> tuple[list[Snippet], list[Snippet]]:
    """Single train/test split by repo. Test share is approximately 1/round(1/test_size)."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be in (0, 0.5)")
    return next(_folds(snippets, round(1 / test_size), seed))


def check_no_leakage(train: Sequence[Snippet], test: Sequence[Snippet]) -> None:
    """Raise if any repo appears on both sides."""
    shared = {s.group_key for s in train} & {s.group_key for s in test}
    if shared:
        raise ValueError(f"leakage: {len(shared)} repo(s) in both sets, e.g. {sorted(shared)[:3]}")


def language_counts(snippets: Sequence[Snippet]) -> Counter[str]:
    return Counter(s.language for s in snippets)
