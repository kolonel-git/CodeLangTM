"""Leak-free splits: every repo lands wholly in train or wholly in test.

Snippets from one repo share identifiers, style and boilerplate. A random snippet-level split
lets the model memorise a repo and inflates test scores. Grouping by `Snippet.group_key`
(the repo) prevents that.

`stable_split` (used by `data build`) assigns repos by hashing their names within each
language, so changing one language's repos never moves another language's repos, and adding
or removing a repo moves at most one existing repo in or out of test. `group_split` /
`group_kfold` (StratifiedGroupKFold) remain as the random alternative.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import Snippet

SPLIT_SALT = "codelangtm-v1"


def _unit(salt: str, kind: str, key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) for a key."""
    digest = hashlib.sha256(f"{salt}:{kind}:{key}".encode()).hexdigest()
    return int(digest[:15], 16) / 16**15


def repo_languages(snippets: Sequence[Snippet]) -> dict[str, str]:
    """Each repo's language: the most common label among its snippets (ties: alphabetical)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for s in snippets:
        counts[s.group_key][s.language] += 1
    return {repo: min(c, key=lambda lang: (-c[lang], lang)) for repo, c in counts.items()}


@dataclass(frozen=True)
class StableSplit:
    test_repos: frozenset[str]
    fold_of: dict[str, int]  # train repo -> CV fold

    def split(self, snippets: Sequence[Snippet]) -> tuple[list[Snippet], list[Snippet]]:
        train = [s for s in snippets if s.group_key not in self.test_repos]
        test = [s for s in snippets if s.group_key in self.test_repos]
        return train, test

    def folds(self, train: Sequence[Snippet]) -> list[int]:
        return [self.fold_of[s.group_key] for s in train]


def stable_split(
    snippets: Sequence[Snippet],
    test_size: float = 0.2,
    n_folds: int = 5,
    salt: str = SPLIT_SALT,
) -> StableSplit:
    """Hash-based, per-language repo split with CV folds.

    Within each language, repos are ordered by a hash of their name; the first
    round(n * test_size) go to test (at least 1 and at most n - 1 when n > 1). Remaining repos
    are ordered by a second hash and cut into `n_folds` equal consecutive folds. Different
    `salt` values give independent splits (used for repeated-split evaluation).
    """
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be in (0, 0.5)")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    by_language: dict[str, list[str]] = defaultdict(list)
    for repo, lang in repo_languages(snippets).items():
        by_language[lang].append(repo)

    test: set[str] = set()
    fold_of: dict[str, int] = {}
    for lang in sorted(by_language):
        repos = sorted(by_language[lang], key=lambda r: (_unit(salt, "test", r), r))
        k = round(len(repos) * test_size)
        if len(repos) > 1:
            k = min(max(k, 1), len(repos) - 1)
        test.update(repos[:k])
        train_repos = sorted(repos[k:], key=lambda r: (_unit(salt, "fold", r), r))
        for i, repo in enumerate(train_repos):
            fold_of[repo] = i * n_folds // len(train_repos)
    return StableSplit(frozenset(test), fold_of)


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
