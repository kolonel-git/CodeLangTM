"""Exact and near-duplicate removal (MinHash + LSH, verified by true Jaccard)."""

from __future__ import annotations

import hashlib
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import Snippet

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_PRIME = (1 << 61) - 1
_SHINGLE = 5


def normalize(text: str) -> str:
    """Whitespace-insensitive form used for exact-duplicate hashing."""
    return " ".join(text.split())


def shingles(text: str, k: int = _SHINGLE) -> frozenset[int]:
    toks = _TOKEN_RE.findall(text)
    if len(toks) <= k:
        return frozenset({zlib.crc32(" ".join(toks).encode())})
    return frozenset(
        zlib.crc32(" ".join(toks[i : i + k]).encode()) for i in range(len(toks) - k + 1)
    )


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class DedupStats:
    total: int
    exact_removed: int
    near_removed: int

    @property
    def kept(self) -> int:
        return self.total - self.exact_removed - self.near_removed


def _signature(sh: frozenset[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # h < 2^32 (crc32) and a, b < 2^28, so a*h + b < 2^61 fits in uint64 without overflow.
    h = np.fromiter(sh, dtype=np.uint64, count=len(sh))
    sig = np.empty(len(a), dtype=np.uint64)
    for i in range(len(a)):
        sig[i] = ((h * a[i] + b[i]) % np.uint64(_PRIME)).min()
    return sig


def dedup(
    snippets: Sequence[Snippet],
    threshold: float = 0.8,
    bands: int = 16,
    rows: int = 4,
    seed: int = 0,
) -> tuple[list[Snippet], DedupStats]:
    """Keep the first occurrence of each snippet; drop exact and near duplicates.

    Near duplicates: true Jaccard of 5-token shingles >= `threshold`. Candidates are found
    with MinHash LSH (bands x rows), so cost stays near-linear for large datasets.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    rng = np.random.default_rng(seed)
    n_perm = bands * rows
    a = rng.integers(1, 1 << 28, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 28, size=n_perm, dtype=np.uint64)

    seen_exact: set[str] = set()
    buckets: list[dict[bytes, list[int]]] = [{} for _ in range(bands)]
    kept: list[Snippet] = []
    kept_sh: list[frozenset[int]] = []
    exact_removed = near_removed = 0

    for s in snippets:
        digest = hashlib.sha1(normalize(s.text).encode()).hexdigest()
        if digest in seen_exact:
            exact_removed += 1
            continue
        seen_exact.add(digest)

        sh = shingles(s.text)
        sig = _signature(sh, a, b)
        keys = [sig[i * rows : (i + 1) * rows].tobytes() for i in range(bands)]
        candidates = {j for band, key in zip(buckets, keys, strict=True) for j in band.get(key, ())}
        if any(jaccard(sh, kept_sh[j]) >= threshold for j in candidates):
            near_removed += 1
            continue

        idx = len(kept)
        kept.append(s)
        kept_sh.append(sh)
        for band, key in zip(buckets, keys, strict=True):
            band.setdefault(key, []).append(idx)

    return kept, DedupStats(len(snippets), exact_removed, near_removed)
