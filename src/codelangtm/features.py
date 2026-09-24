"""Feature extraction: snippets -> binary vectors -> 2M literals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

DELIMITERS = (
    ";", "{", "}", "(", ")", "[", "]", "//", "#", "/*", "--",
    "<", ">", "\t", "    ", "::", "->", "=>",
)  # fmt: skip


class Binarizer:
    """Fit top-M char 2/3-grams (plus structural delimiters) and emit binary features."""

    def __init__(self, n_features: int = 500, ngram_sizes: tuple[int, ...] = (2, 3)) -> None:
        self.n_features = n_features
        self.ngram_sizes = ngram_sizes
        self.vocabulary_: list[str] = []

    def _ngrams(self, text: str) -> set[str]:
        return {text[i : i + n] for n in self.ngram_sizes for i in range(len(text) - n + 1)}

    def fit(self, snippets: Iterable[str]) -> Binarizer:
        counts: Counter[str] = Counter()
        for s in snippets:
            counts.update(self._ngrams(s))  # document frequency
        budget = max(self.n_features - len(DELIMITERS), 0)
        top = [g for g, _ in counts.most_common() if g not in DELIMITERS][:budget]
        self.vocabulary_ = list(DELIMITERS) + top
        return self

    def transform(self, snippets: Iterable[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise RuntimeError("Binarizer is not fitted")
        rows = [[tok in s for tok in self.vocabulary_] for s in snippets]
        return np.asarray(rows, dtype=np.uint32)


def expand_literals(x: np.ndarray) -> np.ndarray:
    """[x_1..x_M] -> [x_1..x_M, not x_1..not x_M] (2M literals)."""
    return np.concatenate([x, 1 - x], axis=1)
