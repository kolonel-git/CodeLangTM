"""Feature extraction: snippets -> binary vectors -> 2M literals."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DELIMITERS = (
    ";", "{", "}", "(", ")", "[", "]", "//", "#", "/*", "--",
    "<", ">", "\t", "    ", "::", "->", "=>",
)  # fmt: skip


def _ngrams(text: str, sizes: Iterable[int]) -> set[str]:
    return {text[i : i + n] for n in sizes for i in range(len(text) - n + 1)}


class Binarizer(BaseEstimator, TransformerMixin):
    """Top-M char n-grams by document frequency (plus structural delimiters) as binary features.

    A scikit-learn transformer, so inside a Pipeline the vocabulary is refit on each training
    fold and never sees validation or test snippets. Frequency ties break alphabetically, so the
    vocabulary does not depend on input order.
    """

    def __init__(
        self,
        n_features: int = 500,
        ngram_sizes: tuple[int, ...] = (2, 3),
        use_delimiters: bool = True,
    ) -> None:
        self.n_features = n_features
        self.ngram_sizes = ngram_sizes
        self.use_delimiters = use_delimiters

    def fit(self, snippets: Iterable[str], y: object = None) -> Binarizer:
        counts: Counter[str] = Counter()
        for s in snippets:
            counts.update(_ngrams(s, self.ngram_sizes))  # document frequency
        base = list(DELIMITERS) if self.use_delimiters else []
        ranked = sorted((g for g in counts if g not in base), key=lambda g: (-counts[g], g))
        self.vocabulary_: list[str] = (base + ranked)[: self.n_features]
        return self

    def transform(self, snippets: Iterable[str]) -> np.ndarray:
        check_is_fitted(self, "vocabulary_")
        lengths = sorted({len(t) for t in self.vocabulary_})
        rows = []
        for s in snippets:
            grams = _ngrams(s, lengths)  # same result as substring search, much faster
            rows.append([t in grams for t in self.vocabulary_])
        return np.asarray(rows, dtype=np.uint32).reshape(len(rows), len(self.vocabulary_))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "vocabulary_")
        return np.asarray(self.vocabulary_, dtype=object)

    def save(self, path: str | Path) -> None:
        check_is_fitted(self, "vocabulary_")
        data = {**self.get_params(), "vocabulary": self.vocabulary_}
        data["ngram_sizes"] = list(self.ngram_sizes)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Binarizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocabulary = data.pop("vocabulary")
        data["ngram_sizes"] = tuple(data["ngram_sizes"])
        b = cls(**data)
        b.vocabulary_ = vocabulary
        return b


def expand_literals(x: np.ndarray) -> np.ndarray:
    """[x_1..x_M] -> [x_1..x_M, not x_1..not x_M] (2M literals)."""
    return np.concatenate([x, 1 - x], axis=1)
