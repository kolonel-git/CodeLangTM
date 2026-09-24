"""Feature extraction: snippets -> binary vectors -> 2M literals."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_is_fitted

DELIMITERS = (
    ";", "{", "}", "(", ")", "[", "]", "//", "#", "/*", "--",
    "<", ">", "\t", "    ", "::", "->", "=>",
)  # fmt: skip

SELECTIONS = ("frequency", "chi2", "class_balanced")

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Word features are stored with this prefix so they can never collide with a character n-gram:
# snippets never contain NUL (binary files are dropped by the label check).
WORD_PREFIX = "\x00"


def _ngrams(text: str, sizes: Iterable[int]) -> set[str]:
    return {text[i : i + n] for n in sizes for i in range(len(text) - n + 1)}


def _words(text: str) -> set[str]:
    return {WORD_PREFIX + w for w in WORD.findall(text)}


def feature_name(token: str) -> str:
    """Human-readable feature name: `word:SELECT` for word features, the n-gram otherwise."""
    return f"word:{token[1:]}" if token.startswith(WORD_PREFIX) else token


def _presence_matrix(docs: Sequence[set[str]], candidates: Sequence[str]) -> csr_matrix:
    index = {g: i for i, g in enumerate(candidates)}
    rows, cols = [], []
    for r, doc in enumerate(docs):
        for g in doc:
            i = index.get(g)
            if i is not None:
                rows.append(r)
                cols.append(i)
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(len(docs), len(candidates)))


def _chi2_order(x: csr_matrix, y: np.ndarray) -> np.ndarray:
    scores, _ = chi2(x, y)
    # Stable sort keeps the alphabetical candidate order among equal scores.
    return np.argsort(-np.nan_to_num(scores), kind="stable")


def _class_balanced_order(x: csr_matrix, y: np.ndarray, budget: int) -> list[int]:
    """Round-robin over languages, each taking its next most distinctive feature.

    Distinctiveness of feature g for class c: P(g | c) - P(g | not c), from document
    frequencies. Every class gets an equal share of the budget.
    """
    n = len(y)
    total = np.asarray(x.sum(axis=0)).ravel()
    orders = []
    for c in sorted(set(y.tolist())):
        mask = y == c
        n_c = int(mask.sum())
        df_c = np.asarray(x[mask].sum(axis=0)).ravel()
        p = df_c / n_c
        q = (total - df_c) / (n - n_c) if n > n_c else np.zeros_like(p)
        orders.append(np.argsort(-(p - q), kind="stable"))
    chosen: dict[int, None] = {}
    for rank in range(x.shape[1]):
        for order in orders:
            if len(chosen) >= budget:
                return list(chosen)
            chosen.setdefault(int(order[rank]), None)
    return list(chosen)


class Binarizer(BaseEstimator, TransformerMixin):
    """Top-M character n-grams (plus structural delimiters) as binary presence features.

    Options:
    - `selection`: how the M slots are filled. `frequency` (document frequency, unsupervised),
      `chi2` (class association) or `class_balanced` (round-robin of per-language distinctive
      features). The last two use labels, so `fit` needs `y`.
    - `min_df`: candidates must occur in at least this many training snippets.
    - `word_tokens`: whole identifier/keyword tokens (`SELECT`, `fn`) join the candidates.

    A scikit-learn transformer: inside a Pipeline, the vocabulary (and any label-based
    selection) is refit on each training fold only. Ties break alphabetically, so the
    vocabulary does not depend on input order.
    """

    def __init__(
        self,
        n_features: int = 500,
        ngram_sizes: tuple[int, ...] = (2, 3),
        use_delimiters: bool = True,
        selection: str = "frequency",
        min_df: int = 1,
        word_tokens: bool = False,
    ) -> None:
        self.n_features = n_features
        self.ngram_sizes = ngram_sizes
        self.use_delimiters = use_delimiters
        self.selection = selection
        self.min_df = min_df
        self.word_tokens = word_tokens

    def _candidates_of(self, text: str) -> set[str]:
        feats = _ngrams(text, self.ngram_sizes)
        return feats | _words(text) if self.word_tokens else feats

    def fit(self, snippets: Iterable[str], y: object = None) -> Binarizer:
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")
        if self.min_df < 1:
            raise ValueError(f"min_df must be >= 1, got {self.min_df}")
        docs = [self._candidates_of(s) for s in snippets]
        counts: Counter[str] = Counter()
        for doc in docs:
            counts.update(doc)  # document frequency
        base = list(DELIMITERS) if self.use_delimiters else []
        excluded = set(base)
        candidates = sorted(g for g, c in counts.items() if c >= self.min_df and g not in excluded)

        if self.selection == "frequency":
            ranked = sorted(candidates, key=lambda g: -counts[g])  # stable: alphabetical ties
        else:
            if y is None:
                raise ValueError(f"selection={self.selection!r} needs labels: call fit(X, y)")
            labels = np.asarray(y)
            if len(labels) != len(docs):
                raise ValueError("y must have one label per snippet")
            x = _presence_matrix(docs, candidates)
            budget = max(self.n_features - len(base), 0)
            if self.selection == "chi2":
                order = _chi2_order(x, labels)[:budget]
            else:
                order = _class_balanced_order(x, labels, budget)
            ranked = [candidates[i] for i in order]
        self.vocabulary_: list[str] = (base + ranked)[: self.n_features]
        return self

    def transform(self, snippets: Iterable[str]) -> np.ndarray:
        check_is_fitted(self, "vocabulary_")
        lengths = sorted({len(t) for t in self.vocabulary_ if not t.startswith(WORD_PREFIX)})
        use_words = any(t.startswith(WORD_PREFIX) for t in self.vocabulary_)
        rows = []
        for s in snippets:
            present = _ngrams(s, lengths)  # same result as substring search, much faster
            if use_words:
                present |= _words(s)
            rows.append([t in present for t in self.vocabulary_])
        return np.asarray(rows, dtype=np.uint32).reshape(len(rows), len(self.vocabulary_))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "vocabulary_")
        return np.asarray([feature_name(t) for t in self.vocabulary_], dtype=object)

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
