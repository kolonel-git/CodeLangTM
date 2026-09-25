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


PACK_MAX = 3  # n-grams up to this length pack exactly into int64 (21 bits per code point)
_BITS = 21  # code points are < 0x110000 < 2**21
DIRECT_MAX = 1 << 22  # largest direct-address table (entries); beyond it, binary search


def _code_points(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32).astype(
        np.int64
    )


def _pack(codes: np.ndarray, n: int) -> np.ndarray:
    """All n-grams (n <= PACK_MAX) of a code-point array as exact int64 keys, one per position."""
    if len(codes) < n:
        return np.empty(0, dtype=np.int64)
    key = codes[: len(codes) - n + 1]
    for k in range(1, n):
        key = (key << _BITS) | codes[k : len(codes) - n + 1 + k]
    return key


def _key(term: str) -> int:
    key = 0
    for ch in term:
        key = (key << _BITS) | ord(ch)
    return key


class _Lookup:
    """Where each vocabulary term lives, grouped by how it is matched."""

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary
        packed: dict[int, list[tuple[int, int]]] = {}
        long_terms: dict[int, list[tuple[str, int]]] = {}
        words: list[tuple[str, int]] = []
        for col, term in enumerate(vocabulary):
            if term.startswith(WORD_PREFIX):
                words.append((term, col))
            elif len(term) <= PACK_MAX:
                packed.setdefault(len(term), []).append((_key(term), col))
            else:
                long_terms.setdefault(len(term), []).append((term, col))
        self.packed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for n, pairs in packed.items():
            pairs.sort()
            self.packed[n] = (
                np.asarray([k for k, _ in pairs], dtype=np.int64),
                np.asarray([c for _, c in pairs], dtype=np.intp),
            )
        self._build_direct_tables(vocabulary, packed)
        self.long_terms = long_terms
        self.words = words

    def _build_direct_tables(self, vocabulary: list[str], packed: dict) -> None:
        """Direct-address tables: char -> small id, then n-gram of ids -> column (or -1).

        Much faster than binary search. Characters outside the vocabulary share one "unknown" id
        whose n-grams are never in the table, so they can never match. A length whose table would
        be too large keeps the binary-search path (`self.packed`).
        """
        chars = sorted({ch for term in vocabulary if len(term) <= PACK_MAX for ch in term})
        self.base = len(chars) + 1  # ids 0..len(chars)-1 are known chars; len(chars) is unknown
        self.max_code = max((ord(c) for c in chars), default=0) + 1
        self.char_id = np.full(self.max_code + 1, self.base - 1, dtype=np.intp)
        ids = {ch: i for i, ch in enumerate(chars)}
        for ch, i in ids.items():
            self.char_id[ord(ch)] = i
        self.direct: dict[int, np.ndarray] = {}
        for n in packed:
            size = self.base**n
            if size > DIRECT_MAX:
                continue
            self.direct[n] = np.full(size, -1, dtype=np.int32)
        for col, term in enumerate(vocabulary):
            table = self.direct.get(len(term))
            if table is not None and not term.startswith(WORD_PREFIX):
                key = 0
                for ch in term:
                    key = key * self.base + ids[ch]
                table[key] = col

    def fill(self, row: np.ndarray, text: str) -> None:
        if self.packed:
            codes = _code_points(text)
            small = None
            for n, (keys, cols) in self.packed.items():
                if len(codes) < n:
                    continue
                table = self.direct.get(n)
                if table is not None:
                    if small is None:
                        small = self.char_id[np.minimum(codes, self.max_code)]
                    key = small[: len(small) - n + 1]
                    for k in range(1, n):
                        key = key * self.base + small[k : len(small) - n + 1 + k]
                    hit = table[key]
                    row[hit[hit >= 0]] = 1
                    continue
                grams = _pack(codes, n)
                pos = np.searchsorted(keys, grams)
                pos[pos == len(keys)] = 0  # any in-range index; the equality test rejects it
                hit = keys[pos] == grams
                row[cols[pos[hit]]] = 1
        for n, terms in self.long_terms.items():
            present = _ngrams(text, (n,))
            for term, col in terms:
                if term in present:
                    row[col] = 1
        if self.words:
            present = _words(text)
            for term, col in self.words:
                if term in present:
                    row[col] = 1


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

    def _lookup(self) -> _Lookup:
        """Lookup tables for `transform`, built once per fitted vocabulary (not pickled)."""
        cached = self.__dict__.get("_lookup_cache")
        if cached is None or cached.vocabulary is not self.vocabulary_:
            cached = _Lookup(self.vocabulary_)
            self._lookup_cache = cached
        return cached

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.pop("_lookup_cache", None)  # derived from vocabulary_; keeps pickles small
        return state

    def transform(self, snippets: Iterable[str]) -> np.ndarray:
        """Binary presence matrix (n_snippets, len(vocabulary_)), dtype uint32.

        Same values as testing `term in snippet` for every term, computed without building the
        set of all substrings: 1-3 character terms are matched on packed integers with NumPy.
        """
        check_is_fitted(self, "vocabulary_")
        lookup = self._lookup()
        snippets = list(snippets)
        out = np.zeros((len(snippets), len(self.vocabulary_)), dtype=np.uint32)
        for row, text in zip(out, snippets, strict=True):
            lookup.fill(row, text)
        return out

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
