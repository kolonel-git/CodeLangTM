"""Tsetlin Machine language classifier: TMU for training, plain NumPy for everything after.

- `TMLanguageClassifier` is a scikit-learn estimator over binary features, so it drops into the
  same `Pipeline([Binarizer, model])`, CV and repeated-split code as the baselines. TMU is imported
  only when training starts.
- `TMState` is a trained model as plain arrays (which literals each clause includes, clause
  weights, class names). Its NumPy prediction equals TMU's exactly (B1 spike, issues-and-fixes M8),
  so saved models can be used and inspected without TMU.
- `save_model` / `load_model` write one self-contained JSON file (format `codelangtm.tm/1`):
  feature vocabulary + clauses + weights + metadata, readable from any language (C export, web
  demo). The format is documented in docs/architecture.md.

Literal ids: with M features, id `j < M` means `has(feature j)` and id `M + j` means
`NOT has(feature j)`, the order TMU uses internally.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .features import Binarizer

MODEL_FORMAT = "codelangtm.tm/1"
LITERAL_ENCODING = (
    "literal id j < n_features means has(feature j); "
    "id n_features + j means NOT has(feature j)"
)
# TMU logs a full traceback at import when pycuda (GPU backend) is missing; harmless on CPU.
_TMU_NOISY_LOGGERS = ("tmu.clause_bank.clause_bank_cuda", "tmu.util.cuda_profiler")


def _tm_classifier_class():
    """Import TMU's classifier lazily, without its pycuda noise; clear error if TMU is absent."""
    for name in _TMU_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError as e:
        raise ImportError("the Tsetlin Machine needs TMU: run `uv sync --extra tm`") from e
    return TMClassifier


def _binary_matrix(x: object) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, got shape {x.shape}")
    if x.size and (x.min() < 0 or x.max() > 1):
        raise ValueError("features must be binary (0/1)")
    return np.ascontiguousarray(x, dtype=np.uint32)


@dataclass
class TMState:
    """A trained TM as plain arrays. Clauses are stored class by class, sparsely.

    `clause_class[i]` is the class index of clause i, `weights[i]` its signed vote, and its
    included literal ids are `indices[indptr[i]:indptr[i + 1]]` (CSR layout). Clauses with no
    included literal are kept (the inspector counts them) and never fire.
    """

    classes: tuple[str, ...]
    n_features: int
    clause_class: np.ndarray  # (n_clauses_total,) int
    weights: np.ndarray  # (n_clauses_total,) int
    indptr: np.ndarray  # (n_clauses_total + 1,) int
    indices: np.ndarray  # (n_included_literals,) int, literal ids in [0, 2 * n_features)
    params: dict = field(default_factory=dict)  # hyperparameters used for training (T, s, ...)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_tmu(cls, tm: object, classes: Sequence[str], n_features: int,
                 params: dict | None = None) -> TMState:  # fmt: skip
        """Read the include matrix and weights out of a trained TMU `TMClassifier`."""
        clause_class, weights, rows = [], [], []
        for k in range(len(classes)):
            include = np.asarray(tm.clause_banks[k].get_literals())
            if include.shape[1] != 2 * n_features:
                raise ValueError(
                    f"TMU has {include.shape[1]} literals, expected {2 * n_features} "
                    "(feature_negation must be on)"
                )
            w = np.asarray(tm.weight_banks[k].get_weights())
            clause_class.append(np.full(len(w), k))
            weights.append(w)
            rows.extend(np.flatnonzero(row) for row in include)
        return cls._from_rows(classes, n_features, np.concatenate(clause_class),
                              np.concatenate(weights), rows, params)  # fmt: skip

    @classmethod
    def _from_rows(cls, classes, n_features, clause_class, weights, rows, params) -> TMState:
        lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        indices = np.concatenate(rows).astype(np.int32) if len(rows) else np.zeros(0, np.int32)
        return cls(
            classes=tuple(classes),
            n_features=int(n_features),
            clause_class=np.asarray(clause_class, dtype=np.int32),
            weights=np.asarray(weights, dtype=np.int32),
            indptr=indptr,
            indices=indices,
            params=dict(params or {}),
        )

    # ------------------------------------------------------------------ inspection

    @property
    def n_clauses(self) -> int:
        return len(self.weights)

    def clause_literals(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i] : self.indptr[i + 1]]

    def literal_counts(self) -> np.ndarray:
        return np.diff(self.indptr)

    def include_matrix(self) -> np.ndarray:
        """Dense 0/1 include matrix (n_clauses_total, 2 * n_features), as TMU stores it."""
        dense = np.zeros((self.n_clauses, 2 * self.n_features), dtype=np.uint8)
        rows = np.repeat(np.arange(self.n_clauses), self.literal_counts())
        dense[rows, self.indices] = 1
        return dense

    # ------------------------------------------------------------------ prediction

    def _prepared(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cached matrices for prediction (derived, never saved)."""
        if "coef" not in self._cache:
            include = self.include_matrix().astype(np.float32)
            pos, neg = include[:, : self.n_features], include[:, self.n_features :]
            # A clause fires iff (included positives absent) + (included negations present) == 0,
            # i.e. pos_count - x @ pos.T + x @ neg.T == 0: one matrix product.
            coef = (neg - pos).T  # (M, n_clauses)
            pos_count = pos.sum(axis=1)
            nonempty = self.literal_counts() > 0
            vote = np.zeros((self.n_clauses, len(self.classes)), dtype=np.float32)
            vote[np.arange(self.n_clauses), self.clause_class] = self.weights
            self._cache.update(coef=coef, pos_count=pos_count, nonempty=nonempty, vote=vote)
        c = self._cache
        return c["coef"], c["pos_count"], c["nonempty"], c["vote"]

    def clause_outputs(self, x: object) -> np.ndarray:
        """(n_samples, n_clauses_total) bool: which clauses fire on each sample."""
        x = _binary_matrix(x)
        if x.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.shape[1]}")
        coef, pos_count, nonempty, _ = self._prepared()
        # float32 is exact here: every term is an integer of magnitude <= 2 * n_features.
        violations = x.astype(np.float32) @ coef + pos_count
        return (violations == 0) & nonempty

    def class_sums(self, x: object) -> np.ndarray:
        """(n_samples, n_classes) int: signed votes per class (TMU's unclipped class sums)."""
        *_, vote = self._prepared()
        return np.rint(self.clause_outputs(x).astype(np.float32) @ vote).astype(np.int64)

    def predict(self, x: object) -> np.ndarray:
        """Class names; ties go to the first class in `classes`, as in TMU."""
        return np.asarray(self.classes, dtype=object)[np.argmax(self.class_sums(x), axis=1)]

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> dict:
        clauses = {}
        for k, name in enumerate(self.classes):
            ids = np.flatnonzero(self.clause_class == k)
            clauses[name] = {
                "weights": self.weights[ids].tolist(),
                "literals": [self.clause_literals(i).tolist() for i in ids],
            }
        return {"n_features": self.n_features, "params": self.params, "clauses": clauses}

    @classmethod
    def from_dict(cls, data: dict, classes: Sequence[str]) -> TMState:
        clause_class, weights, rows = [], [], []
        for k, name in enumerate(classes):
            block = data["clauses"][name]
            if len(block["weights"]) != len(block["literals"]):
                raise ValueError(f"class {name!r}: weights and literals differ in length")
            clause_class += [k] * len(block["weights"])
            weights += block["weights"]
            rows += [np.asarray(r, dtype=np.int32) for r in block["literals"]]
        n_features = int(data["n_features"])
        if any(len(r) and (r.min() < 0 or r.max() >= 2 * n_features) for r in rows):
            raise ValueError("literal id out of range")
        return cls._from_rows(classes, n_features, clause_class, weights, rows,
                              data.get("params"))  # fmt: skip

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TMState):
            return NotImplemented
        return (
            self.classes == other.classes
            and self.n_features == other.n_features
            and self.params == other.params
            and all(np.array_equal(getattr(self, a), getattr(other, a))
                    for a in ("clause_class", "weights", "indptr", "indices"))  # fmt: skip
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = {}  # derived matrices are rebuilt on demand; keeps pickles small
        return state


class TMLanguageClassifier(ClassifierMixin, BaseEstimator):
    """Multi-class Tsetlin Machine (TMU `TMClassifier`) as a scikit-learn estimator.

    Input: the binary matrix from `Binarizer`. `n_clauses` is per class (half vote for, half
    against). `fit` trains `epochs` epochs from scratch; `partial_fit` trains one more epoch.
    `seed` must be >= 1: TMU hangs with seed 0 (issues-and-fixes E6).
    Prediction uses the NumPy `TMState` (`state_`), identical to TMU's; `predict_tmu` calls TMU.
    The TMU object is not pickled: an unpickled model predicts but cannot train further.
    """

    def __init__(
        self,
        n_clauses: int = 400,
        T: int = 100,
        s: float = 5.0,
        epochs: int = 50,
        weighted_clauses: bool = True,
        seed: int = 1,
        platform: str = "CPU",
        max_included_literals: int | None = None,
        clause_drop_p: float = 0.0,
        literal_drop_p: float = 0.0,
    ) -> None:
        self.n_clauses = n_clauses
        self.T = T
        self.s = s
        self.epochs = epochs
        self.weighted_clauses = weighted_clauses
        self.seed = seed
        self.platform = platform
        self.max_included_literals = max_included_literals
        self.clause_drop_p = clause_drop_p
        self.literal_drop_p = literal_drop_p

    def _new_tm(self):
        if self.n_clauses < 2 or self.n_clauses % 2:
            raise ValueError("n_clauses must be an even number >= 2 (half for, half against)")
        # TMU seeds a xorshift128+ generator with the seed; 0 gives an all-zero state that only
        # ever returns 0, and training hangs (issues-and-fixes E6).
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 1:
            raise ValueError(
                f"seed must be an integer >= 1 (TMU hangs with seed 0), got {self.seed!r}"
            )
        tm_class = _tm_classifier_class()
        return tm_class(
            number_of_clauses=self.n_clauses, T=self.T, s=self.s, platform=self.platform,
            weighted_clauses=self.weighted_clauses, seed=self.seed,
            max_included_literals=self.max_included_literals,
            clause_drop_p=self.clause_drop_p, literal_drop_p=self.literal_drop_p,
        )  # fmt: skip

    def _encode(self, y: object) -> np.ndarray:
        y = np.asarray(y)
        index = {c: i for i, c in enumerate(self.classes_)}
        unknown = sorted({str(v) for v in y} - set(map(str, index)))
        if unknown:
            raise ValueError(f"unknown label(s) {unknown}; known: {list(self.classes_)}")
        return np.asarray([index[v] for v in y], dtype=np.uint32)

    def _train_epochs(self, x: np.ndarray, y: object, epochs: int) -> TMLanguageClassifier:
        codes = self._encode(y)
        if len(codes) != len(x):
            raise ValueError("X and y have different lengths")
        for _ in range(epochs):
            self.tm_.fit(x, codes)
        self.epochs_trained_ += epochs
        self.state_ = TMState.from_tmu(self.tm_, [str(c) for c in self.classes_], x.shape[1],
                                       self._params_record())  # fmt: skip
        return self

    def _params_record(self) -> dict:
        return {**self.get_params(), "epochs_trained": self.epochs_trained_}

    def fit(self, x: object, y: object) -> TMLanguageClassifier:
        x = _binary_matrix(x)
        self.classes_ = np.unique(np.asarray(y))
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        self.n_features_in_ = x.shape[1]
        self.tm_ = self._new_tm()
        self.epochs_trained_ = 0
        return self._train_epochs(x, y, self.epochs)

    def partial_fit(self, x: object, y: object, classes: object = None) -> TMLanguageClassifier:
        """One more epoch. First call: `classes` (or the labels in y) fixes the label set."""
        x = _binary_matrix(x)
        if not hasattr(self, "tm_"):
            if hasattr(self, "state_"):
                raise RuntimeError("model was unpickled or loaded; it can predict but not train")
            self.classes_ = np.unique(np.asarray(classes if classes is not None else y))
            self.n_features_in_ = x.shape[1]
            self.tm_ = self._new_tm()
            self.epochs_trained_ = 0
        elif x.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x.shape[1]}")
        return self._train_epochs(x, y, 1)

    def decision_function(self, x: object) -> np.ndarray:
        check_is_fitted(self, "state_")
        return self.state_.class_sums(x)

    def predict(self, x: object) -> np.ndarray:
        check_is_fitted(self, "state_")
        return self.classes_[np.argmax(self.state_.class_sums(x), axis=1)]

    def predict_tmu(self, x: object) -> np.ndarray:
        """Prediction by TMU itself (for comparison and latency); needs the live TMU object."""
        check_is_fitted(self, "tm_")
        return self.classes_[np.asarray(self.tm_.predict(_binary_matrix(x)))]

    def __getstate__(self) -> dict:
        state = dict(super().__getstate__())  # may be the live __dict__ (Python 3.11+): copy
        state.pop("tm_", None)  # C pointers inside; the NumPy state_ carries the model
        return state


# ---------------------------------------------------------------------- saved models


@dataclass
class TrainedModel:
    """A loaded model file: raw text in, language out, no TMU needed."""

    binarizer: Binarizer
    state: TMState
    meta: dict = field(default_factory=dict)

    @property
    def classes(self) -> tuple[str, ...]:
        return self.state.classes

    def class_sums(self, texts: Sequence[str]) -> np.ndarray:
        return self.state.class_sums(self.binarizer.transform(texts))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.state.predict(self.binarizer.transform(texts))


def model_dict(binarizer: Binarizer, state: TMState, meta: dict | None = None) -> dict:
    if len(binarizer.vocabulary_) != state.n_features:
        raise ValueError(
            f"binarizer has {len(binarizer.vocabulary_)} features, model {state.n_features}"
        )
    return {
        "format": MODEL_FORMAT,
        "literal_encoding": LITERAL_ENCODING,
        "classes": list(state.classes),
        "binarizer": binarizer.to_dict(),
        "tm": state.to_dict(),
        "meta": meta or {},
    }


def save_model(path: str | Path, binarizer: Binarizer, state: TMState,
               meta: dict | None = None) -> Path:  # fmt: skip
    """Write one self-contained model file (UTF-8 JSON, compact, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model_dict(binarizer, state, meta), ensure_ascii=False,
                      separators=(",", ":"))  # fmt: skip
    path.write_text(text + "\n", encoding="utf-8")
    return path


def save_pipeline(path: str | Path, pipe: object, meta: dict | None = None) -> Path:
    """Save a fitted `Pipeline([("binarize", Binarizer), ("model", TMLanguageClassifier)])`."""
    binarizer, model = pipe.steps[0][1], pipe.steps[-1][1]
    check_is_fitted(model, "state_")
    return save_model(path, binarizer, model.state_, meta)


def load_model(path: str | Path) -> TrainedModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path}: expected format {MODEL_FORMAT!r}, got {data.get('format')!r}")
    binarizer = Binarizer.from_dict(data["binarizer"])
    state = TMState.from_dict(data["tm"], data["classes"])
    if len(binarizer.vocabulary_) != state.n_features:
        raise ValueError(f"{path}: vocabulary and model disagree on the number of features")
    return TrainedModel(binarizer, state, data.get("meta", {}))
