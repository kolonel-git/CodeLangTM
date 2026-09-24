"""Multi-class Tsetlin Machine wrapper (TMU backend). Phase 2."""

from __future__ import annotations

import numpy as np


class TMLanguageClassifier:
    """Thin wrapper over tmu.models.classification.vanilla_classifier.TMClassifier."""

    def __init__(
        self, n_clauses: int = 100, threshold: int = 30, s: float = 3.5, epochs: int = 50
    ) -> None:
        self.n_clauses = n_clauses
        self.threshold = threshold
        self.s = s
        self.epochs = epochs
        self._tm = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> TMLanguageClassifier:
        try:
            from tmu.models.classification.vanilla_classifier import TMClassifier
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install TMU: `uv sync --extra tm`") from e
        self._tm = TMClassifier(self.n_clauses, self.threshold, self.s)
        for _ in range(self.epochs):
            self._tm.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._tm is None:
            raise RuntimeError("Model is not fitted")
        return self._tm.predict(x)
