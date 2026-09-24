import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from codelangtm.features import DELIMITERS, Binarizer, expand_literals

SNIPPETS = [
    "def f(x):\n    return x + 1\n",
    "int main() { return 0; }\n",
    "fn main() -> i32 {\n    let v = vec![1];\n    0\n}\n",
    "SELECT a FROM t WHERE b = 1;\n",
    "<div class=\"x\">\n\t<p>hi</p>\n</div>\n",
]


def test_matches_substring_semantics():
    b = Binarizer(n_features=120).fit(SNIPPETS)
    x = b.transform(SNIPPETS)
    expected = np.array([[tok in s for tok in b.vocabulary_] for s in SNIPPETS], dtype=np.uint32)
    assert np.array_equal(x, expected)
    assert x.dtype == np.uint32 and x.shape == (5, 120)


def test_vocabulary_independent_of_input_order():
    a = Binarizer(n_features=80).fit(SNIPPETS).vocabulary_
    b = Binarizer(n_features=80).fit(SNIPPETS[::-1]).vocabulary_
    assert a == b


def test_delimiters_first_and_optional():
    b = Binarizer(n_features=60).fit(SNIPPETS)
    assert b.vocabulary_[: len(DELIMITERS)] == list(DELIMITERS)
    no_delim = Binarizer(n_features=60, use_delimiters=False).fit(SNIPPETS)
    assert len(no_delim.vocabulary_) == 60
    assert "\t" not in no_delim.vocabulary_ and "    " not in no_delim.vocabulary_


def test_n_features_caps_vocabulary():
    assert len(Binarizer(n_features=5).fit(SNIPPETS).vocabulary_) == 5


def test_save_load_roundtrip(tmp_path):
    b = Binarizer(n_features=70, ngram_sizes=(2, 3, 4)).fit(SNIPPETS)
    b.save(tmp_path / "vocab.json")
    loaded = Binarizer.load(tmp_path / "vocab.json")
    assert loaded.get_params() == b.get_params()
    assert np.array_equal(loaded.transform(SNIPPETS), b.transform(SNIPPETS))


def test_sklearn_compatible():
    b = Binarizer(n_features=40, ngram_sizes=(3,))
    c = clone(b)
    assert c.get_params() == b.get_params()
    assert not hasattr(c, "vocabulary_")
    assert list(b.fit(SNIPPETS).get_feature_names_out()) == b.vocabulary_


def test_not_fitted():
    with pytest.raises(NotFittedError):
        Binarizer().transform(["x"])


def test_empty_input():
    b = Binarizer(n_features=30).fit(SNIPPETS)
    assert b.transform([]).shape == (0, 30)


def test_expand_literals():
    x = np.array([[1, 0], [0, 0]], dtype=np.uint32)
    assert expand_literals(x).tolist() == [[1, 0, 0, 1], [0, 0, 1, 1]]
