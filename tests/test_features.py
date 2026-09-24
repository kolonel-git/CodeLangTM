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


# Each language has one distinctive bigram (xa / yb); every snippet shares generic filler, so
# frequency ranking prefers filler while label-aware selection prefers the distinctive bigrams.
LABELLED = [
    ("aaaa xa qq", "x"), ("bbbb xa qq", "x"), ("cccc xa qq", "x"),
    ("aaaa yb qq", "y"), ("bbbb yb qq", "y"), ("dddd yb qq", "y"),
]  # fmt: skip
TEXTS = [t for t, _ in LABELLED]
LABELS = [c for _, c in LABELLED]


def fit_labelled(**kw):
    return Binarizer(ngram_sizes=(2,), use_delimiters=False, **kw).fit(TEXTS, LABELS)


def test_frequency_prefers_common_ngrams():
    vocab = fit_labelled(n_features=3).vocabulary_
    assert "xa" not in vocab and "yb" not in vocab  # in 3 of 6 snippets; filler in 6 of 6


@pytest.mark.parametrize("selection", ["chi2", "class_balanced"])
def test_label_aware_selection_prefers_distinctive_ngrams(selection):
    vocab = fit_labelled(n_features=4, selection=selection).vocabulary_
    assert {"xa", "yb"} <= set(vocab)
    assert "qq" not in vocab  # in every snippet: says nothing about the language


def test_class_balanced_alternates_languages():
    vocab = fit_labelled(n_features=4, selection="class_balanced").vocabulary_
    # rank 0 of class x, rank 0 of class y, rank 1 of x, rank 1 of y
    assert vocab[:2] == [" x", " y"]  # alphabetical first among equally distinctive grams
    assert {"xa", "yb"} <= set(vocab)


@pytest.mark.parametrize("selection", ["chi2", "class_balanced"])
def test_label_aware_selection_order_independent(selection):
    a = fit_labelled(n_features=6, selection=selection).vocabulary_
    b = Binarizer(6, (2,), False, selection).fit(TEXTS[::-1], LABELS[::-1]).vocabulary_
    assert a == b


def test_label_aware_selection_needs_labels():
    with pytest.raises(ValueError, match="needs labels"):
        Binarizer(selection="chi2").fit(TEXTS)
    with pytest.raises(ValueError, match="one label per snippet"):
        Binarizer(selection="chi2").fit(TEXTS, LABELS[:2])


def test_invalid_options_rejected():
    with pytest.raises(ValueError, match="selection"):
        Binarizer(selection="random").fit(TEXTS)
    with pytest.raises(ValueError, match="min_df"):
        Binarizer(min_df=0).fit(TEXTS)


def test_min_df_drops_rare_candidates():
    vocab = fit_labelled(n_features=1000, min_df=3).vocabulary_
    assert "cc" not in vocab and "dd" not in vocab  # each in one snippet
    assert {"xa", "yb", "qq"} <= set(vocab)


def test_word_tokens_are_whole_words():
    b = Binarizer(n_features=200, ngram_sizes=(2,), use_delimiters=False, word_tokens=True)
    b.fit(["SELECT a FROM t;", "fn main() { let x = 1; }"])
    names = list(b.get_feature_names_out())
    assert "word:SELECT" in names and "word:fn" in names and "word:let" in names
    x = b.transform(["SELECT 1", "SELECTED 1", "select 1"])
    col = names.index("word:SELECT")
    assert x[:, col].tolist() == [1, 0, 0]  # whole token, case-sensitive
    assert all("\x00" not in n for n in names)


def test_word_features_never_collide_with_ngrams():
    b = Binarizer(n_features=500, ngram_sizes=(2, 3), use_delimiters=False, word_tokens=True)
    b.fit(["ab ab", "fn fn"])
    grams = [t for t in b.vocabulary_ if not t.startswith("\x00")]
    words = [t for t in b.vocabulary_ if t.startswith("\x00")]
    assert "ab" in grams and "\x00ab" in words  # same text, two different features
    x = b.transform(["xaby"])  # contains the bigram "ab" but not the word "ab"
    assert x[0, b.vocabulary_.index("ab")] == 1 and x[0, b.vocabulary_.index("\x00ab")] == 0


def test_save_load_roundtrip_with_options(tmp_path):
    b = Binarizer(60, (2, 3), True, "class_balanced", 2, True).fit(SNIPPETS * 2, list("abcde") * 2)
    b.save(tmp_path / "vocab.json")
    loaded = Binarizer.load(tmp_path / "vocab.json")
    assert loaded.get_params() == b.get_params() and loaded.vocabulary_ == b.vocabulary_
    assert np.array_equal(loaded.transform(SNIPPETS), b.transform(SNIPPETS))


def test_pipeline_passes_labels_to_binarizer():
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("b", fit_labelled(selection="chi2")), ("m", BernoulliNB())])
    pipe.fit(TEXTS, LABELS)  # would raise "needs labels" if y were not forwarded
    assert list(pipe.predict(["zz xa zz", "zz yb zz"])) == ["x", "y"]


def test_expand_literals():
    x = np.array([[1, 0], [0, 0]], dtype=np.uint32)
    assert expand_literals(x).tolist() == [[1, 0, 0, 1], [0, 0, 1, 1]]
