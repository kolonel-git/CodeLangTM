import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from codelangtm.features import Binarizer
from codelangtm.model import (
    MODEL_FORMAT,
    TMLanguageClassifier,
    TMState,
    load_model,
    model_dict,
    save_model,
    save_pipeline,
)

# ------------------------------------------------------------ hand-built state (no TMU)
# 3 features (a, b, c) -> literal ids: a=0 b=1 c=2, NOT a=3 NOT b=4 NOT c=5.
# class "x": clause 0 = a AND NOT b (w +2), clause 1 = empty (w +5, never fires)
# class "y": clause 2 = b (w +1), clause 3 = c AND NOT a (w -1)


def hand_state():
    return TMState._from_rows(
        ("x", "y"), 3, [0, 0, 1, 1], [2, 5, 1, -1],
        [np.array([0, 4]), np.array([], dtype=int), np.array([1]), np.array([2, 3])], {"T": 3},
    )  # fmt: skip


X_HAND = np.array([
    [1, 0, 0],  # a: clause 0 fires -> x=2, y=0
    [0, 1, 0],  # b: clause 2 fires -> x=0, y=1
    [0, 0, 1],  # c, no a: clause 3 fires -> x=0, y=-1
    [1, 1, 1],  # clause 2 only -> x=0, y=1
    [0, 0, 0],  # nothing fires -> tie 0/0 -> first class
])  # fmt: skip


def test_clause_outputs_by_hand():
    fired = hand_state().clause_outputs(X_HAND)
    assert fired.tolist() == [
        [True, False, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [False, False, True, False],
        [False, False, False, False],
    ]  # the empty clause never fires, even on the all-zero row


def test_class_sums_predict_and_ties():
    state = hand_state()
    assert state.class_sums(X_HAND).tolist() == [[2, 0], [0, 1], [0, -1], [0, 1], [0, 0]]
    assert state.predict(X_HAND).tolist() == ["x", "y", "x", "y", "x"]  # ties -> first class


def test_include_matrix_and_counts():
    state = hand_state()
    assert state.literal_counts().tolist() == [2, 0, 1, 2]
    assert state.include_matrix().tolist() == [
        [1, 0, 0, 0, 1, 0], [0] * 6, [0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
    ]  # fmt: skip


def test_state_dict_round_trip_and_validation():
    state = hand_state()
    data = json.loads(json.dumps(state.to_dict()))
    assert data["clauses"]["x"] == {"weights": [2, 5], "literals": [[0, 4], []]}
    assert TMState.from_dict(data, ("x", "y")) == state
    bad = json.loads(json.dumps(data))
    bad["clauses"]["y"]["literals"][0] = [6]  # 2 * n_features is out of range
    with pytest.raises(ValueError, match="out of range"):
        TMState.from_dict(bad, ("x", "y"))
    bad["clauses"]["y"]["literals"] = [[1]]
    with pytest.raises(ValueError, match="differ in length"):
        TMState.from_dict(bad, ("x", "y"))


def test_state_input_checks():
    state = hand_state()
    with pytest.raises(ValueError, match="binary"):
        state.predict(np.array([[2, 0, 0]]))
    with pytest.raises(ValueError, match="2D"):
        state.predict(np.array([1, 0, 0]))
    with pytest.raises(ValueError, match="expected 3 features"):
        state.predict(np.array([[1, 0]]))


def test_state_pickle_drops_cache():
    state = hand_state()
    state.predict(X_HAND)
    assert state._cache and not pickle.loads(pickle.dumps(state))._cache
    assert pickle.loads(pickle.dumps(state)) == state


def fitted_binarizer(n=3):
    texts = ["def f(x):\n    return x", "fn main() {}", "SELECT a FROM t;"] * 2
    b = Binarizer(n_features=n, ngram_sizes=(2,)).fit(texts)
    return b, texts


def test_save_and_load_model(tmp_path):
    binarizer, texts = fitted_binarizer()
    state = hand_state()
    path = save_model(tmp_path / "m" / "tm.json", binarizer, state, {"dataset": "abc"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == MODEL_FORMAT and data["classes"] == ["x", "y"]
    assert "NOT has" in data["literal_encoding"] and data["meta"] == {"dataset": "abc"}
    loaded = load_model(path)
    assert loaded.state == state and loaded.binarizer.vocabulary_ == binarizer.vocabulary_
    assert loaded.predict(texts).tolist() == state.predict(binarizer.transform(texts)).tolist()
    assert loaded.classes == ("x", "y")


def test_load_rejects_wrong_format_and_mismatch(tmp_path):
    binarizer, _ = fitted_binarizer()
    data = model_dict(binarizer, hand_state())
    (tmp_path / "a.json").write_text(json.dumps({**data, "format": "other/1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected format"):
        load_model(tmp_path / "a.json")
    with pytest.raises(ValueError, match="binarizer has 4 features"):
        model_dict(fitted_binarizer(4)[0], hand_state())


def test_binarizer_dict_round_trip():
    binarizer, texts = fitted_binarizer(5)
    again = Binarizer.from_dict(json.loads(json.dumps(binarizer.to_dict())))
    assert again.get_params() == binarizer.get_params()
    assert np.array_equal(again.transform(texts), binarizer.transform(texts))


def test_params_validation_without_tmu():
    with pytest.raises(ValueError, match="even number"):
        TMLanguageClassifier(n_clauses=3).fit(np.eye(3, dtype=int), ["a", "b", "c"])
    for seed in (0, -1, 1.5, True):  # seed 0 hangs TMU (all-zero xorshift state)
        with pytest.raises(ValueError, match="seed must be an integer >= 1"):
            TMLanguageClassifier(seed=seed).fit(np.eye(3, dtype=int), ["a", "b", "c"])


def test_missing_tmu_gives_install_hint():
    code = (
        "import sys; sys.modules['tmu'] = None\n"
        "import numpy as np\n"
        "from codelangtm.model import TMLanguageClassifier\n"
        "try:\n"
        "    TMLanguageClassifier(n_clauses=4, epochs=1).fit(np.eye(2, dtype=int), ['a', 'b'])\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "uv sync --extra tm" in out.stdout


# ----------------------------------------------------------------- with TMU installed


def toy_data(n_per_class=40, n_features=24, seed=0):
    """3 classes; each has 4 signature features that are usually on, plus random noise."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for k, name in enumerate(("go", "python", "sql")):
        base = rng.random((n_per_class, n_features)) < 0.15
        base[:, 4 * k : 4 * k + 4] |= rng.random((n_per_class, 4)) < 0.85
        x.append(base)
        y += [name] * n_per_class
    return np.vstack(x).astype(np.uint32), np.asarray(y)


@pytest.fixture(scope="module")
def trained():
    pytest.importorskip("tmu")
    x, y = toy_data()
    return TMLanguageClassifier(n_clauses=20, T=10, s=3.0, epochs=8, seed=1).fit(x, y), x, y


def test_numpy_prediction_equals_tmu(trained):
    model, x, _ = trained
    probe = np.vstack([x, (np.random.default_rng(5).random((60, x.shape[1])) < 0.3)]).astype(
        np.uint32
    )
    _, tmu_sums = model.tm_.predict(probe, return_class_sums=True)
    assert np.array_equal(model.decision_function(probe), np.asarray(tmu_sums))
    assert np.array_equal(model.predict(probe), model.predict_tmu(probe))


def test_learns_the_toy_problem(trained):
    model, x, y = trained
    assert (model.predict(x) == y).mean() > 0.9
    assert model.state_.n_clauses == 3 * 20 and model.epochs_trained_ == 8
    assert model.state_.params["epochs_trained"] == 8 and model.state_.params["T"] == 10


def test_same_seed_same_model():
    pytest.importorskip("tmu")
    x, y = toy_data()
    a = TMLanguageClassifier(n_clauses=20, T=10, s=3.0, epochs=4, seed=7).fit(x, y)
    b = TMLanguageClassifier(n_clauses=20, T=10, s=3.0, epochs=4, seed=7).fit(x, y)
    assert a.state_ == b.state_


def test_partial_fit_continues_training(trained):
    pytest.importorskip("tmu")
    x, y = toy_data()
    model = TMLanguageClassifier(n_clauses=20, T=10, s=3.0, seed=1)
    model.partial_fit(x, y, classes=["go", "python", "sql"])
    first = model.state_
    model.partial_fit(x, y)
    assert model.epochs_trained_ == 2 and model.state_ != first
    # fit(epochs=2) from scratch with the same seed gives the same model as 2 x partial_fit
    again = TMLanguageClassifier(n_clauses=20, T=10, s=3.0, epochs=2, seed=1).fit(x, y)
    assert again.state_.to_dict()["clauses"] == model.state_.to_dict()["clauses"]
    with pytest.raises(ValueError, match="expected 24 features"):
        model.partial_fit(x[:, :5], y)


def test_unknown_labels_rejected():
    pytest.importorskip("tmu")
    x, y = toy_data()
    model = TMLanguageClassifier(n_clauses=4, epochs=1)
    model.partial_fit(x, y, classes=["go", "python", "sql"])
    with pytest.raises(ValueError, match="unknown label"):
        model.partial_fit(x[:2], ["go", "rust"])


def test_pickle_keeps_prediction_and_original_stays_trainable(trained):
    model, x, y = trained
    blob = pickle.dumps(model)
    assert hasattr(model, "tm_")  # regression: pickling must not strip the live model
    copy = pickle.loads(blob)
    assert np.array_equal(copy.predict(x), model.predict(x))
    with pytest.raises(RuntimeError, match="can predict but not train"):
        copy.partial_fit(x, y)


def test_sklearn_pipeline_clone_and_save(tmp_path):
    pytest.importorskip("tmu")
    texts = (["def f(x):\n    return x + 1"] * 8 + ["func main() {\n\tfmt.Println(1)\n}"] * 8
             + ["SELECT a, b FROM t WHERE a = 1;"] * 8)  # fmt: skip
    labels = ["python"] * 8 + ["go"] * 8 + ["sql"] * 8
    template = TMLanguageClassifier(n_clauses=10, T=5, s=3.0, epochs=5, seed=3)
    assert clone(template).get_params() == template.get_params()
    pipe = Pipeline([("binarize", Binarizer(n_features=30, ngram_sizes=(2,))), ("model", template)])
    pipe.fit(texts, labels)
    assert (pipe.predict(texts) == np.asarray(labels)).mean() > 0.9
    loaded = load_model(save_pipeline(tmp_path / "tm.json", pipe, {"k": 1}))
    assert loaded.predict(texts).tolist() == pipe.predict(texts).tolist()
    assert loaded.meta == {"k": 1}


def test_tmu_import_is_quiet():
    pytest.importorskip("tmu")
    code = "from codelangtm.model import _tm_classifier_class; _tm_classifier_class()"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "pycuda" not in out.stderr and "Traceback" not in out.stderr
