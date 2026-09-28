import io
import json
import sys

import numpy as np
import pytest

from codelangtm import rules as rl
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.data import Snippet, save_snippets
from codelangtm.features import WORD_PREFIX, Binarizer
from codelangtm.model import TMState, TrainedModel, save_model

# A hand-built model with known answers. Features: 0 "fn", 1 "::", 2 ";\n", 3 "de";
# literal j < 4 is has(feature j), 4 + j is NOT has(feature j).
VOCAB = ["fn", "::", ";\n", "de"]
CLAUSES = [  # (class, weight, literals)
    (0, 3, [3]),  # python for: has("de")
    (0, 2, [3, 4]),  # python for: has("de") AND NOT has("fn")
    (0, -2, [0]),  # python against: has("fn")
    (0, 1, []),  # python: empty, never fires
    (1, 4, [0]),  # rust for: has("fn")
    (1, 1, [0, 7]),  # rust for: has("fn") AND NOT has("de")
    (1, -1, [3]),  # rust against: has("de")
    (1, 2, [0]),  # rust for: duplicate of clause 4
]
TEXTS = ["def f():\n    pass\n", "fn main() {}\n", "fn decode() {}\n"]
LABELS = ["python", "rust", "rust"]


def hand_state():
    return TMState._from_rows(("python", "rust"), 4, [c for c, _, _ in CLAUSES],
                              [w for _, w, _ in CLAUSES],
                              [np.asarray(lits, dtype=np.int32) for *_, lits in CLAUSES],
                              {"n_clauses": 4, "T": 2, "s": 3.0, "epochs_trained": 1})  # fmt: skip


def hand_model():
    fitted = Binarizer(n_features=4).fit(TEXTS)
    binarizer = Binarizer.from_dict({**fitted.to_dict(), "vocabulary": VOCAB})
    return TrainedModel(binarizer, hand_state(), {"setting": "hand", "seed": 1})


# ------------------------------------------------------------------ naming


def test_show_term_makes_whitespace_and_quotes_visible():
    assert rl.show_term("{\n\t") == '"{\\n\\t"'
    assert rl.show_term('a"\\') == '"a\\"\\\\"'
    assert rl.show_term("\r\x01\x7f") == '"\\r\\x01\\x7f"'
    assert rl.show_term(WORD_PREFIX + "SELECT") == 'word("SELECT")'


def test_literal_text():
    vocab = ["fn ", WORD_PREFIX + "def"]
    assert rl.literal_text(0, vocab) == 'has("fn ")'
    assert rl.literal_text(2, vocab) == 'NOT has("fn ")'
    assert rl.literal_text(1, vocab) == 'word("def")'
    assert rl.literal_text(3, vocab) == 'NOT word("def")'
    with pytest.raises(ValueError, match="out of range"):
        rl.literal_text(4, vocab)


def test_format_rule():
    assert rl.format_rule("rust", ["fn "], ["\t}"]) == 'rust = has("fn ") AND NOT has("\\t}")'
    assert "never fires" in rl.format_rule("go", [], [])


def test_short_rule():
    rule = " AND ".join(f'NOT has("{c}")' for c in "abcdefgh")
    assert rl.short_rule(rule) == " AND ".join(f'NOT has("{c}")' for c in "abcdef") + (
        " AND ... (+2 more)"
    )
    assert rl.short_rule('has("fn ")') == 'has("fn ")'


def test_code_span_survives_tables_and_backticks():
    assert rl._code('has("|")') == '`has("\\|")`'
    assert rl._code('has("`")') == '``has("`")``'
    assert rl._code("`x") == "`` `x ``"


# ------------------------------------------------------------------ rules and statistics


def test_extract_rules_keeps_every_clause_in_model_order():
    rules = rl.extract_rules(hand_state(), VOCAB)
    assert [r.index for r in rules] == list(range(len(CLAUSES)))
    assert rules[1].text == 'has("de") AND NOT has("fn")'
    assert rules[3].text == "(empty clause: never fires)" and rules[3].literals == []
    assert [r.polarity for r in rules[:3]] == ["for", "for", "against"]
    assert rules[5].language == "rust" and rules[5].text == 'has("fn") AND NOT has("de")'
    with pytest.raises(ValueError, match="vocabulary"):
        rl.extract_rules(hand_state(), VOCAB[:3])


def test_rules_evaluated_from_text_equal_the_model():
    model = hand_model()
    x = model.binarizer.transform(TEXTS)
    assert x.tolist() == [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]]
    rules = rl.extract_rules(model.state, VOCAB)
    fired = model.state.clause_outputs(x)
    for r in rules:
        assert np.array_equal(rl.evaluate_rule(r, x), fired[:, r.index])
    assert rl.rules_match_model(rules, model.state, x)
    rules[0].literals = [0]  # a wrong rule is caught
    assert not rl.rules_match_model(rules, model.state, x)


def test_clause_stats_and_derived_rates():
    model = hand_model()
    rules = rl.extract_rules(model.state, VOCAB)
    rl.clause_stats(rules, model.state, model.binarizer.transform(TEXTS), LABELS)
    r = rules[0]  # has("de"): fires on text 0 (python) and text 2 (rust)
    assert (r.fires_own, r.fires_other, r.own_total, r.other_total) == (1, 1, 1, 2)
    assert (r.coverage, r.false_fire_rate, r.precision) == (1.0, 0.5, 0.5)
    r = rules[4]  # has("fn"): fires on both rust texts only
    assert (r.coverage, r.false_fire_rate, r.precision) == (1.0, 0.0, 1.0)
    assert rules[3].precision == 0.0  # empty clause: never fires, no division by zero
    d = rules[4].to_dict()
    assert d["rule"] == 'has("fn")' and d["polarity"] == "for" and d["n_literals"] == 1


def test_language_summary():
    model = hand_model()
    rules = rl.extract_rules(model.state, VOCAB)
    rl.clause_stats(rules, model.state, model.binarizer.transform(TEXTS), LABELS)
    s = rl.language_summary(rules, ["python", "rust"])
    assert s["python"] | {"mean_literals": 0} == {
        "clauses": 4, "for": 3, "against": 1, "zero_weight": 0, "empty": 1, "mean_literals": 0,
        "median_literals": 1, "distinct_literals": 3, "total_weight_for": 6,
        "total_weight_against": 2, "duplicate_clauses": 0, "narrow_clauses": 1,
        "max_literals": 2,
    }  # fmt: skip
    assert s["python"]["mean_literals"] == pytest.approx(4 / 3)
    assert s["rust"]["duplicate_clauses"] == 1  # clause 7 repeats clause 4
    assert s["rust"]["narrow_clauses"] == 1  # has("fn") AND NOT has("de") fires once


def test_signatures_overlap_and_top_clauses():
    model = hand_model()
    rules = rl.extract_rules(model.state, VOCAB)
    rl.clause_stats(rules, model.state, model.binarizer.transform(TEXTS), LABELS)
    langs = ["python", "rust"]
    sigs = rl.signature_features(rules, langs, VOCAB)
    assert [f["feature"] for f in sigs["python"]] == ['"de"']
    assert [f["feature"] for f in sigs["rust"]] == ['"fn"']
    assert sigs["rust"][0]["share_of_for_clauses"] == 1.0
    assert sigs["rust"][0]["lift"] == pytest.approx(1.01 / 0.01)
    assert rl.class_overlap(rules, langs, 4).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert [r.index for r in rl.top_clauses(rules, "rust")] == [4, 7, 5]
    sm = rl.signature_matrix(rules, sigs, langs, 4)
    assert sm["features"] == ['"de"', '"fn"'] and sm["usage"] == [[1.0, 0.0], [0.0, 1.0]]


def test_seed_stability():
    sigs = {"go": [{"feature": '":="'}, {"feature": '"{\\n\\t"'}]}
    other = {"seed 2": {"go": [{"feature": '":="'}, {"feature": '"fu"'}]}}
    scores = {"go": {"a": 1.0, "b": 0.0, "c": 0.5}}
    st = rl.seed_stability(sigs, other, scores, {"seed 2": {"go": {"a": 2.0, "b": 0.0, "c": 1.0,
                                                                   "d": 9.0}}})  # fmt: skip
    assert st["go"]["per_seed"] == {"seed 2": 0.5} and st["go"]["mean"] == 0.5
    assert st["go"]["mean_correlation"] == pytest.approx(1.0)  # only shared terms compared
    assert "score_correlation" not in rl.seed_stability(sigs, other)["go"]


def test_settled_epoch():
    assert rl.settled_epoch([0.1, 0.2, 0.05, 0.2]) == 4
    assert rl.settled_epoch([0.3, 0.3]) == 1
    assert rl.settled_epoch([0.2, 0.0]) is None
    assert rl.settled_epoch([]) is None


# ------------------------------------------------------------------ explain


def test_explanation_sums_equal_the_model():
    model = hand_model()
    for text in TEXTS:
        e = rl.explain_snippet(model, text)
        sums = model.class_sums([text])[0]
        assert e.class_sums == dict(zip(model.classes, sums.tolist(), strict=True))
        for lang in model.classes:
            assert sum(r.weight for r in e.fired if r.language == lang) == e.class_sums[lang]
        assert e.predicted == model.predict([text])[0]
    e = rl.explain_snippet(model, "fn decode() {}\n")
    assert e.predicted == "rust" and e.class_sums == {"python": 1, "rust": 5}
    assert [r.index for r in e.contributions("rust")] == [4, 7, 6]
    assert e.features_present == ['"fn"', '"de"']
    text = rl.format_explanation(e)
    assert text.startswith("prediction: rust") and "+4  #4" in text
    assert "(+6 for, -1 against)" in text


# ------------------------------------------------------------------ inspection and reports


def test_inspect_model_json_and_markdown():
    model = hand_model()
    data = rl.inspect_model(model, TEXTS, LABELS, {"seed 2": hand_model()}, formation=False)
    assert data["schema"] == rl.CLAUSES_SCHEMA and data["checks"]["rules_match_model"]
    assert [c["index"] for c in data["clauses"]] == list(range(len(CLAUSES)))  # every clause
    assert data["stability"]["rust"]["mean"] == 1.0  # a model agrees with itself
    assert data["formation"] is None
    text = rl.clauses_json_text(data)
    assert json.loads(text)["clauses"] == json.loads(json.dumps(data["clauses"]))
    clause_lines = [line for line in text.splitlines() if line.strip().startswith('{"index"')]
    assert len(clause_lines) == len(CLAUSES)  # one clause per line
    assert '"overlap": [\n    [1.0, 0.0],' in text  # number lists on one line
    md = rl.render_clauses_md(data)
    assert md.startswith("# Clauses: what the Tsetlin Machine learned")
    assert "### python" in md and "### rust" in md and "**yes**" in md
    assert '| 4 | 4 | 100% | 0.0% | 100% | `has("fn")` |' in md
    assert "How the rules formed" not in md
    assert "rust" in rl.summary_table(data)


# ------------------------------------------------------------------ with TMU


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j}\n}}\n" for j in range(8))
    return "package main\n\n" + body


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("tmu")
    from codelangtm.config import load_tm_config
    from codelangtm.tm_results import train_final

    root = tmp_path_factory.mktemp("rules")
    raw = root / "raw"
    raw.mkdir()
    items = []
    for lang, make, ext in (("python", py_text, "py"), ("go", go_text, "go")):
        for r in range(10):
            for i in range(3):
                text = make(f"{lang}{r}x{i}")
                items.append(Snippet(text, lang, "github", f"{lang}-org/r{r}", "sha",
                                     f"src/f{i}.{ext}", "MIT", 1, text.count("\n")))  # fmt: skip
    save_snippets(items, raw / "all.jsonl")
    build_dataset([raw], root / "processed", languages=("python", "go"))
    config = root / "tm.yaml"
    config.write_text(f"data: {(root / 'processed').as_posix()}\nseeds: [1, 2]\n"
                      "baselines: [naive_bayes]\nfeatures: {n_features: 30}\n"
                      "tm:\n  small: {n_clauses: 10, T: 5, s: 3.0, epochs: 4}\n",
                      encoding="utf-8")  # fmt: skip
    cfg = load_tm_config(config)
    models = root / "models"
    for seed in (1, 2):
        train_final(cfg, "small", seed, models / f"small_seed{seed}.json")
    return root, models


def test_formation_replay_reproduces_the_saved_model(trained):
    from codelangtm.audit import load_dataset
    from codelangtm.model import load_model

    root, models = trained
    model = load_model(models / "small_seed1.json")
    train = load_dataset(root / "processed")["train"]
    texts, labels = [s.text for s in train], [s.language for s in train]
    rules = rl.extract_rules(model.state, model.binarizer.vocabulary_)
    sigs = rl.signature_features(rules, list(model.classes), model.binarizer.vocabulary_)
    fm = rl.formation_replay(model, texts, labels, sigs)
    assert fm["reproduces_saved_model"] and fm["epochs"] == 4
    assert len(fm["jaccard_with_final"]) == 4 and fm["jaccard_with_final"][-1] == 1.0
    lang = model.classes[0]
    feature = sigs[lang][0]["feature"]
    assert len(fm["signature_share"][lang][feature]) == 4
    assert fm["settled_epoch"][lang][feature] in range(1, 5)


def test_cli_clauses_and_explain(trained, tmp_path, capsys, monkeypatch):
    root, models = trained
    out = tmp_path / "clauses.md"
    args = ["clauses", "--model", str(models / "small_seed1.json"), "--data",
            str(root / "processed"), "--models-dir", str(models), "--out", str(out)]  # fmt: skip
    assert main(args) == 0
    data = json.loads((tmp_path / "clauses.json").read_text(encoding="utf-8"))
    assert data["schema"] == rl.CLAUSES_SCHEMA and len(data["clauses"]) == 20  # 10 x 2 languages
    assert list(data["stability"]["go"]["per_seed"]) == ["seed 2"]  # other seeds, not itself
    assert data["formation"]["reproduces_saved_model"] and data["checks"]["rules_match_model"]
    assert "How the rules formed" in out.read_text(encoding="utf-8")

    snippet = tmp_path / "s.go"
    snippet.write_text(go_text("zz"), encoding="utf-8")
    capsys.readouterr()
    assert main(["explain", "--model", str(models / "small_seed1.json"), "--file",
                 str(snippet)]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert printed.startswith("prediction: go") and "votes per language" in printed
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(py_text("q").encode())))
    assert main(["explain", "--model", str(models / "small_seed1.json"), "--file", "-"]) == 0
    assert capsys.readouterr().out.startswith("prediction: python")


def test_cli_clauses_refuses_another_dataset(trained, tmp_path, capsys):
    root, models = trained
    from codelangtm.model import load_model

    model = load_model(models / "small_seed1.json")
    meta = {**model.meta, "dataset_files": {"train.jsonl": "000000000000"}}
    path = save_model(tmp_path / "m.json", model.binarizer, model.state, meta)
    assert main(["clauses", "--model", str(path), "--data", str(root / "processed"),
                 "--no-formation", "--out", str(tmp_path / "c.md")]) == 1  # fmt: skip
    assert "not the dataset" in capsys.readouterr().err


def test_cli_missing_model(tmp_path, capsys):
    assert main(["clauses", "--model", str(tmp_path / "none.json")]) == 1
    assert "clauses failed" in capsys.readouterr().err
    assert main(["explain", "--model", str(tmp_path / "none.json"), "--file", "-"]) == 1
