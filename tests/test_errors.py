import json

import numpy as np
import pytest

from codelangtm import errors as er
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.data import Snippet, save_snippets

CLASSES = ["html", "javascript", "python"]
# id: (language, TM prediction for seeds 1 and 2, Naive Bayes prediction, features, text)
ROWS = [
    ("html", ["javascript", "javascript"], "html", 10,
     "<script>\nvar a = 1;\nvar b = 2;\n</script>\n"),
    ("html", ["html", "html"], "html", 40, "<div>\n<p>hi</p>\n</div>\n"),
    ("javascript", ["javascript", "javascript"], "javascript", 30, "const a = 1;\n"),
    ("javascript", ["python", "javascript"], "python", 20, "let b;\n"),
    ("python", ["python", "python"], "python", 50, "def f():\n  pass\n"),
    ("python", ["javascript", "python"], "python", 12, "x = 1\n"),
]  # fmt: skip
TEXTS = {i: r[4] for i, r in enumerate(ROWS)}


def votes(pred, true):
    """For-votes: 10 for the prediction, 5 for the true language if different, 2 elsewhere."""
    return [10 if c == pred else 5 if c == true else 2 for c in CLASSES]


def fake_oof():
    snippets, vf, va, rules = [], {"1": [], "2": []}, {"1": [], "2": []}, {}
    for i, (lang, tm, nb, feats, text) in enumerate(ROWS):
        snippets.append({"id": i, "language": lang, "repo": f"org/r{i}", "path": f"f{i}",
                         "lines": [1, 3], "n_lines": 3, "n_chars": len(text), "fold": i % 2,
                         "features_present": feats, "baselines": {"naive_bayes": nb},
                         "tm": {"1": tm[0], "2": tm[1]}})  # fmt: skip
        for k, seed in enumerate(("1", "2")):
            vf[seed].append(votes(tm[k], lang))
            va[seed].append([1, 1, 1])
            if tm[k] != lang:
                rules[f"{i}:{seed}"] = {
                    "predicted": [{"weight": 7, "rule": 'has("{ ")'}],
                    "true": [{"weight": 3, "rule": 'has("<d")'}],
                    "pulling_rules": ['has("{ ")', 'has("va") AND NOT has("<")'],
                    "pulling_ngrams": ['"{ "', '"va"'],
                }
    return {"setting": "tm_small", "params": {"n_clauses": 4}, "seeds": [1, 2],
            "baselines": ["naive_bayes"], "classes": CLASSES, "config_hash": "abc",
            "snippets": snippets, "tm_votes_for": vf, "tm_votes_against": va,
            "error_rules": rules,
            "fold_f1": {"baselines": {"naive_bayes": [0.9, 0.8]},
                        "tm": {"1": [0.7, 0.6], "2": [0.75, 0.65]}}}  # fmt: skip


# ------------------------------------------------------------------ building blocks


def test_embedded_share():
    assert er.embedded_share("<div>\n<script>\nvar a;\n</script>\n<p>") == 3 / 5
    assert er.embedded_share("var a;\nb();\n</script>\n<div>") == 3 / 4  # window starts inside
    assert er.embedded_share("<p>a</p>\n\n<p>b</p>") == 0.0
    assert er.embedded_share("<script>x()</script>\n<p>") == 1 / 2  # opened and closed inline
    assert er.embedded_share("<STYLE>\na{}\n</STYLE>") == 1.0
    assert er.embedded_share("\n  \n") == 0.0


def test_mcnemar_counts_discordant_pairs():
    t = er.mcnemar(list("aaaa"), list("abba"), list("aaab"))
    assert (t["only_a_wrong"], t["only_b_wrong"], t["both_wrong"]) == (2, 1, 0)
    assert t["p"] == pytest.approx(1.0)
    assert er.mcnemar(list("ab"), list("ab"), list("ab"))["p"] == 1.0  # nothing to compare
    t = er.mcnemar(["a"] * 10, ["b"] * 10, ["a"] * 10)
    assert t["p"] == pytest.approx(2 / 2**10)


def test_confusion_majority_and_seed_agreement():
    oof = fake_oof()
    assert er.confusion(["a", "b", "b"], ["a", "a", "b"], ["a", "b"]) == [[1, 0], [1, 1]]
    assert er.majority_vote(oof)[3] == "javascript"  # tie python/javascript: alphabetical
    sa = er.seed_agreement(oof)
    assert sa["histogram"] == {"0": 3, "1": 2, "2": 1} and sa["hard"] == [0]
    assert sa["sometimes"] == [3, 5]


def test_inflow_and_top_receiver():
    oof = fake_oof()
    flows = er.inflow(oof)
    assert flows["tm"]["javascript"] == {"count": 3, "share": 0.75}
    assert flows["tm"]["python"] == {"count": 1, "share": 0.25}
    assert flows["naive_bayes"]["python"] == {"count": 1, "share": 1.0}
    assert er.top_receiver(oof) == "javascript"


def test_confusable_pairs_scale_the_tm_per_seed():
    oof = fake_oof()
    ov = {a: {b: (0.3 if {a, b} == {"html", "javascript"} else 0.1) for b in CLASSES}
          for a in CLASSES}  # fmt: skip
    out = er.confusable_pairs(oof, CLASSES, ov)
    top = out["pairs"][0]
    assert top["pair"] == ["html", "javascript"]
    assert top["tm"] == {"html->javascript": 1.0, "javascript->html": 0.0, "total": 1.0}
    assert top["overlap"] == 0.3 and "spearman" in out["overlap_correlation"]


def test_consistency_check():
    oof = fake_oof()
    ref = {"baselines": [{"name": "naive_bayes", "cv_f1": [0.9, 0.8000004]}],
           "tm": {"tm_small": {"seeds": [{"seed": 1, "cv_f1": [0.7, 0.6]},
                                         {"seed": 2, "cv_f1": [0.75, 0.65]}]}}}  # fmt: skip
    c = er.consistency(oof, ref)
    assert c["matches"] and c["compared_scores"] == 6
    ref["tm"]["tm_small"]["seeds"][0]["cv_f1"] = [0.7, 0.61]
    assert not er.consistency(oof, ref)["matches"]
    assert er.consistency(oof, None) == {"checked": False}


def test_sink_probes():
    oof = fake_oof()
    sk = er.sink_probes(oof, "javascript")
    assert sk["name"] == "javascript" and (sk["errors"], sk["into_sink"]) == (4, 3)
    h1 = sk["h1_share_of_errors"]
    assert h1["chance"] == 0.5 and h1["shares"] == {"tm": 0.75, "naive_bayes": 0.0}
    assert h1["supported"]
    assert not er.sink_probes(oof, "python")["h1_share_of_errors"]["supported"]  # below chance
    h2 = sk["h2_little_evidence"]
    assert h2["features_median"] == [11, 45] and h2["lines_median"] == [3, 3]
    assert sk["h3_narrow_wins"]["margin_median"] == [5, 5]  # 10 - 5 for every error
    h4 = sk["h4_weak_rejection"]
    assert h4["highest"] in CLASSES and set(h4["foreign_votes"]) == set(CLASSES)
    # equal counts: ordered by text, so the report is identical on every run
    assert sk["h5_pulling"]["rules"] == [['has("va") AND NOT has("<")', 3], ['has("{ ")', 3]]
    assert sk["h5_pulling"]["ngrams"] == [['"va"', 3], ['"{ "', 3]]


def test_embedded_probe():
    em = er.embedded_probe(fake_oof(), TEXTS)
    assert (em["snippets"], em["to_guest"], em["always_right"]) == (2, 1, 1)
    assert em["embedded_median"] == [1.0, 0.0]
    assert em["mostly_embedded"] == {"snippets": 1, "to_guest_rate": 1.0}
    assert em["rest"] == {"snippets": 1, "to_guest_rate": 0.0}
    assert em["points"] == [[1.0, 2], [0.0, 0]]


def test_signature_document_frequency():
    class S:
        def __init__(self, text, language):
            self.text, self.language = text, language

    train = [S("fn a", "rust"), S("fn b", "rust"), S("def", "python"), S("fn", "python")]
    sigs = {"rust": [{"feature": '"fn"', "id": 0}]}
    out = er.signature_document_frequency(sigs, ["fn"], train)
    assert out == {"rust": [{"feature": '"fn"', "own_df": 1.0, "foreign_df": 0.5}]}


def test_analyse_and_reports():
    data = er.analyse(fake_oof(), texts=TEXTS)
    assert data["schema"] == er.ERRORS_SCHEMA and data["sink"]["name"] == "javascript"
    assert data["planned_sink"] == {"name": "python", "count": 1, "share": 0.25,
                                    "chance": 0.5, "replicated": False}  # fmt: skip
    assert data["models"]["tm"]["errors"] == 2.0  # per seed
    assert "pulling_rules" not in data["error_rules"]["0:1"]
    md = er.render_errors_md(data)
    assert md.startswith("# Errors: where the TM goes wrong, and why")
    assert "does not replicate here" in md and "## Why do snippets fall into javascript?" in md
    assert "| H7 |" in md and "not checked" in md
    review = er.render_review_md(data, TEXTS)
    assert "var a = 1;" in review and "(2/2 seeds wrong)" in review
    assert "const a = 1;" not in review  # always right: not in the review file
    assert "javascript" in er.summary_table(data)


# ------------------------------------------------------------------ with TMU


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j}\n}}\n" for j in range(8))
    return "package main\n\n" + body


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("tmu")
    root = tmp_path_factory.mktemp("errors")
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
    config.write_text(f"data: {(root / 'processed').as_posix()}\nseeds: [1]\n"
                      "baselines: [naive_bayes]\nfeatures: {n_features: 30}\n"
                      "tm:\n  small: {n_clauses: 6, T: 3, epochs: 2}\n",
                      encoding="utf-8")  # fmt: skip
    return root, config


def test_out_of_fold_scores_equal_evaluate_model(tiny):
    from codelangtm.audit import load_dataset
    from codelangtm.baselines import evaluate_model, load_folds, make_models
    from codelangtm.config import load_tm_config
    from codelangtm.model import TMLanguageClassifier

    root, config = tiny
    cfg = load_tm_config(config)
    oof = er.run_out_of_fold(cfg, "small")
    dataset = load_dataset(root / "processed")
    folds = load_folds(root / "processed")
    binarizer = cfg.features.binarizer()
    nb = evaluate_model("naive_bayes", make_models(0)["naive_bayes"], dataset, folds, binarizer)
    tm = evaluate_model("small", TMLanguageClassifier(**cfg.setting("small").kwargs(), seed=1),
                        dataset, folds, binarizer)  # fmt: skip
    assert oof["fold_f1"]["baselines"]["naive_bayes"] == nb.cv_f1
    assert oof["fold_f1"]["tm"]["1"] == tm.cv_f1
    assert len(oof["snippets"]) == len(dataset["train"])
    s = oof["snippets"][0]
    assert s["features_present"] > 0 and set(s["tm"]) == {"1"}
    sums = np.asarray(oof["tm_votes_for"]["1"]) - np.asarray(oof["tm_votes_against"]["1"])
    pred = [oof["classes"][k] for k in sums.argmax(axis=1)]
    assert pred == [x["tm"]["1"] for x in oof["snippets"]]  # votes explain every prediction


def test_cli_errors(tiny, tmp_path, capsys):
    root, config = tiny
    out, review = tmp_path / "errors.md", tmp_path / "review.md"
    missing = tmp_path / "none.json"
    assert main(["errors", "--config", str(config), "--setting", "small", "--tm-results",
                 str(missing), "--clauses", str(missing), "--model", str(missing), "--out",
                 str(out), "--review", str(review)]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "skipping the consistency check" in printed and "review file" in printed
    data = json.loads((tmp_path / "errors.json").read_text(encoding="utf-8"))
    assert data["schema"] == er.ERRORS_SCHEMA and data["n_snippets"] == len(data["snippets"])
    assert out.read_text(encoding="utf-8").startswith("# Errors")
    assert review.read_text(encoding="utf-8").startswith("# Error review (local, do not commit)")
    assert main(["errors", "--config", str(config), "--setting", "nope"]) == 1
    assert "errors failed" in capsys.readouterr().err
