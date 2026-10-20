"""Merged classes (v1.0 step 2): JavaScript + TypeScript as one class `js_ts`."""

import json
from dataclasses import replace

import numpy as np
import pytest

from codelangtm import baselines as bl
from codelangtm.audit import load_dataset
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.config import (
    BaselinesConfig,
    config_hash,
    load_ablation_config,
    load_baselines_config,
    load_tm_config,
    parse_ablations,
    parse_baselines,
    parse_tm,
)
from codelangtm.data import (
    MERGE_PRESETS,
    MergePlan,
    Snippet,
    apply_merge,
    merge_plan,
    merged_classes,
    parse_merge_spec,
    save_snippets,
)

FAST = ["naive_bayes", "logistic_regression"]


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def js_text(tag):
    return "".join(f"function f{tag}{j}(x) {{\n  return x + {j};\n}}\n" for j in range(10))


def ts_text(tag):
    return "".join(
        f"function f{tag}{j}(x: number): number {{\n  return x + {j};\n}}\n" for j in range(10)
    )


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """Python, JavaScript and TypeScript: 10 repos each, 3 snippets per repo."""
    root = tmp_path_factory.mktemp("merge")
    raw = root / "raw"
    raw.mkdir()
    items = []
    for lang, make, ext in (
        ("python", py_text, "py"), ("javascript", js_text, "js"), ("typescript", ts_text, "ts"),
    ):  # fmt: skip
        for r in range(10):
            for i in range(3):
                text = make(f"{lang}{r}x{i}")
                items.append(Snippet(text, lang, "github", f"{lang}-org/r{r}", "sha",
                                     f"src/f{i}.{ext}", "MIT", 1, text.count("\n")))  # fmt: skip
    save_snippets(items, raw / "all.jsonl")
    build_dataset([raw], root / "processed")
    kept = {s.language for s in load_dataset(root / "processed")["train"]}
    assert kept == {"python", "javascript", "typescript"}  # the label checks accepted all three
    return root / "processed"


# ------------------------------------------------------------------ the spec


def test_parse_merge_spec():
    assert parse_merge_spec(None) == () and parse_merge_spec("") == ()
    assert parse_merge_spec("none") == () and parse_merge_spec(" None ") == ()
    assert parse_merge_spec("js_ts") == ("js_ts",)
    assert parse_merge_spec("JS_TS, js_ts") == ("js_ts",)  # case-insensitive, no duplicates
    assert parse_merge_spec(["js_ts"]) == ("js_ts",)
    with pytest.raises(ValueError, match=r"unknown merge preset\(s\) \['c_cpp'\]"):
        parse_merge_spec("c_cpp")
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_merge_spec("none,js_ts")
    with pytest.raises(ValueError, match="expected preset names"):
        parse_merge_spec([1, 2])


def test_the_preset_class_name_is_not_a_language_name():
    from codelangtm import ALL_LANGUAGES

    label, members = MERGE_PRESETS["js_ts"]
    assert label == "js_ts" and label not in ALL_LANGUAGES
    assert members == ("javascript", "typescript")


# ------------------------------------------------------------------ the plan


def test_plan_applies_when_both_members_are_chosen():
    plan = merge_plan("js_ts", ("python", "javascript", "typescript"))
    assert plan.applied == {"javascript": "js_ts", "typescript": "js_ts"} and plan.notes == ()
    assert plan.presets == ("js_ts",)
    assert merged_classes(("python", "javascript", "typescript", "go"), plan) == (
        "python", "js_ts", "go",
    )  # fmt: skip


def test_plan_is_empty_without_a_request():
    plan = merge_plan(None, ("javascript", "typescript"))
    assert plan == MergePlan() and plan.applied == {}
    assert merged_classes(("python", "go"), plan) == ("python", "go")


def test_a_preset_with_only_one_member_is_not_applied_and_says_so():
    plan = merge_plan("js_ts", ("python", "javascript"))  # e.g. `--languages core`
    assert plan.applied == {}
    assert plan.presets == ("js_ts",) and len(plan.notes) == 1
    assert "not applied" in plan.notes[0] and "javascript" in plan.notes[0]
    none_chosen = merge_plan("js_ts", ("python",))
    assert none_chosen.applied == {} and "none of" in none_chosen.notes[0]


def test_plan_meta_is_json_ready():
    meta = merge_plan("js_ts", ("javascript", "typescript")).as_meta()
    assert meta == {"presets": ["js_ts"],
                    "applied": {"javascript": "js_ts", "typescript": "js_ts"},
                    "notes": []}  # fmt: skip
    json.dumps(meta)


def test_apply_merge_relabels_every_split_and_leaves_the_input_alone():
    def s(lang, i):
        return Snippet(f"x{i}\n", lang, "github", f"r/{lang}{i}", "c", "a.js", "MIT", 1, 1)

    dataset = {
        "train": [s("javascript", 0), s("python", 1), s("typescript", 2)],
        "test": [s("typescript", 3)],
        "wild": [],
    }
    plan = merge_plan("js_ts", ("python", "javascript", "typescript"))
    merged = apply_merge(dataset, plan)
    assert [x.language for x in merged["train"]] == ["js_ts", "python", "js_ts"]
    assert [x.language for x in merged["test"]] == ["js_ts"] and merged["wild"] == []
    assert [x.language for x in dataset["train"]] == ["javascript", "python", "typescript"]
    assert merged["train"][0].repo == "r/javascript0" and merged["train"][0].text == "x0\n"
    assert apply_merge(dataset, MergePlan()) is dataset  # nothing to do


# ------------------------------------------------------------------ loading


def test_view_without_a_merge_is_the_old_behaviour(processed):
    view = bl.load_view(processed)
    assert view.classes == view.languages == ("python", "javascript", "typescript")
    assert view.meta() == {"languages": ["python", "javascript", "typescript"]}  # no new keys
    dataset, folds, langs = bl.load_evaluation_data(processed)
    assert langs == view.classes and len(folds) == len(dataset["train"])


def test_view_with_a_merge_has_fewer_classes_and_the_same_folds(processed):
    plain = bl.load_view(processed)
    merged = bl.load_view(processed, merge="js_ts")
    assert merged.languages == ("python", "javascript", "typescript")
    assert merged.classes == ("python", "js_ts")
    assert merged.folds.tolist() == plain.folds.tolist()  # only the label names change
    assert [s.repo for s in merged.dataset["train"]] == [s.repo for s in plain.dataset["train"]]
    labels = {s.language for part in merged.dataset.values() for s in part}
    assert labels == {"python", "js_ts"}
    meta = merged.meta()
    assert meta["languages"] == ["python", "js_ts"]
    assert meta["source_languages"] == ["python", "javascript", "typescript"]
    assert meta["merge"]["applied"] == {"javascript": "js_ts", "typescript": "js_ts"}


def test_merge_with_a_language_subset_where_the_pair_is_incomplete(processed):
    view = bl.load_view(processed, "python,javascript", "js_ts")
    assert view.classes == ("python", "javascript")
    merge = view.meta()["merge"]
    assert merge["applied"] == {} and "not applied" in merge["notes"][0]


# ------------------------------------------------------------------ baselines


def test_run_baselines_trains_and_scores_the_merged_class(processed):
    results, meta = bl.run_baselines(processed, FAST, n_features=60, repeats=1, merge="js_ts")
    assert meta["languages"] == ["python", "js_ts"]
    assert meta["merge"]["applied"]["typescript"] == "js_ts"
    for r in results:
        assert set(r.per_language_f1) == {"python", "js_ts"}  # the labels reached the classifier
        assert np.array(r.confusion).shape == (2, 2)
        assert np.array(r.confusion).sum() == meta["n_test"]
    md = bl.render_results_md(results, meta)
    assert "- Languages (2): python, js_ts" in md
    assert "- Merged classes (preset js_ts): `js_ts` = javascript + typescript" in md
    doc = bl.results_json(results, meta)
    assert doc["languages"] == ["python", "js_ts"] and doc["meta"]["merge"]["presets"] == ["js_ts"]


def test_the_merge_note_is_in_the_report_when_it_could_not_apply(processed):
    results, meta = bl.run_baselines(processed, FAST, n_features=60, repeats=0,
                                     languages="python,javascript", merge="js_ts")  # fmt: skip
    md = bl.render_results_md(results, meta)
    assert "- Merged classes (preset js_ts): none applied" in md
    assert "- Merge note: js_ts: not applied" in md


def test_without_a_merge_the_report_is_unchanged(processed):
    results, meta = bl.run_baselines(processed, FAST, n_features=60, repeats=0)
    assert "merge" not in meta and "source_languages" not in meta
    assert "Merged classes" not in bl.render_results_md(results, meta)


# ------------------------------------------------------------------ configs


def test_config_merge_key_and_default():
    assert BaselinesConfig().merge == "none"
    assert parse_baselines({"merge": "js_ts"}).merge == ("js_ts",)
    assert parse_baselines({"merge": ["js_ts"]}).merge == ("js_ts",)
    assert parse_baselines({"merge": "none"}).merge == "none"
    for bad in ("c_cpp", 5, ["js_ts", 3], "none,js_ts"):
        with pytest.raises(ValueError, match="merge"):
            parse_baselines({"merge": bad})
    ab = parse_ablations({"merge": "js_ts", "studies": {"m": {"n_features": [10]}}})
    assert ab.merge == ("js_ts",)
    tm = parse_tm({"merge": "js_ts", "languages": "core", "tm": {"a": {"n_clauses": 4}}})
    assert tm.merge == ("js_ts",) and tm.languages == "core"


def test_cli_overrides_win_over_the_config():
    cfg = parse_baselines({"merge": "js_ts"})
    assert cfg.with_overrides(merge=None).merge == ("js_ts",)
    assert cfg.with_overrides(merge="none").merge == "none"
    with pytest.raises(ValueError, match="merge"):
        cfg.with_overrides(merge="nope")
    tm = parse_tm({"tm": {"a": {"n_clauses": 4}}})
    assert tm.with_overrides(languages="all", merge="js_ts").merge == ("js_ts",)
    assert tm.with_overrides(languages="all").languages == "all"
    assert tm.with_overrides() == tm


def test_config_hashes_of_older_runs_stay_valid_without_a_merge():
    cfg = BaselinesConfig()
    assert config_hash(cfg) == config_hash(replace(cfg, merge="none"))
    assert config_hash(replace(cfg, merge=("js_ts",))) != config_hash(cfg)
    tm = parse_tm({"tm": {"a": {"n_clauses": 4}}})
    assert "merge" not in tm.__dict__ or tm.merge == "none"
    assert config_hash(tm) != config_hash(tm.with_overrides(merge="js_ts"))
    assert config_hash(tm) != config_hash(tm.with_overrides(languages="core"))


def test_repo_yaml_configs_still_load_and_do_not_merge():
    assert load_baselines_config("configs/baselines.yaml").merge == "none"
    assert load_ablation_config("configs/ablations.yaml").merge == "none"
    assert load_tm_config("configs/tm.yaml").merge == "none"
    assert load_tm_config("configs/tm.yaml").languages == "auto"


# ------------------------------------------------------------------ commands


def test_cli_baselines_merge(processed, tmp_path, capsys):
    out = tmp_path / "results.md"
    args = ["baselines", "--data", str(processed), "--out", str(out), "--features", "60",
            "--repeats", "0", "--model", "naive_bayes"]  # fmt: skip
    assert main([*args, "--merge", "js_ts"]) == 0
    assert "languages (2): python, js_ts" in capsys.readouterr().out
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["languages"] == ["python", "js_ts"]
    assert side["meta"]["merge"]["presets"] == ["js_ts"]
    assert main([*args, "--merge", "none"]) == 0
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["languages"] == ["python", "javascript", "typescript"]
    assert "merge" not in side["meta"]


def test_cli_rejects_an_unknown_merge_preset(processed, tmp_path, capsys):
    out = tmp_path / "results.md"
    code = main(["baselines", "--data", str(processed), "--out", str(out), "--merge", "c_cpp"])
    assert code == 1 and not out.exists()
    assert "unknown merge preset" in capsys.readouterr().err


def test_cli_diagnose_takes_merge(processed, tmp_path):
    out = tmp_path / "diag.md"
    assert main(["diagnose", "--data", str(processed), "--out", str(out), "--features", "60",
                 "--merge", "js_ts"]) == 0  # fmt: skip
    text = out.read_text(encoding="utf-8")
    assert "2 languages (python, js_ts)" in text


def test_ablations_take_merge(processed):
    from codelangtm import ablations as ab

    cfg = parse_ablations({
        "data": processed.as_posix(), "models": ["naive_bayes"],
        "base": {"n_features": 40, "ngram_sizes": [2, 3], "use_delimiters": True},
        "studies": {"m": {"n_features": [20, 40]}},
        "merge": "js_ts",
    })
    runs, meta = ab.run_ablations(cfg)
    assert meta["languages"] == ["python", "js_ts"]
    assert set(next(iter(runs.values())).per_language_f1) == {"python", "js_ts"}
    assert "- Merged classes (preset js_ts)" in ab.render_ablations_md(cfg, runs, meta)
    runs, meta = ab.run_ablations(cfg, merge="none")  # an explicit argument wins
    assert meta["languages"] == ["python", "javascript", "typescript"]


# ------------------------------------------------------------------ TM commands (TMU)


@pytest.fixture(scope="module")
def tm_config(processed, tmp_path_factory):
    pytest.importorskip("tmu")
    root = tmp_path_factory.mktemp("tmcfg")
    config = root / "tm.yaml"
    config.write_text(
        f"data: {processed.as_posix()}\nseeds: [1]\nrepeats: 2\nmerge: js_ts\n"
        "baselines: [naive_bayes]\nfeatures: {n_features: 30}\n"
        "curves: {max_epochs: 3, smooth_window: 1}\n"
        "tm:\n  small: {n_clauses: 6, T: 3, epochs: 2}\n",
        encoding="utf-8",
    )
    return config


def test_tm_results_and_model_files_carry_the_merge(processed, tm_config, tmp_path):
    from codelangtm import tm_results as tr
    from codelangtm.model import load_model

    cfg = load_tm_config(tm_config)
    baselines, entries, meta = tr.run_tm_results(cfg, models_dir=tmp_path)
    assert meta["languages"] == ["python", "js_ts"] and meta["merge"]["presets"] == ["js_ts"]
    assert set(baselines[0].per_language_f1) == {"python", "js_ts"}
    model = load_model(tmp_path / "small_seed1.json")
    assert list(model.classes) == ["js_ts", "python"] or set(model.classes) == {"python", "js_ts"}
    assert model.meta["merge"] == ["js_ts"]
    assert model.meta["languages"] == ["python", "javascript", "typescript"]
    md = tr.render_tm_results_md(tr.tm_results_json(cfg, baselines, entries, meta))
    assert "- Merged classes (preset js_ts)" in md


def test_tm_train_and_clauses_follow_the_models_own_classes(processed, tm_config, tmp_path, capsys):
    out = tmp_path / "m.json"
    assert main(["tm-train", "--config", str(tm_config), "--out", str(out)]) == 0
    clauses_md = tmp_path / "clauses.md"
    code = main(["clauses", "--model", str(out), "--data", str(processed), "--models-dir",
                 str(tmp_path), "--no-formation", "--out", str(clauses_md)])  # fmt: skip
    assert code == 0, capsys.readouterr().err
    data = json.loads(clauses_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(data["languages"]) == {"python", "js_ts"}
    assert data["checks"]["rules_match_model"]


def test_tm_curve_and_errors_use_the_merged_classes(processed, tm_config, tmp_path, capsys):
    out = tmp_path / "curves.md"
    assert main(["tm-curve", "--config", str(tm_config), "--out", str(out)]) == 0
    curves = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert curves["meta"]["languages"] == ["python", "js_ts"]
    assert "- Merged classes (preset js_ts)" in out.read_text(encoding="utf-8")

    errors_md, review = tmp_path / "errors.md", tmp_path / "review.md"
    missing = tmp_path / "none.json"
    assert main(["errors", "--config", str(tm_config), "--setting", "small", "--tm-results",
                 str(missing), "--clauses", str(missing), "--model", str(missing), "--out",
                 str(errors_md), "--review", str(review)]) == 0  # fmt: skip
    capsys.readouterr()
    data = json.loads(errors_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(data["languages"]) == {"python", "js_ts"}  # the merged class is not dropped
    assert "javascript" not in data["languages"]


def test_cli_override_beats_the_tm_config(processed, tm_config, tmp_path):
    out = tmp_path / "curves.md"
    assert main(["tm-curve", "--config", str(tm_config), "--merge", "none", "--out",
                 str(out)]) == 0  # fmt: skip
    curves = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert curves["meta"]["languages"] == ["python", "javascript", "typescript"]
