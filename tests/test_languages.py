"""Language handling for experiments (Stage B, B-S4): auto | core | all | explicit list."""

import json
from dataclasses import replace

import numpy as np
import pytest

from codelangtm import ALL_LANGUAGES, CORE_LANGUAGES, LANGUAGES
from codelangtm import baselines as bl
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.config import (
    BaselinesConfig,
    config_dict,
    config_hash,
    parse_ablations,
    parse_baselines,
)
from codelangtm.data import (
    Snippet,
    parse_language_spec,
    resolve_languages,
    restrict_dataset,
    save_snippets,
)

FAST = ["naive_bayes", "logistic_regression"]


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j}\n}}\n" for j in range(8))
    return "package main\n\n" + body


def kt_text(tag):
    body = "".join(f"fun f{tag}{j}(x: Int): Int {{\n    return x + {j}\n}}\n" for j in range(9))
    return "package app\n\n" + body


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """Three languages: two core ones and one stretch language (Kotlin)."""
    root = tmp_path_factory.mktemp("ds")
    raw = root / "raw"
    raw.mkdir()
    items = []
    for lang, make, ext in (
        ("python", py_text, "py"), ("go", go_text, "go"), ("kotlin", kt_text, "kt"),
    ):  # fmt: skip
        for r in range(10):
            for i in range(3):
                text = make(f"{lang}{r}x{i}")
                items.append(Snippet(text, lang, "github", f"{lang}-org/r{r}", "sha",
                                     f"src/f{i}.{ext}", "MIT", 1, text.count("\n")))  # fmt: skip
    save_snippets(items, raw / "all.jsonl")
    build_dataset([raw], root / "processed")
    return root / "processed"


# ------------------------------------------------------------------ resolve_languages


def test_auto_is_every_language_present_in_project_order():
    assert resolve_languages("auto", {"ruby", "go", "python"}) == ("python", "go", "ruby")
    assert resolve_languages(None, ["kotlin", "python"]) == ("python", "kotlin")


def test_core_and_all_are_fixed_sets():
    assert resolve_languages("core", ALL_LANGUAGES) == CORE_LANGUAGES
    assert resolve_languages("all", ALL_LANGUAGES) == ALL_LANGUAGES


def test_explicit_list_is_ordered_like_the_project_not_like_the_request():
    assert resolve_languages(["ruby", "python"], ALL_LANGUAGES) == ("python", "ruby")
    assert resolve_languages("go,python", ALL_LANGUAGES) == ("python", "go")


def test_a_requested_language_without_data_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match=r"no data for language\(s\) \['ruby'\]"):
        resolve_languages(["python", "ruby"], {"python", "go"})
    with pytest.raises(ValueError, match="no data for"):
        resolve_languages("core", {"python", "go"})  # the data has 2 of the 8 core languages
    with pytest.raises(ValueError, match="no data for"):
        resolve_languages("all", set(CORE_LANGUAGES))  # v5-like data cannot be evaluated as 14


def test_unknown_and_empty_are_errors():
    with pytest.raises(ValueError, match="unknown language"):
        resolve_languages(["python", "klingon"], {"python", "klingon"})
    with pytest.raises(ValueError, match="no languages"):
        resolve_languages("auto", [])


def test_parse_language_spec():
    assert parse_language_spec(" Auto ") == "auto"
    assert parse_language_spec("core") == "core"
    assert parse_language_spec("python, go") == ("python", "go")
    with pytest.raises(ValueError, match="empty language list"):
        parse_language_spec(" , ")


# ------------------------------------------------------------------ restrict / load


def test_restrict_dataset_filters_every_split_and_keeps_original_folds():
    def s(lang, i):
        return Snippet(f"x{i}\n" * 3, lang, "github", f"r/{lang}{i}", "c", "a.py", "MIT", 1, 3)

    dataset = {
        "train": [s("python", 0), s("go", 1), s("python", 2), s("ruby", 3)],
        "test": [s("go", 4), s("ruby", 5)],
        "wild": [s("ruby", 6)],
    }
    folds = np.array([0, 1, 2, 3])
    sub, sub_folds = restrict_dataset(dataset, folds, ("python", "go"))
    assert [x.language for x in sub["train"]] == ["python", "go", "python"]
    assert sub_folds.tolist() == [0, 1, 2]  # each kept snippet keeps its own fold id
    assert len(sub["test"]) == 1 and sub["wild"] == []
    assert len(dataset["train"]) == 4  # the original is untouched


def test_load_evaluation_data_auto_core_and_all(processed):
    dataset, folds, langs = bl.load_evaluation_data(processed)
    assert langs == ("python", "go", "kotlin") and len(folds) == len(dataset["train"])

    dataset, folds, langs = bl.load_evaluation_data(processed, "python,go")
    assert langs == ("python", "go")
    assert {s.language for s in dataset["train"] + dataset["test"]} == {"python", "go"}
    assert len(folds) == len(dataset["train"])

    with pytest.raises(ValueError, match="no data for language"):
        bl.load_evaluation_data(processed, "all")  # 11 of the 14 languages are missing


def test_subset_evaluation_uses_the_same_folds_as_the_full_run(processed):
    full_ds, full_folds, _ = bl.load_evaluation_data(processed)
    sub_ds, sub_folds, _ = bl.load_evaluation_data(processed, ("python", "go"))
    keep = [s.language != "kotlin" for s in full_ds["train"]]
    assert sub_folds.tolist() == full_folds[keep].tolist()


def test_evaluating_a_dataset_with_unlisted_languages_is_refused(processed):
    dataset, folds, _ = bl.load_evaluation_data(processed)  # includes kotlin
    with pytest.raises(ValueError, match=r"language\(s\) \['kotlin'\] that are not in"):
        bl.evaluate_model("nb", bl.make_models()["naive_bayes"], dataset, folds,
                          bl.Binarizer(n_features=40), languages=("python", "go"))  # fmt: skip


# ------------------------------------------------------------------ run_baselines


def test_run_baselines_covers_the_data_by_default(processed):
    results, meta = bl.run_baselines(processed, FAST, n_features=60, repeats=0)
    assert meta["languages"] == ["python", "go", "kotlin"]
    for r in results:
        assert set(r.per_language_f1) == {"python", "go", "kotlin"}
        assert np.array(r.confusion).shape == (3, 3)
    doc = bl.results_json(results, meta)
    assert doc["languages"] == ["python", "go", "kotlin"]
    md = bl.render_results_md(results, meta)
    assert "- Languages (3): python, go, kotlin" in md
    assert "| Model | python | go | kotlin |" in md  # the per-language table has all three


def test_run_baselines_subset_matches_a_data_set_without_the_other_languages(processed):
    results, meta = bl.run_baselines(processed, FAST, n_features=60, languages="python,go",
                                     repeats=0)  # fmt: skip
    assert meta["languages"] == ["python", "go"]
    assert meta["n_train"] == sum(
        v["snippets"] for k, v in meta["composition"]["train"].items() if k in ("python", "go")
    )
    assert "kotlin" not in meta["composition"]["train"]
    for r in results:
        assert set(r.per_language_f1) == {"python", "go"}
        assert np.array(r.confusion).sum() == meta["n_test"]


# ------------------------------------------------------------------ config


def test_config_languages_key_and_default():
    assert BaselinesConfig().languages == "auto"
    assert parse_baselines({"languages": "core"}).languages == "core"
    assert parse_baselines({"languages": ["python", "go"]}).languages == ("python", "go")
    assert parse_baselines({"languages": "python,go"}).languages == ("python", "go")
    for bad in (5, [], [1, 2], "", " , "):
        with pytest.raises(ValueError, match="languages"):
            parse_baselines({"languages": bad})
    ab = parse_ablations({"languages": "all", "studies": {"m": {"n_features": [10]}}})
    assert ab.languages == "all"


def test_cli_override_and_config_precedence():
    cfg = parse_baselines({"languages": "core"})
    assert cfg.with_overrides(languages=None).languages == "core"
    assert cfg.with_overrides(languages="all").languages == "all"
    assert cfg.with_overrides(languages="python,go").languages == ("python", "go")
    with pytest.raises(ValueError, match="languages"):
        cfg.with_overrides(languages=" , ")


def test_config_hash_of_older_runs_stays_valid_when_languages_is_auto():
    cfg = BaselinesConfig()
    without = {k: v for k, v in config_dict(cfg).items() if k != "languages"}
    import hashlib

    blob = json.dumps(without, sort_keys=True).encode("utf-8")
    assert config_hash(cfg) == hashlib.sha256(blob).hexdigest()[:12]
    assert config_hash(replace(cfg, languages="core")) != config_hash(cfg)


def test_repo_yaml_configs_still_load():
    from codelangtm.config import load_ablation_config, load_baselines_config

    assert load_baselines_config("configs/baselines.yaml").languages == "auto"
    assert load_ablation_config("configs/ablations.yaml").languages == "auto"


# ------------------------------------------------------------------ CLI


def test_cli_baselines_records_and_prints_the_languages(processed, tmp_path, capsys):
    out = tmp_path / "results.md"
    code = main(["baselines", "--data", str(processed), "--out", str(out), "--features", "60",
                 "--repeats", "0", "--model", "naive_bayes"])  # fmt: skip
    assert code == 0
    assert "languages (3): python, go, kotlin" in capsys.readouterr().out
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["languages"] == ["python", "go", "kotlin"] == side["meta"]["languages"]
    assert "- Languages (3): python, go, kotlin" in out.read_text(encoding="utf-8")

    code = main(["baselines", "--data", str(processed), "--out", str(out), "--features", "60",
                 "--repeats", "0", "--model", "naive_bayes",
                 "--languages", "python,go"])  # fmt: skip
    assert code == 0
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["languages"] == ["python", "go"]
    assert "languages (2): python, go" in capsys.readouterr().out


def test_cli_stops_with_a_clear_message_when_a_language_has_no_data(processed, tmp_path, capsys):
    out = tmp_path / "results.md"
    code = main(["baselines", "--data", str(processed), "--out", str(out), "--languages", "core"])
    assert code == 1 and not out.exists()
    err = capsys.readouterr().err
    assert "baselines failed" in err and "no data for language(s)" in err


def test_cli_diagnose_takes_languages(processed, tmp_path, capsys):
    out = tmp_path / "diag.md"
    assert main(["diagnose", "--data", str(processed), "--out", str(out), "--features", "60",
                 "--languages", "python,go"]) == 0  # fmt: skip
    text = out.read_text(encoding="utf-8")
    assert "2 languages (python, go)" in text and "kotlin" not in text
    assert main(["diagnose", "--data", str(processed), "--out", str(out), "--languages",
                 "ruby"]) == 1  # fmt: skip
    assert "no data for language(s) ['ruby']" in capsys.readouterr().err


def test_ablations_take_languages(processed):
    from codelangtm import ablations as ab

    cfg = parse_ablations({
        "data": processed.as_posix(), "models": ["naive_bayes"],
        "base": {"n_features": 40, "ngram_sizes": [2, 3], "use_delimiters": True},
        "studies": {"m": {"n_features": [20, 40]}},
        "languages": "python,go",
    })
    runs, meta = ab.run_ablations(cfg)  # from the config
    assert meta["languages"] == ["python", "go"]
    run = next(iter(runs.values()))
    assert set(run.per_language_f1) == {"python", "go"}
    runs, meta = ab.run_ablations(cfg, languages="auto")  # an explicit argument wins
    assert meta["languages"] == ["python", "go", "kotlin"]
    assert "| kotlin |" in ab.render_ablations_md(cfg, runs, meta)


def test_evaluation_default_constant_is_still_the_core_set():
    assert LANGUAGES == CORE_LANGUAGES
