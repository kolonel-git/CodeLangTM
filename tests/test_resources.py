import json
import pickle

import pytest

from codelangtm import resources as rs
from codelangtm.build import build_dataset
from codelangtm.cli import main
from codelangtm.data import Snippet, save_snippets

LANGS = ("python", "go")


def test_memory_is_measured_and_grows():
    current, peak = rs.memory_mb()
    assert 0 < current <= peak
    blob = bytearray(80 * 2**20)  # touch 80 MB so the working set really grows
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    current_after, peak_after = rs.memory_mb()
    # current memory must grow by the touched pages; the peak may already have been higher
    assert current_after >= current + 60 and peak_after >= current_after
    del blob


def py_text(tag):
    return "".join(f"def f_{tag}_{j}(x):\n    return x + len('{tag}') * {j}\n" for j in range(12))


def go_text(tag):
    body = "".join(f"func F{tag}{j}(x int) int {{\n\treturn x + {j}\n}}\n" for j in range(8))
    return "package main\n\n" + body


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("res")
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
    build_dataset([raw], root / "processed", languages=LANGS)
    config = root / "tm.yaml"
    config.write_text(
        f"data: {(root / 'processed').as_posix()}\nseeds: [1, 2]\n"
        "baselines: [naive_bayes]\nfeatures: {n_features: 30}\n"
        "tm:\n  small: {n_clauses: 6, T: 3, epochs: 1}\n",
        encoding="utf-8",
    )
    return root, config


def test_fit_job_in_a_fresh_process(setup):
    _, config = setup
    out = rs.run_job(["fit", "--spec", "baseline:naive_bayes", "--config", str(config)])
    assert out["wall_s"] > 0 and out["cpu_s"] >= 0
    assert out["peak_rss_mb"] >= out["rss_before_mb"] > 0
    assert out["added_mb"] == pytest.approx(out["peak_rss_mb"] - out["rss_before_mb"])


def test_predict_job_with_a_pickled_pipeline(setup, tmp_path):
    root, config = setup
    _, pipe = rs._make_pipeline("baseline:naive_bayes", config)
    from codelangtm.audit import load_dataset

    train = load_dataset(root / "processed")["train"]
    pipe.fit([s.text for s in train], [s.language for s in train])
    path = tmp_path / "nb.pkl"
    path.write_bytes(pickle.dumps(pipe))
    out = rs.run_job(["predict", "--model-file", str(path), "--data", str(root / "processed"),
                      "--passes", "2"])  # fmt: skip
    assert out["n_snippets"] > 0 and out["batch_snippets_per_s"] > 0
    for part in ("binarize", "predict", "total"):
        assert 0 < out[f"{part}_ms_median"] <= out[f"{part}_ms_p95"]


def test_bad_job_raises_with_stderr(setup):
    _, config = setup
    with pytest.raises(RuntimeError, match="resource job failed"):
        rs.run_job(["fit", "--spec", "nonsense:x", "--config", str(config)])


def test_model_specs_use_the_median_seed(setup, tmp_path):
    _, config = setup
    from codelangtm.config import load_tm_config

    cfg = load_tm_config(config)
    results = {"tm": {"small": {"seeds": [{"seed": 1, "cv_mean": 0.9},
                                          {"seed": 2, "cv_mean": 0.8}]}}}  # fmt: skip
    with pytest.raises(FileNotFoundError, match="tm-results"):
        rs.model_specs(cfg, results, tmp_path)
    (tmp_path / "small_seed2.json").write_text("{}", encoding="utf-8")
    specs = rs.model_specs(cfg, results, tmp_path)
    assert [s["spec"] for s in specs] == ["baseline:naive_bayes", "tm:small:2"]  # lower median


def test_measure_all_end_to_end(setup, tmp_path, capsys):
    pytest.importorskip("tmu")
    root, config = setup
    from codelangtm.config import load_tm_config
    from codelangtm.tm_results import train_final

    cfg = load_tm_config(config)
    models = tmp_path / "models"
    train_final(cfg, "small", 1, models / "small_seed1.json")
    results = {"schema": "codelangtm.tm-results/1",
               "tm": {"small": {"seeds": [{"seed": 1, "cv_mean": 0.9}]}}}  # fmt: skip
    (tmp_path / "tm-results.json").write_text(json.dumps(results), encoding="utf-8")
    out = tmp_path / "resources.md"
    assert main(["resources", "--config", str(config), "--results",
                 str(tmp_path / "tm-results.json"), "--models-dir", str(models), "--repeats",
                 "1", "--passes", "1", "--out", str(out)]) == 0  # fmt: skip
    assert "small (seed 1)" in capsys.readouterr().out
    data = json.loads((tmp_path / "resources.json").read_text(encoding="utf-8"))
    assert data["schema"] == rs.RESOURCES_SCHEMA
    assert [m["name"] for m in data["models"]] == ["naive_bayes", "small (seed 1)"]
    tm = data["models"][1]
    assert tm["model_kind"].startswith("TM model file") and len(tm["fit_runs"]) == 1
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# Resources (process level)") and "| small (seed 1) |" in md


def test_cli_missing_results_fails(tmp_path, capsys):
    assert main(["resources", "--results", str(tmp_path / "none.json")]) == 1
    assert "sidecar not found" in capsys.readouterr().err
