from pathlib import Path

import pytest

from codelangtm.baselines import MODEL_NAMES
from codelangtm.config import (
    BaselinesConfig,
    FeatureConfig,
    config_hash,
    load_baselines_config,
    parse_baselines,
    parse_features,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baselines.yaml"


def test_repo_config_matches_defaults():
    # configs/baselines.yaml documents the defaults; `codelangtm baselines` alone == the file.
    cfg = load_baselines_config(REPO_CONFIG)
    assert cfg.models == MODEL_NAMES
    assert cfg.features == FeatureConfig()
    assert config_hash(cfg) == config_hash(BaselinesConfig(models=MODEL_NAMES))


def test_parse_full_config():
    cfg = parse_baselines({
        "data": "d", "out": "o.md", "seed": 3, "repeats": 0, "models": ["naive_bayes"],
        "features": {"n_features": 100, "ngram_sizes": [4, 2, 2], "use_delimiters": False},
    })  # fmt: skip
    assert cfg.data == Path("d") and cfg.out == Path("o.md")
    assert (cfg.seed, cfg.repeats, cfg.models) == (3, 0, ("naive_bayes",))
    assert cfg.features == FeatureConfig(100, (2, 4), False)  # sorted, deduplicated


def test_empty_config_is_defaults():
    assert parse_baselines(None) == BaselinesConfig()
    assert parse_features({}) == FeatureConfig()


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"sede": 1}, "unknown key"),
        ({"features": {"n_feature": 5}}, "unknown key"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"repeats": "10"}, "repeats"),
        ({"models": "naive_bayes"}, "models"),
        ({"data": 5}, "data"),
        ({"features": {"n_features": 0}}, "n_features"),
        ({"features": {"ngram_sizes": []}}, "ngram_sizes"),
        ({"features": {"ngram_sizes": [2, 0]}}, "ngram_sizes"),
        ({"features": {"use_delimiters": "yes"}}, "use_delimiters"),
        ({"features": [1]}, "mapping"),
        ([1, 2], "mapping"),
    ],
)
def test_invalid_values_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_baselines(raw)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_baselines_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("features: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_baselines_config(bad)


def test_overrides_only_apply_given_values():
    base = BaselinesConfig(seed=4, features=FeatureConfig(ngram_sizes=(3,)))
    cfg = base.with_overrides(seed=None, n_features=50, models=["naive_bayes"], repeats=None)
    assert cfg.seed == 4 and cfg.repeats == base.repeats
    assert cfg.features == FeatureConfig(50, (3,), True)  # other feature options kept
    assert cfg.models == ("naive_bayes",)


def test_hash_tracks_settings():
    a = BaselinesConfig()
    assert config_hash(a) == config_hash(BaselinesConfig())
    assert config_hash(a) != config_hash(a.with_overrides(seed=1))
    assert config_hash(a) != config_hash(a.with_overrides(n_features=250))


def test_feature_config_builds_binarizer():
    b = FeatureConfig(64, (3, 4), False).binarizer()
    assert (b.n_features, b.ngram_sizes, b.use_delimiters) == (64, (3, 4), False)
