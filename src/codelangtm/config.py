"""YAML experiment configs: one file per experiment in `configs/`, validated strictly.

Unknown keys and wrong types raise `ValueError`, so a typo cannot silently fall back to a
default. `config_hash` fingerprints the effective settings (file + CLI overrides) and is
recorded in generated results.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .data import parse_language_spec
from .features import SELECTIONS, Binarizer


@dataclass(frozen=True)
class FeatureConfig:
    n_features: int = 500
    ngram_sizes: tuple[int, ...] = (2, 3)
    use_delimiters: bool = True
    selection: str = "frequency"
    min_df: int = 1
    word_tokens: bool = False

    def binarizer(self) -> Binarizer:
        return Binarizer(
            n_features=self.n_features,
            ngram_sizes=self.ngram_sizes,
            use_delimiters=self.use_delimiters,
            selection=self.selection,
            min_df=self.min_df,
            word_tokens=self.word_tokens,
        )


@dataclass(frozen=True)
class BaselinesConfig:
    data: Path = Path("data/processed")
    out: Path = Path("docs/results.md")
    models: tuple[str, ...] | None = None  # None = all
    seed: int = 0
    repeats: int = 10
    features: FeatureConfig = field(default_factory=FeatureConfig)
    languages: str | tuple[str, ...] = "auto"  # auto | core | all | explicit names

    def with_overrides(self, **overrides: Any) -> BaselinesConfig:
        """Apply CLI overrides; `None` means "not given". `n_features` targets the feature block."""
        given = {k: v for k, v in overrides.items() if v is not None}
        cfg = self
        if "n_features" in given:
            cfg = replace(cfg, features=replace(cfg.features, n_features=given.pop("n_features")))
        if "models" in given:
            given["models"] = tuple(given["models"])
        if "languages" in given:
            given["languages"] = _language_spec("config.languages", given["languages"])
        return replace(cfg, **given)


def _check_keys(section: str, raw: dict, allowed: set[str]) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{section}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def _language_spec(where: str, value: Any) -> str | tuple[str, ...]:
    """`auto`, `core`, `all`, a comma-separated string or a list of language names."""
    try:
        if isinstance(value, str):
            return parse_language_spec(value)
        if isinstance(value, list | tuple) and value and all(isinstance(v, str) for v in value):
            return tuple(value)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from e
    raise ValueError(f"{where}: expected auto, core, all or a list of language names")


def _int(section: str, key: str, value: Any, minimum: int) -> int:
    # bool is an int subclass; reject it explicitly
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{section}.{key}: expected an integer >= {minimum}, got {value!r}")
    return value


def parse_features(raw: dict | None, section: str = "features") -> FeatureConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{section}: expected a mapping, got {raw!r}")
    _check_keys(section, raw, {f.name for f in fields(FeatureConfig)})
    cfg = FeatureConfig()
    if "n_features" in raw:
        cfg = replace(cfg, n_features=_int(section, "n_features", raw["n_features"], 1))
    if "ngram_sizes" in raw:
        sizes = raw["ngram_sizes"]
        if not isinstance(sizes, list) or not sizes:
            raise ValueError(f"{section}.ngram_sizes: expected a non-empty list, got {sizes!r}")
        sizes = tuple(sorted({_int(section, "ngram_sizes", n, 1) for n in sizes}))
        cfg = replace(cfg, ngram_sizes=sizes)
    for key in ("use_delimiters", "word_tokens"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ValueError(f"{section}.{key}: expected true/false")
            cfg = replace(cfg, **{key: raw[key]})
    if "selection" in raw:
        if raw["selection"] not in SELECTIONS:
            raise ValueError(f"{section}.selection: expected one of {list(SELECTIONS)}")
        cfg = replace(cfg, selection=raw["selection"])
    if "min_df" in raw:
        cfg = replace(cfg, min_df=_int(section, "min_df", raw["min_df"], 1))
    return cfg


def parse_baselines(raw: dict | None) -> BaselinesConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config: expected a mapping, got {raw!r}")
    _check_keys("config", raw, {f.name for f in fields(BaselinesConfig)})
    cfg = BaselinesConfig(features=parse_features(raw.get("features")))
    for key in ("data", "out"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise ValueError(f"config.{key}: expected a path string")
            cfg = replace(cfg, **{key: Path(raw[key])})
    if "models" in raw:
        models = raw["models"]
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError("config.models: expected a list of model names")
        cfg = replace(cfg, models=tuple(models))
    if "seed" in raw:
        cfg = replace(cfg, seed=_int("config", "seed", raw["seed"], 0))
    if "repeats" in raw:
        cfg = replace(cfg, repeats=_int("config", "repeats", raw["repeats"], 0))
    if "languages" in raw:
        cfg = replace(cfg, languages=_language_spec("config.languages", raw["languages"]))
    return cfg


@dataclass(frozen=True)
class Study:
    """One ablation: every combination of the varied feature options, the rest from `base`."""

    name: str
    settings: tuple[FeatureConfig, ...]
    varied: tuple[str, ...]


@dataclass(frozen=True)
class AblationConfig:
    data: Path = Path("data/processed")
    out: Path = Path("docs/ablations.md")
    seed: int = 0
    models: tuple[str, ...] = ("logistic_regression", "naive_bayes")
    base: FeatureConfig = field(default_factory=FeatureConfig)
    studies: tuple[Study, ...] = ()
    languages: str | tuple[str, ...] = "auto"

    def feature_settings(self) -> list[FeatureConfig]:
        """Unique settings across all studies (base first), in first-seen order."""
        seen = {self.base: None}
        for study in self.studies:
            seen.update(dict.fromkeys(study.settings))
        return list(seen)


def parse_study(name: str, raw: Any, base: FeatureConfig) -> Study:
    section = f"studies.{name}"
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{section}: expected a mapping of feature option -> list of values")
    _check_keys(section, raw, {f.name for f in fields(FeatureConfig)})
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{section}.{key}: expected a non-empty list of values")
    varied = tuple(raw)
    settings = []
    for combo in itertools.product(*(raw[k] for k in varied)):
        options = {**config_dict(base), **dict(zip(varied, combo, strict=True))}
        settings.append(parse_features(options, section))
    return Study(name, tuple(dict.fromkeys(settings)), varied)


def parse_ablations(raw: dict | None) -> AblationConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config: expected a mapping, got {raw!r}")
    _check_keys("config", raw, {f.name for f in fields(AblationConfig)})
    base = parse_features(raw.get("base"), "base")
    cfg = AblationConfig(base=base)
    for key in ("data", "out"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise ValueError(f"config.{key}: expected a path string")
            cfg = replace(cfg, **{key: Path(raw[key])})
    if "seed" in raw:
        cfg = replace(cfg, seed=_int("config", "seed", raw["seed"], 0))
    if "models" in raw:
        models = raw["models"]
        if not (isinstance(models, list) and models and all(isinstance(m, str) for m in models)):
            raise ValueError("config.models: expected a non-empty list of model names")
        cfg = replace(cfg, models=tuple(models))
    if "languages" in raw:
        cfg = replace(cfg, languages=_language_spec("config.languages", raw["languages"]))
    studies = raw.get("studies")
    if not isinstance(studies, dict) or not studies:
        raise ValueError("config.studies: expected a mapping of study name -> varied options")
    return replace(cfg, studies=tuple(parse_study(n, s, base) for n, s in studies.items()))


EPOCH_RULES = ("plateau", "max")


@dataclass(frozen=True)
class CurveConfig:
    """Training-curve scan (B4): how far to train and how to pick the epoch count."""

    max_epochs: int = 150
    smooth_window: int = 11  # moving-average width over epochs (odd)
    plateau_tolerance: float = 0.005  # plateau rule: first epoch within this of the smoothed max
    epoch_rule: str = "plateau"  # "plateau" (first levelled-off epoch) or "max" (smoothed peak)


@dataclass(frozen=True)
class TMSetting:
    """One named TM configuration; `params` are TMLanguageClassifier arguments except seed."""

    name: str
    params: tuple[tuple[str, Any], ...]  # sorted (key, value) pairs, hashable

    def kwargs(self) -> dict:
        return dict(self.params)


@dataclass(frozen=True)
class TMConfig:
    data: Path = Path("data/processed")
    out: Path = Path("docs/tm-results.md")
    curves_out: Path = Path("docs/tm-curves.md")
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    repeats: int = 10
    baselines: tuple[str, ...] = ("naive_bayes", "logistic_regression")
    features: FeatureConfig = field(default_factory=FeatureConfig)
    curves: CurveConfig = field(default_factory=CurveConfig)
    settings: tuple[TMSetting, ...] = ()

    def setting(self, name: str) -> TMSetting:
        for s in self.settings:
            if s.name == name:
                return s
        available = [s.name for s in self.settings]
        raise ValueError(f"unknown TM setting {name!r}; available: {available}")


def _tm_param_names() -> set[str]:
    from .model import TMLanguageClassifier

    return set(TMLanguageClassifier().get_params()) - {"seed"}


def _parse_tm_setting(name: str, raw: Any) -> TMSetting:
    section = f"tm.{name}"
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{section}: expected a mapping of TMLanguageClassifier options")
    _check_keys(section, raw, _tm_param_names())
    for key in ("n_clauses", "T", "epochs"):
        if key in raw:
            _int(section, key, raw[key], 1)
    if raw.get("n_clauses", 2) % 2:
        raise ValueError(f"{section}.n_clauses: must be even (half for, half against)")
    if "s" in raw and (isinstance(raw["s"], bool) or not isinstance(raw["s"], int | float)
                       or raw["s"] < 1):  # fmt: skip
        raise ValueError(f"{section}.s: expected a number >= 1, got {raw['s']!r}")
    if "weighted_clauses" in raw and not isinstance(raw["weighted_clauses"], bool):
        raise ValueError(f"{section}.weighted_clauses: expected true/false")
    return TMSetting(name, tuple(sorted(raw.items())))


def parse_tm(raw: dict | None) -> TMConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config: expected a mapping, got {raw!r}")
    allowed = {f.name for f in fields(TMConfig)} - {"settings"} | {"tm"}
    _check_keys("config", raw, allowed)
    cfg = TMConfig(features=parse_features(raw.get("features")))
    for key in ("data", "out", "curves_out"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise ValueError(f"config.{key}: expected a path string")
            cfg = replace(cfg, **{key: Path(raw[key])})
    if "seeds" in raw:
        seeds = raw["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("config.seeds: expected a non-empty list of integers >= 1")
        # TMU hangs with seed 0 (issues-and-fixes E6)
        seeds = tuple(_int("config", "seeds", s, 1) for s in seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("config.seeds: duplicate seeds")
        cfg = replace(cfg, seeds=seeds)
    if "repeats" in raw:
        cfg = replace(cfg, repeats=_int("config", "repeats", raw["repeats"], 0))
    if "baselines" in raw:
        names = raw["baselines"]
        if not isinstance(names, list) or not all(isinstance(m, str) for m in names):
            raise ValueError("config.baselines: expected a list of baseline model names")
        cfg = replace(cfg, baselines=tuple(names))
    if "curves" in raw:
        c = raw["curves"]
        if not isinstance(c, dict):
            raise ValueError("config.curves: expected a mapping")
        _check_keys("curves", c, {f.name for f in fields(CurveConfig)})
        curves = CurveConfig()
        if "max_epochs" in c:
            curves = replace(curves, max_epochs=_int("curves", "max_epochs", c["max_epochs"], 1))
        if "smooth_window" in c:
            w = _int("curves", "smooth_window", c["smooth_window"], 1)
            if w % 2 == 0:
                raise ValueError("curves.smooth_window: must be odd (centred average)")
            curves = replace(curves, smooth_window=w)
        if "plateau_tolerance" in c:
            tol = c["plateau_tolerance"]
            if isinstance(tol, bool) or not isinstance(tol, int | float) or tol < 0:
                raise ValueError("curves.plateau_tolerance: expected a number >= 0")
            curves = replace(curves, plateau_tolerance=float(tol))
        if "epoch_rule" in c:
            if c["epoch_rule"] not in EPOCH_RULES:
                raise ValueError(f"curves.epoch_rule: expected one of {list(EPOCH_RULES)}")
            curves = replace(curves, epoch_rule=c["epoch_rule"])
        cfg = replace(cfg, curves=curves)
    tm = raw.get("tm")
    if not isinstance(tm, dict) or not tm:
        raise ValueError("config.tm: expected a mapping of setting name -> TM options")
    return replace(cfg, settings=tuple(_parse_tm_setting(n, s) for n, s in tm.items()))


def load_tm_config(path: str | Path) -> TMConfig:
    return parse_tm(load_yaml(path))


def load_ablation_config(path: str | Path) -> AblationConfig:
    return parse_ablations(load_yaml(path))


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: invalid YAML ({e})") from e


def load_baselines_config(path: str | Path) -> BaselinesConfig:
    return parse_baselines(load_yaml(path))


def config_dict(cfg: Any) -> dict:
    """JSON-safe dict of a config dataclass (paths as POSIX strings, tuples as lists)."""

    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        return value

    return clean(asdict(cfg))


def config_hash(cfg: Any) -> str:
    """Short SHA-256 of the effective settings; identical settings give identical hashes."""
    settings = config_dict(cfg)
    if settings.get("languages") == "auto":  # the default: hashes of older runs stay valid
        settings.pop("languages")
    blob = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
