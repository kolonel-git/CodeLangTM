"""Data diagnostics with a baseline model.

1. Confident learning (Northcutt et al., 2021): out-of-fold probabilities flag training snippets
   the model confidently assigns to another language: likely mislabels or ambiguous windows.
2. Shortcut probe: top logistic-regression features per language, with how many repos each
   appears in. A heavy feature seen in only one or two repos memorises a project, not a language.

Only the training split is used; the test set stays untouched.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .audit import _fence, load_dataset, permalink
from .baselines import load_folds, make_models
from .data import Snippet
from .features import Binarizer

MIN_FEATURE_REPOS = 3  # a top feature seen in fewer repos of its class is a shortcut suspect


def _lr_pipeline(n_features: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("binarize", Binarizer(n_features=n_features)),
            ("model", clone(make_models(seed)["logistic_regression"])),
        ]
    )


def oof_probabilities(
    train: Sequence[Snippet], folds: np.ndarray, n_features: int = 500, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """Per-snippet class probabilities from models that never saw that snippet."""
    classes = sorted({s.language for s in train})
    texts = np.asarray([s.text for s in train], dtype=object)
    labels = np.asarray([s.language for s in train])
    probs = np.zeros((len(train), len(classes)))
    for k in sorted(set(folds.tolist())):
        pipe = _lr_pipeline(n_features, seed)
        pipe.fit(list(texts[folds != k]), labels[folds != k])
        cols = [classes.index(c) for c in pipe.classes_]
        probs[np.ix_(np.flatnonzero(folds == k), cols)] = pipe.predict_proba(
            list(texts[folds == k])
        )
    return probs, classes


@dataclass(frozen=True)
class LabelIssue:
    snippet: Snippet
    given: str
    suggested: str
    p_given: float
    p_suggested: float


def confident_learning(
    train: Sequence[Snippet], probs: np.ndarray, classes: Sequence[str]
) -> tuple[np.ndarray, list[LabelIssue]]:
    """Confident joint and label-issue candidates, most confident first.

    Threshold t_j = mean probability of class j over snippets labelled j. A snippet counts
    toward class j' = argmax of p_k among classes with p_k >= t_k; if j' differs from its
    label, it is a candidate issue.
    """
    y = np.asarray([classes.index(s.language) for s in train])
    thresholds = np.array(
        [probs[y == j, j].mean() if np.any(y == j) else 1.0 for j in range(len(classes))]
    )
    joint = np.zeros((len(classes), len(classes)), dtype=int)
    issues: list[LabelIssue] = []
    for i, s in enumerate(train):
        above = np.flatnonzero(probs[i] >= thresholds)
        if above.size == 0:
            continue
        j = int(above[np.argmax(probs[i, above])])
        joint[y[i], j] += 1
        if j != y[i]:
            issues.append(
                LabelIssue(s, classes[y[i]], classes[j], float(probs[i, y[i]]), float(probs[i, j]))
            )
    issues.sort(key=lambda x: x.p_suggested - x.p_given, reverse=True)
    return joint, issues


@dataclass(frozen=True)
class FeatureInfo:
    token: str
    weight: float
    class_snippets: int  # snippets of this language containing the feature
    class_repos: int  # distinct repos among them
    other_share: float  # share of other-language snippets containing it

    @property
    def shortcut_suspect(self) -> bool:
        return self.class_repos < MIN_FEATURE_REPOS


def shortcut_probe(
    train: Sequence[Snippet], n_features: int = 500, seed: int = 0, top_k: int = 15
) -> dict[str, list[FeatureInfo]]:
    """Top positive-weight features per language, with repo spread and cross-language share."""
    texts = [s.text for s in train]
    labels = np.asarray([s.language for s in train])
    repos = np.asarray([s.repo for s in train])
    pipe = _lr_pipeline(n_features, seed).fit(texts, labels)
    vocab = pipe.named_steps["binarize"].vocabulary_
    x = pipe.named_steps["binarize"].transform(texts).astype(bool)
    model = pipe.named_steps["model"]
    coef = model.coef_
    if coef.shape[0] == 1:  # binary problem: sklearn stores one row for the positive class
        coef = np.vstack([-coef[0], coef[0]])

    out: dict[str, list[FeatureInfo]] = {}
    for c, lang in enumerate(model.classes_):
        in_class, others = labels == lang, labels != lang
        feats = []
        for idx in np.argsort(-coef[c])[:top_k]:
            has = x[:, idx]
            feats.append(
                FeatureInfo(
                    token=vocab[idx],
                    weight=float(coef[c, idx]),
                    class_snippets=int(np.sum(has & in_class)),
                    class_repos=len(set(repos[has & in_class])),
                    other_share=float(has[others].mean()) if others.any() else 0.0,
                )
            )
        out[str(lang)] = feats
    return out


def show_token(token: str) -> str:
    """Inline-code rendering with visible whitespace."""
    shown = json.dumps(token, ensure_ascii=False)[1:-1].replace(" ", "␠")
    return f"`` {shown} ``" if "`" in shown else f"`{shown}`"


def confused_pairs(joint: np.ndarray, classes: Sequence[str]) -> list[tuple[str, str, int]]:
    pairs = [
        (classes[i], classes[j], int(joint[i, j]))
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and joint[i, j]
    ]
    return sorted(pairs, key=lambda p: -p[2])


def render_diagnostics(
    train: Sequence[Snippet],
    joint: np.ndarray,
    classes: Sequence[str],
    issues: Sequence[LabelIssue],
    probe: dict[str, list[FeatureInfo]],
    meta: dict,
    max_issues: int = 40,
) -> str:
    md = [
        "# Data diagnostics",
        "",
        f"Generated {meta['generated']} from `{meta['data_dir']}` (train only, "
        f"{len(train)} snippets), logistic regression, M={meta['n_features']}, seed "
        f"{meta['seed']}. Regenerating overwrites this file.",
        "",
        "## 1. Label-issue candidates (confident learning)",
        "",
        f"{len(issues)} of {len(train)} training snippets ({len(issues) / max(len(train), 1):.1%}) "
        "are confidently predicted as another language by a model that never saw them.",
        "",
        "| Given → suggested | Count |",
        "| --- | --- |",
    ]
    md += [f"| {g} → {s} | {n} |" for g, s, n in confused_pairs(joint, classes)] or ["| none | 0 |"]
    md += [
        "",
        "Review each: **mislabel** (wrong language), **ambiguous** (mixed/embedded language, "
        "too little signal), or **fine** (model error).",
    ]
    for n, issue in enumerate(issues[:max_issues], 1):
        s = issue.snippet
        link = permalink(s)
        where = f"[`{s.repo}`]({link})" if link else f"`{s.repo}`"
        fence = _fence(s.text)
        md += [
            "",
            f"- [ ] **{n}.** labelled **{issue.given}** (p={issue.p_given:.2f}), model says "
            f"**{issue.suggested}** (p={issue.p_suggested:.2f}) — {where} `{s.path}` "
            f"L{s.start_line}-{s.end_line}",
            "",
            f"{fence}{issue.given}",
            s.text.rstrip("\n"),
            fence,
        ]
    if len(issues) > max_issues:
        md += ["", f"_{len(issues) - max_issues} more not shown._"]

    md += [
        "",
        "## 2. Shortcut probe (top features per language)",
        "",
        f"`repos` = distinct repos of that language containing the feature. Fewer than "
        f"{MIN_FEATURE_REPOS} marks a shortcut suspect (memorises a project, not a language). "
        "`other` = share of other-language snippets containing it. Whitespace: `␠` space, "
        "`\\n` newline, `\\t` tab.",
    ]
    for lang, feats in probe.items():
        md += [
            "",
            f"### {lang}",
            "",
            "| # | Feature | Weight | Snippets | Repos | Other | Flag |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r, f in enumerate(feats, 1):
            flag = "shortcut?" if f.shortcut_suspect else ""
            md.append(
                f"| {r} | {show_token(f.token)} | {f.weight:.2f} | {f.class_snippets} | "
                f"{f.class_repos} | {f.other_share:.0%} | {flag} |"
            )
    return "\n".join(md) + "\n"


@dataclass
class Diagnostics:
    joint: np.ndarray
    classes: list[str]
    issues: list[LabelIssue]
    probe: dict[str, list[FeatureInfo]]
    n_train: int

    def summary(self) -> str:
        lines = [f"label-issue candidates: {len(self.issues)} of {self.n_train}"]
        lines += [f"  {g} -> {s}: {n}" for g, s, n in confused_pairs(self.joint, self.classes)]
        suspects = [
            f"shortcut suspect: {lang} {json.dumps(f.token)} "
            f"(weight {f.weight:.2f}, {f.class_repos} repo(s))"
            for lang, feats in self.probe.items()
            for f in feats
            if f.shortcut_suspect
        ]
        return "\n".join(lines + (suspects or ["shortcut suspects: none"]))


def run_diagnostics(
    data_dir: str | Path,
    out: str | Path,
    n_features: int = 500,
    seed: int = 0,
    top_k: int = 15,
) -> Diagnostics:
    data_dir = Path(data_dir)
    train = load_dataset(data_dir)["train"]
    folds = load_folds(data_dir)
    if len(folds) != len(train):
        raise ValueError("folds.json does not match train.jsonl; rebuild the dataset")
    probs, classes = oof_probabilities(train, folds, n_features, seed)
    joint, issues = confident_learning(train, probs, classes)
    probe = shortcut_probe(train, n_features, seed, top_k)
    meta = {
        "generated": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "data_dir": str(data_dir),
        "n_features": n_features,
        "seed": seed,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_diagnostics(train, joint, classes, issues, probe, meta), encoding="utf-8"
    )
    return Diagnostics(joint, classes, issues, probe, len(train))
