"""The clause inspector (M3 step B6): what exactly the Tsetlin Machine learned, and why it
predicts what it predicts. No black box: every clause is shown as a readable rule.

Works on a saved model file (`TrainedModel`), so TMU is not needed, except for the optional
formation replay, which retrains the model with per-epoch snapshots. Statistics describe clauses
on the training set only; the test set is never used to describe or select clauses.

Vocabulary used here:
- *literal*: `has("fn ")` (the n-gram is present) or `NOT has(";")` (it is absent);
- *rule / clause*: an AND of literals; it *fires* when all its literals are true;
- *for / against*: a clause with positive weight adds votes to its language when it fires,
  a negative-weight clause subtracts them;
- *signature feature* of a language: an n-gram its "for" clauses rely on much more than the
  "for" clauses of other languages do.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .features import WORD_PREFIX
from .model import TMState, TrainedModel

CLAUSES_SCHEMA = "codelangtm.clauses/1"
TOP_CLAUSES = 20
TOP_FEATURES = 10


# ------------------------------------------------------------------ naming


def show_term(term: str) -> str:
    """Quote a vocabulary term so whitespace is visible: `"{\\n\\t"`, `word("SELECT")`."""
    if term.startswith(WORD_PREFIX):
        return f'word("{_escape(term[1:])}")'
    return f'"{_escape(term)}"'


def _escape(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 32 or ord(ch) == 127:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)


def literal_text(literal: int, vocabulary: Sequence[str]) -> str:
    """Literal id -> `has("fn ")` or `NOT has(";")` (ids >= M are negations)."""
    m = len(vocabulary)
    if not 0 <= literal < 2 * m:
        raise ValueError(f"literal id {literal} out of range for {m} features")
    term = show_term(vocabulary[literal % m])
    if term.startswith("word("):
        return term if literal < m else f"NOT {term}"
    return f"has({term})" if literal < m else f"NOT has({term})"


def format_rule(language: str, included: list[str], negated: list[str]) -> str:
    """Render a clause from raw terms, e.g. `python = has("def ") AND NOT has(";")`."""
    parts = [f"has({show_term(t)})" for t in included]
    parts += [f"NOT has({show_term(t)})" for t in negated]
    return f"{language} = " + (" AND ".join(parts) if parts else "(empty clause: never fires)")


# ------------------------------------------------------------------ rules


@dataclass
class Rule:
    index: int  # position in the model (clause id)
    language: str
    weight: int
    literals: list[int]
    text: str
    # statistics on the training set (filled by `clause_stats`)
    fires_own: int = 0
    fires_other: int = 0
    own_total: int = 0
    other_total: int = 0

    @property
    def polarity(self) -> str:
        return "for" if self.weight > 0 else "against" if self.weight < 0 else "none"

    @property
    def n_literals(self) -> int:
        return len(self.literals)

    @property
    def coverage(self) -> float:
        """Share of the language's own training snippets the clause fires on."""
        return self.fires_own / self.own_total if self.own_total else 0.0

    @property
    def false_fire_rate(self) -> float:
        """Share of other languages' training snippets the clause fires on."""
        return self.fires_other / self.other_total if self.other_total else 0.0

    @property
    def precision(self) -> float:
        """Of the snippets it fires on, the share that belong to its language."""
        fired = self.fires_own + self.fires_other
        return self.fires_own / fired if fired else 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index, "language": self.language, "polarity": self.polarity,
            "weight": self.weight, "literals": self.literals, "rule": self.text,
            "n_literals": self.n_literals, "fires_own": self.fires_own,
            "fires_other": self.fires_other, "coverage": self.coverage,
            "false_fire_rate": self.false_fire_rate, "precision": self.precision,
        }  # fmt: skip


def extract_rules(state: TMState, vocabulary: Sequence[str]) -> list[Rule]:
    """Every clause of every language as a readable rule, in model order. Nothing is dropped:
    empty clauses and zero-weight clauses are included (they never influence a prediction)."""
    if len(vocabulary) != state.n_features:
        raise ValueError(f"vocabulary has {len(vocabulary)} terms, model {state.n_features}")
    rules = []
    for i in range(state.n_clauses):
        literals = sorted(int(v) for v in state.clause_literals(i))
        # plain literals first, then negations, each in vocabulary order
        text = " AND ".join(literal_text(v, vocabulary) for v in literals)
        rules.append(Rule(i, state.classes[state.clause_class[i]], int(state.weights[i]),
                          literals, text or "(empty clause: never fires)"))  # fmt: skip
    return rules


def evaluate_rule(rule: Rule, x: np.ndarray) -> np.ndarray:
    """Evaluate a rule directly from its literal list (independent of TMState's matrix code)."""
    x = np.asarray(x)
    m = x.shape[1]
    if not rule.literals:
        return np.zeros(len(x), dtype=bool)
    fire = np.ones(len(x), dtype=bool)
    for lit in rule.literals:
        fire &= (x[:, lit] == 1) if lit < m else (x[:, lit - m] == 0)
    return fire


def clause_stats(rules: list[Rule], state: TMState, x: np.ndarray, y: Sequence[str]) -> None:
    """Fill each rule's firing counts on (training) data, in place."""
    y = np.asarray(y)
    fired = state.clause_outputs(x)  # (n, clauses)
    counts = Counter(y.tolist())
    for rule in rules:
        own = y == rule.language
        column = fired[:, rule.index]
        rule.fires_own = int(column[own].sum())
        rule.fires_other = int(column[~own].sum())
        rule.own_total = counts[rule.language]
        rule.other_total = len(y) - counts[rule.language]


# ------------------------------------------------------------------ bird's-eye view


def language_summary(rules: list[Rule], languages: Sequence[str]) -> dict:
    out = {}
    for lang in languages:
        mine = [r for r in rules if r.language == lang]
        nonempty = [r for r in mine if r.literals]
        lengths = [r.n_literals for r in nonempty]
        features = {lit for r in nonempty for lit in r.literals}
        out[lang] = {
            "clauses": len(mine),
            "for": sum(r.polarity == "for" for r in mine),
            "against": sum(r.polarity == "against" for r in mine),
            "zero_weight": sum(r.polarity == "none" for r in mine),
            "empty": len(mine) - len(nonempty),
            "mean_literals": statistics.fmean(lengths) if lengths else 0.0,
            "median_literals": statistics.median(lengths) if lengths else 0,
            "distinct_literals": len(features),
            "total_weight_for": sum(r.weight for r in mine if r.weight > 0),
            "total_weight_against": -sum(r.weight for r in mine if r.weight < 0),
            "duplicate_clauses": _duplicates(nonempty),
            # needs `clause_stats` first: clauses that fire on at most one training snippet
            "narrow_clauses": sum(r.fires_own + r.fires_other <= 1 for r in nonempty),
            "max_literals": max(lengths, default=0),
        }
    return out


def _duplicates(rules: list[Rule]) -> int:
    """Clauses whose literal set exactly repeats another clause of the same language."""
    seen = Counter(tuple(r.literals) for r in rules)
    return sum(n - 1 for n in seen.values() if n > 1)


def literal_usage(rules: list[Rule], languages: Sequence[str], n_features: int,
                  polarity: str = "for") -> np.ndarray:  # fmt: skip
    """(languages, 2M) share of each language's `polarity` clauses that include each literal."""
    usage = np.zeros((len(languages), 2 * n_features))
    for k, lang in enumerate(languages):
        mine = [r for r in rules if r.language == lang and r.polarity == polarity and r.literals]
        for r in mine:
            usage[k, r.literals] += 1
        if mine:
            usage[k] /= len(mine)
    return usage


def signature_features(rules: list[Rule], languages: Sequence[str], vocabulary: Sequence[str],
                       top: int = TOP_FEATURES) -> dict:  # fmt: skip
    """Per language, the plain literals (`has(...)`) its "for" clauses use most, with lift:
    usage in this language's "for" clauses / mean usage in the other languages' (+ small
    smoothing so a feature no other language uses gets a large but finite lift)."""
    usage, lift, score = _signature_arrays(rules, languages, len(vocabulary))
    out = {}
    for k, lang in enumerate(languages):
        order = [j for j in np.argsort(-score[k], kind="stable") if usage[k, j] > 0][:top]
        out[lang] = [{"feature": show_term(vocabulary[j]), "id": int(j),
                      "share_of_for_clauses": float(usage[k, j]), "lift": float(lift[k, j])}
                     for j in order]  # fmt: skip
    return out


def _signature_arrays(rules: list[Rule], languages: Sequence[str], n_features: int):
    """(usage, lift, score), each (languages, M), over plain literals."""
    usage = literal_usage(rules, languages, n_features)[:, :n_features]
    lift, score = np.zeros_like(usage), np.zeros_like(usage)
    for k in range(len(languages)):
        others = np.delete(usage, k, axis=0).mean(axis=0)
        lift[k] = (usage[k] + 0.01) / (others + 0.01)
        score[k] = usage[k] * np.log(lift[k])  # used often here AND rarely elsewhere
    return usage, lift, score


def signature_scores(rules: list[Rule], languages: Sequence[str], vocabulary: Sequence[str]
                     ) -> dict[str, dict[str, float]]:  # fmt: skip
    """Per language, the signature score of every vocabulary term (for cross-seed comparisons:
    terms, not ids, so models with different vocabularies still line up)."""
    _, _, score = _signature_arrays(rules, languages, len(vocabulary))
    return {lang: dict(zip(vocabulary, score[k].tolist(), strict=True))
            for k, lang in enumerate(languages)}  # fmt: skip


def class_overlap(rules: list[Rule], languages: Sequence[str], n_features: int) -> np.ndarray:
    """Jaccard similarity between languages' sets of plain literals used in "for" clauses."""
    sets = []
    for lang in languages:
        sets.append({lit for r in rules if r.language == lang and r.polarity == "for"
                     for lit in r.literals if lit < n_features})  # fmt: skip
    k = len(languages)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            union = sets[i] | sets[j]
            out[i, j] = len(sets[i] & sets[j]) / len(union) if union else 0.0
    return out


def top_clauses(rules: list[Rule], language: str, top: int = TOP_CLAUSES) -> list[Rule]:
    """The language's "for" clauses ranked by coverage x precision, then weight."""
    mine = [r for r in rules if r.language == language and r.polarity == "for" and r.literals]
    return sorted(mine, key=lambda r: (-(r.coverage * r.precision), -r.weight, r.index))[:top]


def seed_stability(
    signatures: dict,
    other_models: dict[str, dict],
    scores: dict | None = None,
    other_scores: dict | None = None,
) -> dict:
    """Per language: share of the official model's top signature features that also appear
    in each other seed's top list (same length), and the mean over seeds. With `scores`
    (from `signature_scores`) also the Pearson correlation of the signature scores of all
    terms both vocabularies share: a rank-free view that near-ties in the top list cannot flip."""
    out = {}
    for lang, feats in signatures.items():
        mine = {f["feature"] for f in feats}
        per_seed, corr = {}, {}
        for seed, sigs in other_models.items():
            theirs = {f["feature"] for f in sigs.get(lang, [])}
            per_seed[seed] = len(mine & theirs) / len(mine) if mine else 0.0
            if scores and other_scores:
                a, b = scores[lang], other_scores[seed][lang]
                common = sorted(set(a) & set(b))
                pair = np.corrcoef([a[t] for t in common], [b[t] for t in common])
                corr[seed] = float(pair[0, 1])
        out[lang] = {"per_seed": per_seed,
                     "mean": statistics.fmean(per_seed.values()) if per_seed else None}  # fmt: skip
        if corr:
            out[lang]["score_correlation"] = corr
            out[lang]["mean_correlation"] = statistics.fmean(corr.values())
    return out


# ------------------------------------------------------------------ explain one snippet


@dataclass
class Explanation:
    text_length: int
    predicted: str
    class_sums: dict[str, int]
    fired: list[Rule] = field(default_factory=list)  # every clause that fired, any language
    features_present: list[str] = field(default_factory=list)

    def contributions(self, language: str) -> list[Rule]:
        """Fired clauses of one language, biggest absolute contribution first."""
        mine = [r for r in self.fired if r.language == language and r.weight]
        return sorted(mine, key=lambda r: (-abs(r.weight), r.index))


def explain_snippet(
    model: TrainedModel, text: str, rules: list[Rule] | None = None
) -> Explanation:
    """Votes per language for one snippet and the exact clauses that fired.

    The class sums equal the model's own `class_sums`; each fired clause contributes its weight
    to its language (positive: for, negative: against)."""
    vocabulary = model.binarizer.vocabulary_
    rules = rules or extract_rules(model.state, vocabulary)
    x = model.binarizer.transform([text])
    fired_mask = model.state.clause_outputs(x)[0]
    sums = model.state.class_sums(x)[0]
    return Explanation(
        text_length=len(text),
        predicted=str(model.state.predict(x)[0]),
        class_sums={c: int(s) for c, s in zip(model.state.classes, sums, strict=True)},
        fired=[rules[i] for i in np.flatnonzero(fired_mask)],
        features_present=[show_term(vocabulary[j]) for j in np.flatnonzero(x[0])],
    )


def short_rule(text: str, max_parts: int = 6) -> str:
    """First `max_parts` parts of a long rule, then how many are left out."""
    parts = text.split(" AND ")  # terms are at most 3 characters: never contain " AND "
    if len(parts) <= max_parts:
        return text
    return " AND ".join(parts[:max_parts]) + f" AND ... (+{len(parts) - max_parts} more)"


def format_explanation(e: Explanation, top: int = 8) -> str:
    ranked = sorted(e.class_sums.items(), key=lambda kv: -kv[1])
    lines = [f"prediction: {e.predicted}", "", "votes per language (sum of fired clause weights):"]
    lines += [f"  {lang:<12}{votes:>6}" for lang, votes in ranked]
    for lang, _ in ranked[:2]:
        mine = e.contributions(lang)
        gained = sum(r.weight for r in mine if r.weight > 0)
        lost = -sum(r.weight for r in mine if r.weight < 0)
        lines += ["", f"{lang}: {len(mine)} clauses fired (+{gained} for, -{lost} against); "
                  f"strongest {min(top, len(mine))}:"]  # fmt: skip
        for r in mine[:top]:
            lines.append(f"  {r.weight:+4d}  #{r.index:<5} {short_rule(r.text)}")
    lines += ["", f"{len(e.features_present)} of the model's features are present in the snippet"]
    return "\n".join(lines)


# ------------------------------------------------------------------ formation replay


def formation_replay(
    model: TrainedModel,
    texts: Sequence[str],
    labels: Sequence[str],
    signatures: dict,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Retrain the model's setting and seed with a snapshot after every epoch (needs TMU).

    Per epoch: Jaccard similarity between the included (clause, literal) pairs so far and the
    final model's, and literals per non-empty clause. Per language signature feature: the share
    of the language's "for" clauses that include it after every epoch, and its *settled epoch*
    (from then on the share stays at least half its final share). Checks that the replay ends
    at exactly the saved model."""
    from .model import TMLanguageClassifier

    params = {k: v for k, v in model.state.params.items() if k != "epochs_trained"}
    epochs = int(model.state.params.get("epochs_trained", params.get("epochs", 1)))
    params.pop("epochs", None)
    clf = TMLanguageClassifier(**params)
    x = model.binarizer.transform(texts)
    final = model.state.include_matrix().astype(bool)
    langs = list(model.state.classes)
    share = {lang: {f["feature"]: [] for f in signatures[lang]} for lang in langs}
    jaccard, literals = [], []
    for epoch in range(1, epochs + 1):
        clf.partial_fit(x, labels, classes=langs)
        state = clf.state_
        include = state.include_matrix().astype(bool)
        jaccard.append(float((include & final).sum() / max(1, (include | final).sum())))
        counts = state.literal_counts()
        literals.append(float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0)
        positive = state.weights > 0
        for k, lang in enumerate(langs):
            rows = include[(state.clause_class == k) & positive]
            for f in signatures[lang]:
                share[lang][f["feature"]].append(float(rows[:, f["id"]].mean()) if len(rows)
                                                 else 0.0)  # fmt: skip
        if progress and epoch % 20 == 0:
            progress(f"formation replay: epoch {epoch}/{epochs}")
    reproduces = clf.state_.to_dict()["clauses"] == model.state.to_dict()["clauses"]
    settled = {lang: {f: settled_epoch(s) for f, s in feats.items()}
               for lang, feats in share.items()}  # fmt: skip
    return {"epochs": epochs, "jaccard_with_final": jaccard, "literals_per_clause": literals,
            "signature_share": share, "settled_epoch": settled,
            "reproduces_saved_model": reproduces}  # fmt: skip


def settled_epoch(shares: Sequence[float]) -> int | None:
    """First epoch (1-based) from which the share never drops below half its final value."""
    if not shares or shares[-1] <= 0:
        return None
    half = shares[-1] / 2
    below = [i for i, v in enumerate(shares) if v < half]
    return (below[-1] + 2) if below else 1


# ------------------------------------------------------------------ full inspection


def rules_match_model(rules: list[Rule], state: TMState, x: np.ndarray) -> bool:
    """Independent check: every rule, evaluated from its literal list, fires exactly where the
    model's own matrix code says the clause fires."""
    fired = state.clause_outputs(x)
    return all(np.array_equal(evaluate_rule(r, x), fired[:, r.index]) for r in rules)


def signature_matrix(rules: list[Rule], signatures: dict, languages: Sequence[str],
                     n_features: int, per_language: int = 3) -> dict:  # fmt: skip
    """The top `per_language` signature features of every language, and how often each
    language's "for" clauses use them (rows: features, columns: languages)."""
    usage = literal_usage(rules, languages, n_features)
    ids, owner, names = [], [], []
    for lang in languages:
        for f in signatures[lang][:per_language]:
            if f["id"] not in ids:
                ids.append(f["id"])
                owner.append(lang)
                names.append(f["feature"])
    rows = [[float(usage[k, j]) for k in range(len(languages))] for j in ids]
    return {"features": names, "ids": ids, "owner": owner, "usage": rows}


def inspect_model(
    model: TrainedModel,
    texts: Sequence[str],
    labels: Sequence[str],
    other_models: dict[str, TrainedModel] | None = None,
    formation: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Everything the inspector reports, as one JSON-ready dict (schema `codelangtm.clauses/1`).

    `texts`/`labels` must be the training set the model was trained on (statistics describe
    clauses on training data; the formation replay retrains on it)."""
    say = progress or (lambda _: None)
    state, vocabulary = model.state, model.binarizer.vocabulary_
    languages = list(state.classes)
    rules = extract_rules(state, vocabulary)
    x = model.binarizer.transform(texts)
    clause_stats(rules, state, x, labels)
    say(f"{len(rules)} clauses extracted; statistics on {len(labels)} training snippets")
    signatures = signature_features(rules, languages, vocabulary)
    stability = None
    if other_models:
        others, other_scores = {}, {}
        for name, other in other_models.items():
            other_voc = other.binarizer.vocabulary_
            other_rules = extract_rules(other.state, other_voc)
            others[name] = signature_features(other_rules, languages, other_voc)
            other_scores[name] = signature_scores(other_rules, languages, other_voc)
        stability = seed_stability(signatures, others,
                                   signature_scores(rules, languages, vocabulary),
                                   other_scores)  # fmt: skip
        say(f"stability: compared with {len(others)} other seed(s)")
    replay = None
    if formation:
        replay = formation_replay(model, texts, labels, signatures, progress=say)
    return {
        "schema": CLAUSES_SCHEMA,
        "model": {"meta": model.meta, "params": state.params, "n_features": state.n_features,
                  "n_clauses": state.n_clauses},  # fmt: skip
        "n_train": len(labels),
        "languages": languages,
        "checks": {"rules_match_model": rules_match_model(rules, state, x)},
        "summary": language_summary(rules, languages),
        "signatures": signatures,
        "signature_matrix": signature_matrix(rules, signatures, languages, state.n_features),
        "overlap": class_overlap(rules, languages, state.n_features).tolist(),
        "top_clauses": {lang: [r.index for r in top_clauses(rules, lang)] for lang in languages},
        "stability": stability,
        "formation": replay,
        "clauses": [r.to_dict() for r in rules],
    }


# ------------------------------------------------------------------ reports


def clauses_json_text(data: dict) -> str:
    """The sidecar as text: indented like the other sidecars, but one clause per line and every
    list of numbers on one line, so 3,200 clauses stay a readable, diff-friendly file."""
    from .baselines import _json_ready

    ready = _json_ready(data)
    clauses = ready.pop("clauses")
    text = json.dumps({**ready, "clauses": "__CLAUSES__"}, indent=2, ensure_ascii=False)
    # JSON strings never hold a raw newline, so this only matches lists json.dumps broke up.
    number = r"(?:-?[\d.eE+-]+|null)"
    text = re.sub(rf"\[\n\s*({number}(?:,\n\s*{number})*)\n\s*\]",
                  lambda m: "[" + re.sub(r",\n\s*", ", ", m.group(1)) + "]", text)  # fmt: skip
    lines = ",\n    ".join(json.dumps(c, ensure_ascii=False) for c in clauses)
    body = f"[\n    {lines}\n  ]" if clauses else "[]"
    return text.replace('"__CLAUSES__"', body) + "\n"


def _code(text: str) -> str:
    """Inline code that survives a markdown table: `|` escaped, fence longer than any backtick
    run inside, padded when the text starts or ends with a backtick."""
    text = text.replace("|", "\\|")
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def render_clauses_md(data: dict) -> str:
    meta, params = data["model"]["meta"], data["model"]["params"]
    langs = data["languages"]
    canonical = meta.get("canonical", {})
    md = [
        "# Clauses: what the Tsetlin Machine learned",
        "",
        "Generated by `codelangtm clauses`; do not edit by hand. Every clause of the model, with "
        "statistics, is in [clauses.json](clauses.json).",
        "",
        "## Setup",
        "",
        f"- Model: setting `{meta.get('setting', '?')}`, seed {meta.get('seed', '?')}"
        + (f" (official model: {canonical['rule']})" if canonical else ""),
        f"- Hyperparameters: {params.get('n_clauses')} clauses per language, T={params.get('T')}, "
        f"s={params.get('s')}, {params.get('epochs_trained')} epochs; "
        f"{data['model']['n_features']} features, {data['model']['n_clauses']} clauses in total",
        f"- Statistics on the {data['n_train']} training snippets only (the test set is never "
        "used to describe or pick clauses)",
        "- Check: every rule, evaluated directly from its text, fires exactly where the model's "
        "clause fires: " + ("**yes**" if data["checks"]["rules_match_model"] else "**NO**"),
        "",
        "How to read a rule: `has(\"fn \")` means the snippet contains the n-gram `fn ` (quotes "
        "show spaces; `\\n` is a line break, `\\t` a tab); `NOT has(...)` means it does not. A "
        "clause *fires* when every part is true. *Coverage*: share of the language's training "
        "snippets it fires on. *False fires*: share of other languages' snippets it fires on. "
        "*Precision*: of the snippets it fires on, the share in its language.",
        "",
        "## Overview per language",
        "",
        "*Literals per clause*: median (mean, longest) over non-empty clauses. *Narrow*: "
        "non-empty clauses that fire on at most one training snippet.",
        "",
        "| Language | Clauses | For | Against | Empty | Literals per clause "
        "| Distinct literals | Duplicate clauses | Narrow |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lang in langs:
        s = data["summary"][lang]
        md.append(f"| {lang} | {s['clauses']} | {s['for']} | {s['against']} | {s['empty']} | "
                  f"{s['median_literals']:g} ({s['mean_literals']:.1f}, {s['max_literals']}) | "
                  f"{s['distinct_literals']} | {s['duplicate_clauses']} | "
                  f"{s['narrow_clauses']} |")  # fmt: skip
    md += [
        "",
        "## Signature features",
        "",
        "The n-grams each language's \"for\" clauses rely on most, compared with the other "
        "languages (share of the language's \"for\" clauses that include it; lift = that share "
        "divided by the other languages' mean share).",
        "",
        "| Language | Signature features: share of \"for\" clauses (lift) |",
        "| --- | --- |",
    ]
    for lang in langs:
        feats = ", ".join(f"{_code(f['feature'])} {f['share_of_for_clauses']:.0%} "
                          f"(×{f['lift']:.0f})" for f in data["signatures"][lang])  # fmt: skip
        md.append(f"| {lang} | {feats} |")
    if data.get("stability"):
        seeds = list(next(iter(data["stability"].values()))["per_seed"])
        md += [
            "",
            "## Stability across seeds",
            "",
            "Models trained with other seeds (same setting, same data), compared with this one. "
            "*Top-10 overlap*: share of this model's top signature features that are also in "
            "the other model's top 10. *Score correlation*: Pearson correlation of the signature "
            "scores of every n-gram, which near-ties at the edge of a top-10 list cannot flip.",
            "",
            "| Language | Top-10 overlap: mean | "
            + " | ".join(seeds) + " | Score correlation: mean | " + " | ".join(seeds) + " |",
            "| --- | --- | " + " | ".join("---" for _ in seeds) + " | --- | "
            + " | ".join("---" for _ in seeds) + " |",
        ]
        for lang in langs:
            st = data["stability"][lang]
            corr = st.get("score_correlation", {})
            md.append(f"| {lang} | {st['mean']:.0%} | "
                      + " | ".join(f"{st['per_seed'][k]:.0%}" for k in seeds)
                      + f" | {st.get('mean_correlation', float('nan')):.2f} | "
                      + " | ".join(f"{corr.get(k, float('nan')):.2f}" for k in seeds)
                      + " |")  # fmt: skip
    md += [
        "",
        "## Overlap between languages",
        "",
        "Jaccard similarity of the sets of n-grams used by each language's \"for\" clauses "
        "(1 = the same n-grams, 0 = none shared).",
        "",
        "| | " + " | ".join(langs) + " |",
        "| --- | " + " | ".join("---" for _ in langs) + " |",
    ]
    for lang, row in zip(langs, data["overlap"], strict=True):
        md.append(f"| **{lang}** | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    if data.get("formation"):
        fm = data["formation"]
        marks = [e for e in (1, 5, 10, 20, 40, 80) if e < fm["epochs"]] + [fm["epochs"]]
        md += [
            "",
            "## How the rules formed",
            "",
            f"The model was retrained epoch by epoch with the same seed ({fm['epochs']} epochs); "
            "the replay ends at exactly the saved model: "
            + ("**yes**" if fm["reproduces_saved_model"] else "**NO**") + ". *Similarity to "
            "the final model*: Jaccard similarity between the (clause, n-gram) include decisions "
            "so far and the final ones (1 = identical clauses).",
            "",
            "| Epoch | " + " | ".join(str(e) for e in marks) + " |",
            "| --- | " + " | ".join("---" for _ in marks) + " |",
            "| Similarity to the final model | "
            + " | ".join(f"{fm['jaccard_with_final'][e - 1]:.2f}" for e in marks) + " |",
            "| Literals per non-empty clause | "
            + " | ".join(f"{fm['literals_per_clause'][e - 1]:.1f}" for e in marks) + " |",
            "",
            "Signature features: *settled epoch* (from then on, the share of the language's "
            "\"for\" clauses that include the n-gram stays at least half its final share) and "
            "the share after epoch 1 → at the end.",
            "",
            "| Language | Signature feature: settled epoch (share epoch 1 → end) |",
            "| --- | --- |",
        ]
        for lang in langs:
            parts = []
            for f, shares in fm["signature_share"][lang].items():
                e = fm["settled_epoch"][lang][f]
                parts.append(f"{_code(f)} {e if e is not None else 'never'} "
                             f"({shares[0]:.0%} → {shares[-1]:.0%})")  # fmt: skip
            md.append(f"| {lang} | " + ", ".join(parts) + " |")
    clauses = data["clauses"]
    md += ["", f"## Top {TOP_CLAUSES} \"for\" clauses per language", "",
           "Ranked by coverage × precision (clauses that fire often and mostly on their own "
           "language)."]  # fmt: skip
    for lang in langs:
        md += ["", f"### {lang}", "",
               "| # | Weight | Coverage | False fires | Precision | Rule |",
               "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
        for i in data["top_clauses"][lang]:
            c = clauses[i]
            md.append(f"| {i} | {c['weight']} | {c['coverage']:.0%} | "
                      f"{c['false_fire_rate']:.1%} | {c['precision']:.0%} | "
                      f"{_code(c['rule'])} |")  # fmt: skip
    return "\n".join(md) + "\n"


def summary_table(data: dict) -> str:
    rows = [f"{'language':<12}{'empty':>6}{'narrow':>7}{'lits':>6}  signature features"]
    for lang in data["languages"]:
        s = data["summary"][lang]
        feats = " ".join(f["feature"] for f in data["signatures"][lang][:4])
        rows.append(f"{lang:<12}{s['empty']:>6}{s['narrow_clauses']:>7}"
                    f"{s['median_literals']:>6g}  {feats}")  # fmt: skip
    return "\n".join(rows)
