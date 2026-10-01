"""Error analysis (M3 step B6b): where the Tsetlin Machine goes wrong, and why.

Every training snippet is predicted *out of fold*: by a model trained on the other 4 CV folds,
so it never saw the snippet. This is done for the TM (every seed) and the baselines, with the
exact fold pipeline of `evaluate_model` (the per-fold scores must equal the tm-results run). The
test set is not used, so findings here may guide later choices (M4) without leaking test data.

Analyses: confusable pairs (and whether they match the languages' shared n-grams), hard vs
unlucky errors across seeds, TM vs baselines per snippet (McNemar test), and hypotheses for why
other languages fall into Python, each with a verdict rule stated before looking at the result.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import f1_score

from . import ALL_LANGUAGES
from .baselines import load_evaluation_data, make_models
from .config import TMConfig, config_hash
from .labels import embedded_share
from .rules import _code, extract_rules, short_rule, show_term

ERRORS_SCHEMA = "codelangtm.errors/1"
PLANNED_SINK = "python"  # the test set suggested Python attracts errors (B3); checked here
TOP_RULES = 5  # fired clauses kept per error, for the predicted and the true language
ALPHA = 0.05


# ------------------------------------------------------------------ out-of-fold predictions


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    present = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def run_out_of_fold(
    cfg: TMConfig,
    setting: str,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Out-of-fold predictions for every train snippet: baselines and the TM setting per seed.

    Returns a JSON-ready dict with per-snippet facts (reference, label, length, features present,
    predictions), TM class sums split into "for" and "against" votes, the clauses that fired on
    every TM error, and the per-fold macro-F1 of every model (for the consistency check)."""
    from .model import TMLanguageClassifier

    say = progress or (lambda _: None)
    data_dir = Path(cfg.data)
    dataset, folds, _ = load_evaluation_data(data_dir)
    train = dataset["train"]
    texts = [s.text for s in train]
    y = np.asarray([s.language for s in train])
    n = len(train)
    kwargs = cfg.setting(setting).kwargs()
    available = make_models(0)
    base_pred = {name: np.empty(n, dtype=object) for name in cfg.baselines}
    base_f1 = {name: [] for name in cfg.baselines}
    seeds = list(cfg.seeds)
    tm_pred = {s: np.empty(n, dtype=object) for s in seeds}
    tm_f1 = {s: [] for s in seeds}
    classes = sorted(set(y.tolist()))
    votes_for = {s: np.zeros((n, len(classes)), dtype=np.int64) for s in seeds}
    votes_against = {s: np.zeros((n, len(classes)), dtype=np.int64) for s in seeds}
    features_present = np.zeros(n, dtype=np.int64)
    error_rules: dict[str, dict] = {}  # "<snippet>:<seed>" -> fired clauses

    for k in sorted(set(folds.tolist())):
        tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        binarizer = cfg.features.binarizer()
        x_tr = binarizer.fit([texts[i] for i in tr], y[tr]).transform([texts[i] for i in tr])
        x_ho = binarizer.transform([texts[i] for i in ho])
        features_present[ho] = x_ho.sum(axis=1)
        for name in cfg.baselines:
            pred = clone(available[name]).fit(x_tr, y[tr]).predict(x_ho)
            base_pred[name][ho] = pred
            base_f1[name].append(_macro_f1(y[ho], pred))
        for seed in seeds:
            say(f"fold {k}: {setting} seed {seed}")
            state = TMLanguageClassifier(**kwargs, seed=seed).fit(x_tr, y[tr]).state_
            if list(state.classes) != classes:
                raise ValueError("every training fold must contain every language")
            fired = state.clause_outputs(x_ho).astype(np.int64)
            onehot = np.zeros((state.n_clauses, len(classes)), dtype=np.int64)
            onehot[np.arange(state.n_clauses), state.clause_class] = 1
            w = state.weights.astype(np.int64)[:, None] * onehot
            votes_for[seed][ho] = fired @ np.where(w > 0, w, 0)
            votes_against[seed][ho] = -(fired @ np.where(w < 0, w, 0))
            pred = state.predict(x_ho)
            tm_pred[seed][ho] = pred
            tm_f1[seed].append(_macro_f1(y[ho], pred))
            wrong = np.flatnonzero(pred != y[ho])
            if len(wrong):
                rules = extract_rules(state, binarizer.vocabulary_)
                for j in wrong:
                    error_rules[f"{ho[j]}:{seed}"] = _fired_rules(
                        rules, fired[j], pred[j], y[ho[j]], binarizer.vocabulary_
                    )

    snippets = [{
        "id": i, "language": s.language, "repo": s.repo, "path": s.path,
        "lines": [s.start_line, s.end_line], "n_lines": s.end_line - s.start_line + 1,
        "n_chars": len(s.text), "fold": int(folds[i]), "features_present": int(features_present[i]),
        "baselines": {name: str(base_pred[name][i]) for name in cfg.baselines},
        "tm": {str(s_): str(tm_pred[s_][i]) for s_ in seeds},
    } for i, s in enumerate(train)]  # fmt: skip
    return {
        "setting": setting,
        "params": kwargs,
        "seeds": seeds,
        "baselines": list(cfg.baselines),
        "classes": classes,
        "config_hash": config_hash(cfg),
        "snippets": snippets,
        "tm_votes_for": {str(s): votes_for[s].tolist() for s in seeds},
        "tm_votes_against": {str(s): votes_against[s].tolist() for s in seeds},
        "error_rules": error_rules,
        "fold_f1": {"baselines": base_f1, "tm": {str(s): tm_f1[s] for s in seeds}},
    }


def _fired_rules(rules, fired_row, predicted, true, vocabulary) -> dict:
    """The strongest fired clauses of the predicted and the true language, as readable rules,
    plus the n-grams (plain literals) of the predicted language's fired "for" clauses."""
    fired = [r for r in rules if fired_row[r.index] and r.weight]
    out = {}
    for key, lang in (("predicted", predicted), ("true", true)):
        mine = sorted((r for r in fired if r.language == lang), key=lambda r: (-abs(r.weight),
                                                                              r.index))  # fmt: skip
        out[key] = [{"weight": r.weight, "rule": r.text} for r in mine[:TOP_RULES]]
    m = len(vocabulary)
    pulling = [r for r in fired if r.language == predicted and r.weight > 0]
    out["pulling_rules"] = [r.text for r in pulling]
    out["pulling_ngrams"] = sorted({show_term(vocabulary[lit]) for r in pulling
                                    for lit in r.literals if lit < m})  # fmt: skip
    return out


# ------------------------------------------------------------------ analyses (pure)


def consistency(oof: dict, tm_results: dict | None) -> dict:
    """Largest difference between these per-fold scores and the tm-results run's."""
    if not tm_results:
        return {"checked": False}
    diffs = []
    for name, f1 in oof["fold_f1"]["baselines"].items():
        ref = next((b for b in tm_results["baselines"] if b["name"] == name), None)
        if ref:
            diffs += [abs(a - b) for a, b in zip(f1, ref["cv_f1"], strict=True)]
    ref_seeds = {str(s["seed"]): s for s in tm_results["tm"].get(oof["setting"], {})
                 .get("seeds", [])}  # fmt: skip
    for seed, f1 in oof["fold_f1"]["tm"].items():
        if seed in ref_seeds:
            diffs += [abs(a - b) for a, b in zip(f1, ref_seeds[seed]["cv_f1"], strict=True)]
    worst = max(diffs) if diffs else None
    # sidecar floats are rounded to 6 decimals, so equal scores differ by up to 5e-7
    return {"checked": bool(diffs), "compared_scores": len(diffs), "max_abs_diff": worst,
            "matches": bool(diffs) and worst <= 1e-6}  # fmt: skip


def confusion(true: Sequence[str], pred: Sequence[str], langs: Sequence[str]) -> list[list[int]]:
    index = {g: i for i, g in enumerate(langs)}
    out = [[0] * len(langs) for _ in langs]
    for t, p in zip(true, pred, strict=True):
        out[index[t]][index[p]] += 1
    return out


def model_views(oof: dict) -> dict[str, tuple[list[str], list[str]]]:
    """(true labels, predictions) per model; the TM pools all seeds (each snippet once per seed)."""
    snips = oof["snippets"]
    views = {name: ([s["language"] for s in snips], [s["baselines"][name] for s in snips])
             for name in oof["baselines"]}  # fmt: skip
    true, pred = [], []
    for seed in oof["seeds"]:
        true += [s["language"] for s in snips]
        pred += [s["tm"][str(seed)] for s in snips]
    views["tm"] = (true, pred)
    return views


def confusable_pairs(oof: dict, langs: Sequence[str], overlap: dict | None = None) -> dict:
    """Errors per unordered language pair (both directions) for every model; TM counts are
    per seed on average so they compare with a single baseline run. With `overlap` (language ->
    language -> Jaccard, from the clause inspector): Spearman correlation between the TM's pair
    error counts and the pairs' shared n-grams."""
    views = model_views(oof)
    n_seeds = len(oof["seeds"])
    pairs = []
    for i, a in enumerate(langs):
        for b in langs[i + 1:]:
            row = {"pair": [a, b]}
            for name, (true, pred) in views.items():
                ab = sum(t == a and p == b for t, p in zip(true, pred, strict=True))
                ba = sum(t == b and p == a for t, p in zip(true, pred, strict=True))
                scale = n_seeds if name == "tm" else 1
                row[name] = {f"{a}->{b}": ab / scale, f"{b}->{a}": ba / scale,
                             "total": (ab + ba) / scale}  # fmt: skip
            if overlap:
                row["overlap"] = overlap[a][b]
            pairs.append(row)
    pairs.sort(key=lambda r: (-r["tm"]["total"], r["pair"]))
    out = {"pairs": pairs}
    if overlap:
        rho, p = stats.spearmanr([r["overlap"] for r in pairs], [r["tm"]["total"] for r in pairs])
        out["overlap_correlation"] = {"spearman": float(rho), "p": float(p),
                                      "supported": bool(rho > 0 and p < ALPHA)}  # fmt: skip
    return out


def seed_agreement(oof: dict) -> dict:
    """How many seeds get each snippet wrong: 0 (always right) to all (hard)."""
    n_seeds = len(oof["seeds"])
    counts = Counter()
    hard, unlucky = [], []
    for s in oof["snippets"]:
        wrong = sum(p != s["language"] for p in s["tm"].values())
        counts[wrong] += 1
        if wrong == n_seeds:
            hard.append(s["id"])
        elif wrong:
            unlucky.append(s["id"])
    return {"histogram": {str(k): counts.get(k, 0) for k in range(n_seeds + 1)},
            "hard": hard, "sometimes": unlucky}  # fmt: skip


def mcnemar(true: Sequence[str], a: Sequence[str], b: Sequence[str]) -> dict:
    """Exact McNemar test: do models a and b make *different* mistakes on the same snippets?
    Only snippets where exactly one of them is wrong count."""
    only_a = sum(pa != t and pb == t for t, pa, pb in zip(true, a, b, strict=True))
    only_b = sum(pb != t and pa == t for t, pa, pb in zip(true, a, b, strict=True))
    both = sum(pa != t and pb != t for t, pa, pb in zip(true, a, b, strict=True))
    n = only_a + only_b
    p = float(stats.binomtest(min(only_a, only_b), n, 0.5).pvalue) if n else 1.0
    return {"only_a_wrong": only_a, "only_b_wrong": only_b, "both_wrong": both, "p": p}


def majority_vote(oof: dict) -> list[str]:
    """The TM's most common prediction over seeds for each snippet (ties: alphabetical)."""
    out = []
    for s in oof["snippets"]:
        c = Counter(s["tm"].values())
        top = max(c.values())
        out.append(min(k for k, v in c.items() if v == top))
    return out


def agreement(oof: dict) -> dict:
    """TM vs each baseline per seed, and for the TM's majority vote over seeds."""
    true = [s["language"] for s in oof["snippets"]]
    out = {}
    for name in oof["baselines"]:
        base = [s["baselines"][name] for s in oof["snippets"]]
        per_seed = {str(seed): mcnemar(true, [s["tm"][str(seed)] for s in oof["snippets"]], base)
                    for seed in oof["seeds"]}  # fmt: skip
        out[name] = {"per_seed": per_seed, "majority": mcnemar(true, majority_vote(oof), base)}
    return out


# ------------------------------------------------------------------ error sinks


def _instances(oof: dict):
    """(snippet, seed, predicted) for every TM prediction."""
    for seed in oof["seeds"]:
        for s in oof["snippets"]:
            yield s, str(seed), s["tm"][str(seed)]


def inflow(oof: dict) -> dict:
    """Per model: how many errors each language *receives* (wrongly predicted as it), and its
    share of the model's errors. TM counts pool all seeds."""
    out = {}
    for name, (true, pred) in model_views(oof).items():
        wrong = [p for t, p in zip(true, pred, strict=True) if p != t]
        counts = Counter(wrong)
        out[name] = {g: {"count": counts.get(g, 0),
                         "share": counts.get(g, 0) / len(wrong) if wrong else 0.0}
                     for g in oof["classes"]}  # fmt: skip
    return out


def top_receiver(oof: dict) -> str:
    """The language that receives the most TM errors (ties: alphabetical)."""
    tm = inflow(oof)["tm"]
    return min(tm, key=lambda g: (-tm[g]["count"], g))


def sink_probes(oof: dict, sink: str, signature_df: dict | None = None) -> dict:
    """Hypotheses for why other languages' snippets fall into `sink`. Each verdict rule is
    fixed in advance (see the report); the numbers are reported whatever the verdict."""
    classes = oof["classes"]
    errors = [(s, seed, p) for s, seed, p in _instances(oof) if p != s["language"]]
    into = [(s, seed, p) for s, seed, p in errors if p == sink]
    other = [(s, seed, p) for s, seed, p in errors if p != sink]
    out = {"name": sink, "errors": len(errors), "into_sink": len(into)}

    # H1: the sink is real: the TM sends it more of its errors than chance (errors spread evenly
    # over the other languages) and than every baseline does.
    flows = inflow(oof)
    shares = {name: flows[name][sink]["share"] for name in ("tm", *oof["baselines"])}
    chance = 1 / (len(classes) - 1)
    out["h1_share_of_errors"] = {
        "shares": shares, "chance": chance,
        "supported": shares["tm"] > chance and all(shares["tm"] > v for k, v in shares.items()
                                                    if k != "tm"),
    }  # fmt: skip

    # H2: snippets that fall into the sink carry little evidence (few features present).
    into_ids = {s["id"] for s, _, _ in into}
    right_ids = [s["id"] for s in oof["snippets"] if s["language"] != sink
                 and all(p == s["language"] for p in s["tm"].values())]  # fmt: skip
    feats = {s["id"]: s["features_present"] for s in oof["snippets"]}
    lines = {s["id"]: s["n_lines"] for s in oof["snippets"]}
    a = [feats[i] for i in sorted(into_ids)]
    b = [feats[i] for i in right_ids]
    h2 = {"into_sink_snippets": len(a), "always_right_snippets": len(b)}
    if a and b:
        test = stats.mannwhitneyu(a, b, alternative="less")
        h2 |= {"features_median": [statistics.median(a), statistics.median(b)],
               "lines_median": [statistics.median(lines[i] for i in sorted(into_ids)),
                                statistics.median(lines[i] for i in right_ids)],
               "p": float(test.pvalue), "supported": bool(test.pvalue < ALPHA)}  # fmt: skip
    out["h2_little_evidence"] = h2

    # H3: errors into the sink are narrow wins (the sink edges out the true language).
    def margin(s, seed, p):
        sums = _sums(oof, seed, s["id"])
        return sums[classes.index(p)] - sums[classes.index(s["language"])]

    m_into = [margin(*e) for e in into]
    m_other = [margin(*e) for e in other]
    h3 = {"ties": sum(m == 0 for m in m_into + m_other)}
    if m_into and m_other:
        test = stats.mannwhitneyu(m_into, m_other, alternative="less")
        h3 |= {"margin_median": [statistics.median(m_into), statistics.median(m_other)],
               "p": float(test.pvalue), "supported": bool(test.pvalue < ALPHA)}  # fmt: skip
    out["h3_narrow_wins"] = h3

    # H4: the sink rejects foreign code weakly: its class sum on other languages' snippets is
    # the highest of all languages (split into "for" and "against" votes).
    foreign = {}
    for k, lang in enumerate(classes):
        f, g = [], []
        for s, seed, _ in _instances(oof):
            if s["language"] != lang:
                f.append(oof["tm_votes_for"][seed][s["id"]][k])
                g.append(oof["tm_votes_against"][seed][s["id"]][k])
        foreign[lang] = {"for": statistics.fmean(f), "against": statistics.fmean(g),
                         "sum": statistics.fmean(f) - statistics.fmean(g)}  # fmt: skip
    highest = max(classes, key=lambda c: foreign[c]["sum"])
    out["h4_weak_rejection"] = {"foreign_votes": foreign, "highest": highest,
                                "supported": highest == sink}  # fmt: skip

    # H5: which rules and n-grams pull snippets into the sink (descriptive, no verdict).
    rule_count, gram_count = Counter(), Counter()
    for s, seed, _ in into:
        fired = oof["error_rules"].get(f"{s['id']}:{seed}", {})
        rule_count.update(set(fired.get("pulling_rules", [])))
        gram_count.update(fired.get("pulling_ngrams", []))
    def ranked(counter, n):  # count, then text: stable across runs (ties are common)
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    out["h5_pulling"] = {"rules": [[short_rule(r), c] for r, c in ranked(rule_count, 10)],
                         "ngrams": [[g, c] for g, c in ranked(gram_count, 15)]}  # fmt: skip

    # H6: the sink's signature n-grams are generic: they occur in other languages' code more
    # often than any other language's signature n-grams do.
    if signature_df:
        means = {lang: statistics.fmean(v["foreign_df"] for v in feats_)
                 for lang, feats_ in signature_df.items() if feats_}  # fmt: skip
        top = max(means, key=means.get)
        out["h6_generic_signatures"] = {"foreign_df_mean": means, "per_feature": signature_df,
                                        "highest": top, "supported": top == sink}  # fmt: skip
    return out


def _sums(oof: dict, seed: str, i: int) -> list[int]:
    f, g = oof["tm_votes_for"][seed][i], oof["tm_votes_against"][seed][i]
    return [a - b for a, b in zip(f, g, strict=True)]


def embedded_probe(oof: dict, texts: dict[int, str], host: str = "html",
                   guest: str = "javascript") -> dict:  # fmt: skip
    """H7: `host` snippets that the TM calls `guest` (on any seed) are mostly embedded code:
    their share of lines inside <script>/<style> is higher than for always-right `host`
    snippets (Mann-Whitney, one-sided)."""
    hosts = [s for s in oof["snippets"] if s["language"] == host]
    to_guest = [s for s in hosts if any(p == guest for p in s["tm"].values())]
    right = [s for s in hosts if all(p == host for p in s["tm"].values())]
    share = {s["id"]: embedded_share(texts[s["id"]]) for s in hosts}
    out = {"host": host, "guest": guest, "snippets": len(hosts), "to_guest": len(to_guest),
           "always_right": len(right),
           # per host snippet: [embedded share, seeds that call it `guest`]
           "points": [[share[s["id"]], sum(p == guest for p in s["tm"].values())]
                      for s in hosts]}  # fmt: skip
    if to_guest and right:
        a, b = [share[s["id"]] for s in to_guest], [share[s["id"]] for s in right]
        test = stats.mannwhitneyu(a, b, alternative="greater")
        mostly = [s for s in hosts if share[s["id"]] > 0.5]
        rest = [s for s in hosts if share[s["id"]] <= 0.5]

        def rate(group):
            wrong = sum(p == guest for s in group for p in s["tm"].values())
            return wrong / (len(group) * len(oof["seeds"])) if group else 0.0

        out |= {"embedded_median": [statistics.median(a), statistics.median(b)],
                "p": float(test.pvalue), "supported": bool(test.pvalue < ALPHA),
                "mostly_embedded": {"snippets": len(mostly), "to_guest_rate": rate(mostly)},
                "rest": {"snippets": len(rest), "to_guest_rate": rate(rest)}}  # fmt: skip
    return out


def signature_document_frequency(signatures: dict, vocabulary: Sequence[str],
                                 train: Sequence, top: int = 3) -> dict:  # fmt: skip
    """For each language's top signature n-grams (clause inspector, official model): the share
    of the language's own and of other languages' training snippets that contain the n-gram."""
    out = {}
    for lang, feats in signatures.items():
        rows = []
        for f in feats[:top]:
            term = vocabulary[f["id"]]
            own = [term in s.text for s in train if s.language == lang]
            other = [term in s.text for s in train if s.language != lang]
            rows.append({"feature": f["feature"], "own_df": statistics.fmean(own),
                         "foreign_df": statistics.fmean(other)})  # fmt: skip
        out[lang] = rows
    return out


# ------------------------------------------------------------------ assembled result


def analyse(oof: dict, tm_results: dict | None = None, overlap: dict | None = None,
            signature_df: dict | None = None, texts: dict[int, str] | None = None,
            languages: Sequence[str] | None = None) -> dict:  # fmt: skip
    langs = [g for g in (languages or ALL_LANGUAGES) if g in oof["classes"]]
    flows = inflow(oof)
    chance = 1 / (len(oof["classes"]) - 1)
    planned = None
    if PLANNED_SINK in oof["classes"]:
        share = flows["tm"][PLANNED_SINK]["share"]
        planned = {"name": PLANNED_SINK, "count": flows["tm"][PLANNED_SINK]["count"],
                   "share": share, "chance": chance, "replicated": share > chance}  # fmt: skip
    embedded = None
    if texts and {"html", "javascript"} <= set(oof["classes"]):
        embedded = embedded_probe(oof, texts)
    views = model_views(oof)
    per_model = {}
    for name, (true, pred) in views.items():
        wrong = sum(t != p for t, p in zip(true, pred, strict=True))
        scale = len(oof["seeds"]) if name == "tm" else 1
        per_model[name] = {"confusion": confusion(true, pred, langs), "errors": wrong / scale,
                           "error_rate": wrong / len(true)}  # fmt: skip
    return {
        "schema": ERRORS_SCHEMA,
        "setting": oof["setting"],
        "params": oof["params"],
        "seeds": oof["seeds"],
        "baselines": oof["baselines"],
        "config_hash": oof["config_hash"],
        "n_snippets": len(oof["snippets"]),
        "languages": langs,
        "consistency": consistency(oof, tm_results),
        "models": per_model,
        "pairs": confusable_pairs(oof, langs, overlap),
        "seed_agreement": seed_agreement(oof),
        "agreement": agreement(oof),
        "inflow": flows,
        "planned_sink": planned,
        "sink": sink_probes(oof, top_receiver(oof), signature_df),
        "embedded": embedded,
        "snippets": oof["snippets"],
        # the strongest fired clauses per error (the full pulling lists only feed H5)
        "error_rules": {k: {"predicted": v["predicted"], "true": v["true"]}
                        for k, v in oof["error_rules"].items()},  # fmt: skip
    }


# ------------------------------------------------------------------ reports


def _yes(flag: bool | None) -> str:
    return "n/a" if flag is None else ("**supported**" if flag else "**not supported**")


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _rule(columns: int) -> str:
    """Markdown table separator row."""
    return "|" + " --- |" * columns


def _ref(s: dict) -> str:
    return f"`{s['repo']}` `{s['path']}` L{s['lines'][0]}-{s['lines'][1]}"


def render_errors_md(data: dict) -> str:
    langs, seeds = data["languages"], data["seeds"]
    tm_name = f"{data['setting']} (TM)"
    names = {"tm": tm_name, **{b: b.replace("_", " ") for b in data["baselines"]}}
    c = data["consistency"]
    params = ", ".join(f"{k}={v}" for k, v in data["params"].items())
    md = [
        "# Errors: where the TM goes wrong, and why",
        "",
        "Generated by `codelangtm errors`; do not edit by hand. Machine-readable version: "
        "[errors.json](errors.json).",
        "",
        "## Setup",
        "",
        f"- Every one of the {data['n_snippets']} training snippets is predicted *out of fold*: "
        "by a model trained on the other 4 CV folds, which never saw it. The test set is not "
        "used.",
        f"- Models: `{data['setting']}` ({params}) with seeds {', '.join(map(str, seeds))}; "
        "baselines " + ", ".join(names[b] for b in data["baselines"]) + ", on the same folds.",
        "- Consistency check: the per-fold macro-F1 of every model equals the `tm-results` run: "
        + (f"**yes** ({c['compared_scores']} scores compared)" if c.get("matches")
           else ("**NO**" if c.get("checked") else "not checked")),
        "- Code of the misclassified snippets is not committed; `codelangtm errors` writes it, "
        "with the clauses that fired, to a local review file (`data/processed/errors-review.md`).",
        "",
        "## Errors per model",
        "",
        "TM counts are per seed on average (each snippet is predicted once per seed).",
        "",
        "| Model | Errors | Error rate |",
        "| --- | --- | --- |",
    ]  # fmt: skip
    for key in ["tm", *data["baselines"]]:
        m = data["models"][key]
        md.append(f"| {names[key]} | {m['errors']:.1f} | {m['error_rate']:.1%} |")
    tm = data["models"]["tm"]
    md += ["", f"### Confusion, {tm_name}, all {len(seeds)} seeds pooled", "",
           "Rows: true language; columns: predicted.", "",
           "| | " + " | ".join(langs) + " |", _rule(len(langs) + 1)]  # fmt: skip
    for lang, row in zip(langs, tm["confusion"], strict=True):
        md.append(f"| **{lang}** | " + " | ".join(str(v) if v else "·" for v in row) + " |")

    pairs = data["pairs"]
    md += ["", "## Confusable pairs", "",
           "Errors in both directions (TM: mean per seed). *Overlap*: Jaccard similarity of the "
           "n-grams the two languages' \"for\" clauses use (official model, "
           "[clauses.md](clauses.md)).", "",
           "| Pair | TM | " + " | ".join(names[b] for b in data["baselines"])
           + " | Direction (TM) | Overlap |",
           _rule(len(data["baselines"]) + 4)]  # fmt: skip
    for r in pairs["pairs"]:
        if not (r["tm"]["total"] or any(r[b]["total"] for b in data["baselines"])):
            continue
        a, b = r["pair"]
        direction = f"{a}→{b} {r['tm'][f'{a}->{b}']:.1f}, {b}→{a} {r['tm'][f'{b}->{a}']:.1f}"
        md.append(f"| {a}-{b} | {r['tm']['total']:.1f} | "
                  + " | ".join(f"{r[x]['total']:.0f}" for x in data["baselines"])
                  + f" | {direction} | {r.get('overlap', float('nan')):.2f} |")  # fmt: skip
    if "overlap_correlation" in pairs:
        oc = pairs["overlap_correlation"]
        md += ["", f"Do languages that share more n-grams get confused more? Spearman correlation "
               f"over all {len(pairs['pairs'])} pairs: ρ = {oc['spearman']:.2f}, "
               f"p = {_p(oc['p'])}: "
               f"{_yes(oc['supported'])} (rule: ρ > 0 and p < {ALPHA})."]  # fmt: skip

    sa = data["seed_agreement"]
    hist = sa["histogram"]
    md += ["", "## Hard errors and unlucky errors", "",
           "How many of the seeds get each snippet wrong.", "",
           "| Seeds wrong | " + " | ".join(hist) + " |",
           "| --- | " + " | ".join("---" for _ in hist) + " |",
           "| Snippets | " + " | ".join(str(v) for v in hist.values()) + " |", "",
           f"{len(sa['hard'])} snippets are wrong for every seed (*hard*); "
           f"{len(sa['sometimes'])} are wrong for some seeds only (*unlucky*: the seed "
           "decides)."]  # fmt: skip
    snips = {s["id"]: s for s in data["snippets"]}
    if sa["hard"]:
        md += ["", "Hard snippets (code in the local review file):", "",
               "| Snippet | True | TM predicts | " + " | ".join(names[b] for b in data["baselines"])
               + " | Features present |",
               _rule(len(data["baselines"]) + 4)]  # fmt: skip
        for i in sa["hard"]:
            s = snips[i]
            preds = Counter(s["tm"].values())
            tm_text = ", ".join(f"{p} ×{n}" for p, n in preds.most_common())
            md.append(f"| {_ref(s)} | {s['language']} | {tm_text} | "
                      + " | ".join(s["baselines"][b] for b in data["baselines"])
                      + f" | {s['features_present']} |")  # fmt: skip

    md += ["", "## TM vs baselines, snippet by snippet", "",
           "Exact McNemar test: among snippets where exactly one of the two models is wrong, is "
           "one model wrong significantly more often? *Majority*: the TM's most common prediction "
           "over the seeds.", "",
           "| Baseline | TM run | Both wrong | Only TM wrong | Only baseline wrong | p |",
           "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for b in data["baselines"]:
        ag = data["agreement"][b]
        rows = [*((f"seed {k}", v) for k, v in ag["per_seed"].items()),
                ("majority", ag["majority"])]
        for label, t in rows:
            md.append(f"| {names[b]} | {label} | {t['both_wrong']} | {t['only_a_wrong']} | "
                      f"{t['only_b_wrong']} | {_p(t['p'])} |")  # fmt: skip

    ps = data.get("planned_sink")
    flows = data["inflow"]
    md += ["", "## Which languages collect the errors", "",
           "Errors *received*: snippets of other languages wrongly predicted as this language "
           "(TM: all seeds pooled; share of the model's errors).", "",
           "| Language | " + " | ".join(names[k] for k in flows) + " |",
           "| --- | " + " | ".join("---" for _ in flows) + " |"]  # fmt: skip
    for lang in langs:
        md.append(f"| {lang} | " + " | ".join(f"{flows[k][lang]['count']} "
                                              f"({flows[k][lang]['share']:.0%})"
                                              for k in flows) + " |")  # fmt: skip
    if ps:
        md += ["", f"**The {ps['name']} sink seen on the test set does not replicate here:** "
               f"{ps['name']} receives {ps['count']} of the TM's errors ({ps['share']:.0%}), "
               + ("above" if ps["replicated"] else "below")
               + f" the {ps['chance']:.0%} expected if errors spread evenly. On CV the language "
               f"that collects most errors is {data['sink']['name']}, so the hypotheses below are "
               "tested for it."]  # fmt: skip

    sk = data["sink"]
    sink = sk["name"]
    h1, h2, h3, h4 = (sk["h1_share_of_errors"], sk["h2_little_evidence"], sk["h3_narrow_wins"],
                      sk["h4_weak_rejection"])  # fmt: skip
    md += ["", f"## Why do snippets fall into {sink}?", "",
           f"{sk['into_sink']} of the TM's {sk['errors']} errors (all seeds) predict {sink}. "
           "Each hypothesis has a verdict rule fixed before the run, except H1's chance "
           "condition, added after seeing the data (see the note under the table).", "",
           "| # | Hypothesis | Rule | Result | Verdict |", _rule(5)]  # fmt: skip
    shares = ", ".join(f"{names[k]} {v:.0%}" for k, v in h1["shares"].items())
    md.append(f"| H1 | {sink} really attracts the TM's errors | TM share above chance and above "
              f"every baseline's | {shares} (chance {h1['chance']:.0%}) | "
              f"{_yes(h1['supported'])} |")  # fmt: skip
    if "p" in h2:
        md.append(f"| H2 | Snippets that fall into {sink} carry little evidence | fewer features "
                  f"present than always-right snippets, Mann-Whitney p < {ALPHA} | features "
                  f"(median) {h2['features_median'][0]:g} vs {h2['features_median'][1]:g}; lines "
                  f"{h2['lines_median'][0]:g} vs {h2['lines_median'][1]:g}; p = {_p(h2['p'])} | "
                  f"{_yes(h2['supported'])} |")  # fmt: skip
    if "p" in h3:
        md.append(f"| H3 | Errors into {sink} are narrow wins | vote margin (predicted − true) "
                  f"smaller than other errors', Mann-Whitney p < {ALPHA} | median "
                  f"{h3['margin_median'][0]:g} vs {h3['margin_median'][1]:g} votes; p = "
                  f"{_p(h3['p'])} | {_yes(h3['supported'])} |")  # fmt: skip
    fv = h4["foreign_votes"]
    md.append(f"| H4 | {sink} rejects other languages' code weakly | its mean vote total on "
              f"other languages' snippets is the highest of all languages | {sink} "
              f"{fv[sink]['sum']:.1f}; highest: {h4['highest']} {fv[h4['highest']]['sum']:.1f} | "
              f"{_yes(h4['supported'])} |")  # fmt: skip
    if "h6_generic_signatures" in sk:
        h6 = sk["h6_generic_signatures"]
        md.append(f"| H6 | {sink}'s signature n-grams are generic | its top-3 signature n-grams "
                  "occur in other languages' snippets more often than any other language's do | "
                  f"{sink} {h6['foreign_df_mean'][sink]:.0%}; highest: {h6['highest']} "
                  f"{h6['foreign_df_mean'][h6['highest']]:.0%} | "
                  f"{_yes(h6['supported'])} |")  # fmt: skip
    em = data.get("embedded")
    if em and "p" in em:
        md.append(f"| H7 | {em['host']} snippets that the TM calls {em['guest']} are mostly "
                  f"embedded code | larger share of lines inside `<script>`/`<style>` than "
                  f"always-right {em['host']} snippets, Mann-Whitney p < {ALPHA} | median "
                  f"{em['embedded_median'][0]:.0%} vs {em['embedded_median'][1]:.0%}; p = "
                  f"{_p(em['p'])} | {_yes(em['supported'])} |")  # fmt: skip
    md += ["", "*Note on H1:* the first version of the rule only compared the TM with the "
           "baselines. For Python it passed although Python received fewer errors than chance, "
           "so the chance condition was added after seeing the data "
           "([issues-and-fixes](issues-and-fixes.md) M11)."]  # fmt: skip
    if em and "p" in em:
        md += ["", f"H7 in numbers: of {em['snippets']} {em['host']} snippets, "
               f"{em['mostly_embedded']['snippets']} are more than half `<script>`/`<style>`; "
               f"they are called {em['guest']} in {em['mostly_embedded']['to_guest_rate']:.0%} of "
               f"predictions, against {em['rest']['to_guest_rate']:.0%} for the other "
               f"{em['rest']['snippets']}. Whether such snippets should be labelled {em['host']} "
               "at all is the embedded-language policy planned for Stage B."]  # fmt: skip
    md += ["", "Mean votes each language receives on *other* languages' snippets (all seeds):", "",
           "| Language | For | Against | Total |", "| --- | --- | --- | --- |"]  # fmt: skip
    for lang in langs:
        v = fv[lang]
        md.append(f"| {lang} | {v['for']:.0f} | {v['against']:.0f} | {v['sum']:.0f} |")
    if "h6_generic_signatures" in sk:
        md += ["", "Top-3 signature n-grams: share of the language's own training snippets and of "
               "other languages' snippets that contain them:", "",
               "| Language | Signature n-gram: own → others |", "| --- | --- |"]  # fmt: skip
        for lang in langs:
            rows = sk["h6_generic_signatures"]["per_feature"].get(lang, [])
            md.append(f"| {lang} | " + ", ".join(
                f"{_code(r['feature'])} {r['own_df']:.0%} → {r['foreign_df']:.0%}"
                for r in rows) + " |")  # fmt: skip
    h5 = sk["h5_pulling"]
    md += ["", f"**H5 (descriptive):** the {sink} \"for\" rules that fired most often on snippets "
           f"wrongly predicted as {sink} (count of errors they fired on):", ""]  # fmt: skip
    md += [f"- {n} × {_code(r)}" for r, n in h5["rules"]]
    md += ["", "Most common n-grams in those rules: "
           + ", ".join(f"{_code(g)} ({n})" for g, n in h5["ngrams"]) + "."]  # fmt: skip
    return "\n".join(md) + "\n"


def render_review_md(data: dict, texts: dict[int, str]) -> str:
    """Local review file (gitignored data/): every TM error snippet with its code and the
    clauses that fired, hardest first. Contains third-party code: never commit it."""
    n_seeds = len(data["seeds"])
    rows = []
    for s in data["snippets"]:
        wrong = [seed for seed, p in s["tm"].items() if p != s["language"]]
        if wrong:
            rows.append((len(wrong), s, wrong))
    rows.sort(key=lambda r: (-r[0], r[1]["language"], r[1]["id"]))
    md = ["# Error review (local, do not commit)", "",
          f"{len(rows)} snippets the TM gets wrong on at least one of {n_seeds} seeds "
          "(out-of-fold). Hardest first.", ""]  # fmt: skip
    for n, s, wrong in rows:
        seed = wrong[0]
        fired = data["error_rules"].get(f"{s['id']}:{seed}", {})
        preds = ", ".join(f"seed {k}: {p}" for k, p in s["tm"].items())
        base = ", ".join(f"{k}: {p}" for k, p in s["baselines"].items())
        md += [f"## #{s['id']} {s['language']} ({n}/{n_seeds} seeds wrong)", "",
               f"- {_ref(s)}, {s['n_lines']} lines, {s['features_present']} features present",
               f"- TM: {preds}", f"- baselines: {base}", "",
               f"Strongest fired clauses, seed {seed}:", ""]  # fmt: skip
        for key in ("predicted", "true"):
            for r in fired.get(key, []):
                md.append(f"- {key} `{r['weight']:+d}` `{short_rule(r['rule'])}`")
        md += ["", "~~~~", texts[s["id"]].rstrip("\n"), "~~~~", ""]
    return "\n".join(md) + "\n"


def summary_table(data: dict) -> str:
    rows = [f"{'model':<24}{'errors':>8}{'rate':>8}"]
    for key, m in data["models"].items():
        rows.append(f"{key:<24}{m['errors']:>8.1f}{m['error_rate']:>8.1%}")
    sk = data["sink"]
    rows.append(f"\nTM errors into {sk['name']} (top receiver): {sk['into_sink']} of "
                f"{sk['errors']}")
    for key in ("h1_share_of_errors", "h2_little_evidence", "h3_narrow_wins",
                "h4_weak_rejection", "h6_generic_signatures"):  # fmt: skip
        if key in sk and "supported" in sk[key]:
            rows.append(f"  {key}: {'supported' if sk[key]['supported'] else 'not supported'}")
    return "\n".join(rows)
