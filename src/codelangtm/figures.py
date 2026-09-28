"""Figures for the report, drawn from the JSON sidecars (`results.json`, `ablations.json`).

matplotlib is an optional extra (`uv sync --extra viz`) and is imported only when a figure is
drawn, so the rest of the package imports without it. Output is deterministic: a fixed style and
font (DejaVu Sans ships with matplotlib), fixed size and dpi, and no PNG metadata, so drawing the
same sidecar twice gives byte-identical files and clean git diffs.

Colors follow the project's validated palette: three categorical slots (checked for colour-vision
deficiency, all pairs) for series, one blue ramp for magnitudes. Series are also told apart by
marker shape and a legend, never by colour alone; the generated markdown tables are the exact-value
view of every figure.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

INSTALL_HINT = "figures need matplotlib: run `uv sync --extra viz`"

# Validated palette (light surface). Series slots in fixed order; never cycled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "D")
BLUE_RAMP = ("#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b")  # fmt: skip
DPI = 150
TARGET_F1 = 0.96

# Colour follows the entity: each selection method keeps its slot in every figure.
SELECTION_ORDER = ("frequency", "chi2", "class_balanced")


def _mpl():
    """Import matplotlib lazily with the non-interactive Agg backend; clear error if missing."""
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    return matplotlib, Figure


def _style() -> dict:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "svg.hashsalt": "codelangtm",
        "path.simplify": False,
    }


def save_png(fig: object, path: str | Path) -> Path:
    """Write a figure as PNG without metadata (no software/version/date stamps)."""
    matplotlib, _ = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    # Style applies here too: some artists (tick labels) are only created at draw time.
    with matplotlib.rc_context(_style()):
        fig.savefig(buf, format="png", dpi=DPI, metadata={"Software": None})
    path.write_bytes(buf.getvalue())
    return path


def _new_figure(width: float, height: float, ncols: int = 1, sharey: bool = False):
    matplotlib, Figure = _mpl()
    with matplotlib.rc_context(_style()):
        fig = Figure(figsize=(width, height), layout="constrained")
        axes = fig.subplots(1, ncols, sharey=sharey, squeeze=False)[0]
    return fig, list(axes)


def _draw(fn: Callable[[], object]) -> object:
    """Run a drawing function under the project style (rcParams apply at draw time too)."""
    matplotlib, _ = _mpl()
    with matplotlib.rc_context(_style()):
        return fn()


def _label(name: str) -> str:
    return name.replace("_", " ")


def _num(v: float) -> str:
    """3 significant digits, but whole numbers with thousands separators from 100 up."""
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


# ---------------------------------------------------------------- results.json figures


def dataset_composition(results: dict) -> object:
    """Snippets per language, train and test stacked, with repo counts at the bar end."""

    def draw():
        comp = results["meta"]["composition"]
        langs = list(results["languages"])[::-1]  # first language on top
        train = [comp["train"].get(lang, {}).get("snippets", 0) for lang in langs]
        test = [comp["test"].get(lang, {}).get("snippets", 0) for lang in langs]
        fig, (ax,) = _new_figure(6.4, 3.4)
        y = range(len(langs))
        ax.barh(y, train, height=0.6, color=SERIES[0], label="train", edgecolor=SURFACE,
                linewidth=1)  # fmt: skip
        ax.barh(y, test, left=train, height=0.6, color=SERIES[1], label="test",
                edgecolor=SURFACE, linewidth=1)  # fmt: skip
        for i, lang in enumerate(langs):
            repos = sum(comp[s].get(lang, {}).get("repos", 0) for s in ("train", "test"))
            ax.text(train[i] + test[i] + 2, i, f"{train[i] + test[i]} snippets, {repos} repos",
                    va="center", fontsize=8, color=INK_2)  # fmt: skip
        ax.set_yticks(list(y), langs)
        ax.set_xlim(0, max(a + b for a, b in zip(train, test, strict=True)) * 1.45)
        ax.set_xlabel("snippets (20-50 line windows)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Dataset: {results['meta']['n_train']} train, "
                     f"{results['meta']['n_test']} test snippets")  # fmt: skip
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2)
        return fig

    return _draw(draw)


def baseline_comparison(results: dict) -> object:
    """Per model: CV mean ± std, repeated test mean ± std and the single test score."""

    def draw():
        models = results["models"][::-1]
        names = [_label(m["name"]) for m in models]
        fig, (ax,) = _new_figure(6.4, 0.5 * len(models) + 1.4)
        y = list(range(len(models)))
        cv = [m["cv_mean"] for m in models]
        ax.errorbar(cv, [i + 0.15 for i in y], xerr=[m["cv_std"] for m in models], fmt=MARKERS[0],
                    color=SERIES[0], ms=6, capsize=3, lw=1.5, label="CV, mean ± std")  # fmt: skip
        rep = [(i, m) for i, m in zip(y, models, strict=True) if m["repeat_f1"]]
        if rep:
            ax.errorbar([m["repeat_mean"] for _, m in rep], [i - 0.15 for i, _ in rep],
                        xerr=[m["repeat_std"] for _, m in rep], fmt=MARKERS[1], color=SERIES[1],
                        ms=6, capsize=3, lw=1.5, label="repeated test, mean ± std")  # fmt: skip
        ax.scatter([m["test_f1"] for m in models], y, marker="|", s=120, color=INK_2, zorder=3,
                   label="single test")  # fmt: skip
        ax.axvline(TARGET_F1, color=MUTED, lw=1)
        ax.text(TARGET_F1, len(models) - 0.45, f" target {TARGET_F1}", color=MUTED, fontsize=8,
                va="bottom")  # fmt: skip
        ax.set_yticks(y, names)
        ax.set_ylim(-0.6, len(models) - 0.1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("macro-F1")
        best = _label(results["best_by_cv"])
        ax.set_title(f"Baselines: macro-F1 (best by CV: {best})")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncols=3)
        return fig

    return _draw(draw)


def _ramp():
    matplotlib, _ = _mpl()
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("codelangtm_blue", BLUE_RAMP)


def _heatmap(ax, values, rows, cols, vmin, vmax, fmt, blank_zero=False):
    cmap = _ramp()
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # A static PNG has no hover: the value sits in the cell (ink flips on dark cells).
    span = (vmax - vmin) or 1
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if blank_zero and v == 0:
                continue
            dark = (v - vmin) / span > 0.55
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark else INK)  # fmt: skip
    return image


def per_language_f1(results: dict) -> object:
    """Test F1 per language (columns) for every model (rows)."""

    def draw():
        langs = results["languages"]
        models = results["models"]
        values = [[m["per_language_f1"][lang] for lang in langs] for m in models]
        low = min(min(row) for row in values)
        vmin = min(0.5, round(low - 0.05, 1))
        fig, (ax,) = _new_figure(6.4, 0.42 * len(models) + 1.5)
        image = _heatmap(ax, values, [_label(m["name"]) for m in models], langs, vmin, 1.0, ".2f")
        fig.colorbar(image, ax=ax, shrink=0.8, label="test F1").outline.set_visible(False)
        ax.set_title("Test F1 per language")
        return fig

    return _draw(draw)


def confusion(results: dict, model: str | None = None) -> object:
    """Confusion matrix of one model (default: best by CV); colour = share of the true row."""

    def draw():
        name = model or results["best_by_cv"]
        entry = next(m for m in results["models"] if m["name"] == name)
        langs = results["languages"]
        counts = entry["confusion"]
        shares = [[c / (sum(row) or 1) for c in row] for row in counts]
        fig, (ax,) = _new_figure(5.2, 4.4)
        image = _heatmap(ax, shares, langs, langs, 0.0, 1.0, ".0%", blank_zero=True)
        # Replace the percentage text with counts (colour stays the row share).
        for text in list(ax.texts):
            text.remove()
        for i, row in enumerate(counts):
            for j, c in enumerate(row):
                if c:
                    ax.text(j, i, str(c), ha="center", va="center", fontsize=8,
                            color="#ffffff" if shares[i][j] > 0.55 else INK)  # fmt: skip
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        bar = fig.colorbar(image, ax=ax, shrink=0.8, label="share of true language")
        bar.outline.set_visible(False)
        ax.set_title(f"Confusion matrix (test): {_label(name)}")
        return fig

    return _draw(draw)


RESOURCE_PANELS = (
    ("fit_seconds", "fit time (s)", False),
    ("latency_ms", "latency (ms/snippet)", False),
    ("size_kb", "model size (KB, log scale)", True),
)


def resources(results: dict) -> object:
    """Small multiples (one scale each, never a dual axis): fit time, latency, size."""

    def draw():
        models = results["models"][::-1]
        names = [_label(m["name"]) for m in models]
        fig, axes = _new_figure(7.6, 0.4 * len(models) + 1.5, ncols=3, sharey=True)
        for ax, (key, title, log) in zip(axes, RESOURCE_PANELS, strict=True):
            values = [m["resources"][key] for m in models]
            ax.barh(range(len(models)), values, height=0.6, color=SERIES[0])
            if log:
                ax.set_xscale("log")
            for i, v in enumerate(values):
                ax.text(v, i, f" {_num(v)}", va="center", fontsize=7.5, color=INK_2)
            ax.set_xlim(right=max(values) * (8 if log else 1.35))
            ax.set_title(title, fontsize=9)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(range(len(models)), names)
        fig.suptitle("Resources (same measurement for every model)", x=0.01, ha="left",
                     fontweight="bold", fontsize=10, color=INK)  # fmt: skip
        return fig

    return _draw(draw)


# ------------------------------------------------------------- ablations.json figures


def _settings(ablations: dict, studies: Sequence[str]) -> list[dict]:
    """Unique settings of the named studies, in first-seen order (missing studies skipped)."""
    by_name = {s["name"]: s for s in ablations["studies"]}
    ids = []
    for name in studies:
        for i in by_name.get(name, {}).get("settings", []):
            if i not in ids:
                ids.append(i)
    return [ablations["settings"][i] for i in ids]


def _selection_series(settings: list[dict]) -> list[tuple[int, str, list[dict]]]:
    """(slot, selection, settings) for each selection method present, fixed slot per method."""
    out = []
    for slot, method in enumerate(SELECTION_ORDER):
        group = [s for s in settings if s["features"]["selection"] == method]
        if group:
            out.append((slot, method, group))
    return out


def _model_panels(ablations: dict, width: float, height: float):
    models = ablations["models"]
    fig, axes = _new_figure(width, height, ncols=len(models), sharey=True)
    for ax, m in zip(axes, models, strict=True):
        ax.set_title(_label(m), fontsize=9)
    return fig, axes, models


def ablation_vocabulary(
    ablations: dict, studies: Sequence[str] = ("vocabulary_size", "selection_x_vocabulary")
) -> object:
    """CV macro-F1 vs vocabulary size M, one line per selection method, one panel per model."""

    def draw():
        settings = _settings(ablations, studies)
        fig, axes, models = _model_panels(ablations, 7.6, 3.2)
        for ax, m in zip(axes, models, strict=True):
            for slot, method, group in _selection_series(settings):
                group = sorted(group, key=lambda s: s["features"]["n_features"])
                x = [s["features"]["n_features"] for s in group]
                mean = [s["models"][m]["cv_mean"] for s in group]
                std = [s["models"][m]["cv_std"] for s in group]
                ax.fill_between(x, [a - b for a, b in zip(mean, std, strict=True)],
                                [a + b for a, b in zip(mean, std, strict=True)],
                                color=SERIES[slot], alpha=0.12, lw=0)  # fmt: skip
                ax.plot(x, mean, color=SERIES[slot], marker=MARKERS[slot], ms=5,
                        label=_label(method))  # fmt: skip
            ax.set_xscale("log")
            ax.set_xlabel("vocabulary size M (log scale)")
            ax.axhline(TARGET_F1, color=MUTED, lw=1)
        ticks = sorted({s["features"]["n_features"] for s in settings})
        for ax in axes:
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
        axes[0].set_ylabel("CV macro-F1 (band: ± std over folds)")
        axes[0].legend(title="feature selection", loc="lower right", title_fontsize=8)
        fig.suptitle("Ablation: vocabulary size and feature selection", x=0.01, ha="left",
                     fontweight="bold", fontsize=10, color=INK)  # fmt: skip
        return fig

    return _draw(draw)


def ablation_ngrams(ablations: dict, study: str = "selection_x_ngrams") -> object:
    """CV macro-F1 per n-gram size set, grouped by selection method, one panel per model."""

    def draw():
        settings = _settings(ablations, [study])
        sizes = []
        for s in settings:
            key = "+".join(str(n) for n in s["features"]["ngram_sizes"])
            if key not in sizes:
                sizes.append(key)
        series = _selection_series(settings)
        fig, axes, models = _model_panels(ablations, 7.6, 3.2)
        width = 0.6 / max(len(series), 1)
        for ax, m in zip(axes, models, strict=True):
            for k, (slot, method, group) in enumerate(series):
                offset = (k - (len(series) - 1) / 2) * width
                points = {"+".join(map(str, s["features"]["ngram_sizes"])): s for s in group}
                xs = [i + offset for i, key in enumerate(sizes) if key in points]
                runs = [points[key]["models"][m] for key in sizes if key in points]
                ax.errorbar(xs, [r["cv_mean"] for r in runs], yerr=[r["cv_std"] for r in runs],
                            fmt=MARKERS[slot], color=SERIES[slot], ms=5, capsize=2.5, lw=1.2,
                            label=_label(method))  # fmt: skip
            ax.set_xticks(range(len(sizes)), [f"n = {s}" for s in sizes])
            ax.set_xlabel("character n-gram sizes")
            ax.grid(axis="x", visible=False)
            ax.axhline(TARGET_F1, color=MUTED, lw=1)
        axes[0].set_ylabel("CV macro-F1 (mean ± std over folds)")
        axes[0].legend(title="feature selection", loc="lower right", title_fontsize=8)
        fig.suptitle("Ablation: n-gram sizes", x=0.01, ha="left", fontweight="bold",
                     fontsize=10, color=INK)  # fmt: skip
        return fig

    return _draw(draw)


_OPTION_ORDER = ("n_features", "ngram_sizes", "use_delimiters", "selection", "min_df",
                 "word_tokens")  # fmt: skip


def _option_text(key: str, value: object) -> str:
    if key == "n_features":
        return f"M={value}"
    if key == "ngram_sizes":
        return "n=" + "+".join(str(n) for n in value)
    if key in ("use_delimiters", "word_tokens"):
        return ("delimiters " if key == "use_delimiters" else "words ") + ("on" if value else "off")
    if key == "selection":
        return _label(str(value))
    return f"{key}={value}"


def diff_label(features: dict, base: dict) -> str:
    """Only the options that differ from the base setting, e.g. `M=2000, chi2`."""
    parts = [_option_text(k, features[k]) for k in _OPTION_ORDER
             if k in features and features[k] != base.get(k)]  # fmt: skip
    return ", ".join(parts) or "base"


def ablation_deltas(ablations: dict) -> object:
    """Paired Δ vs base for every setting. Filled marker: |mean Δ| > its std (beyond fold noise)."""

    def draw():
        settings = [s for s in ablations["settings"] if not s["is_base"]]
        models = ablations["models"]
        first = models[0]
        settings.sort(key=lambda s: s["models"][first]["delta_mean"])
        fig, axes, _ = _model_panels(ablations, 7.6, 0.24 * len(settings) + 1.8)
        y = list(range(len(settings)))
        for ax, m in zip(axes, models, strict=True):
            for i, s in zip(y, settings, strict=True):
                r = s["models"][m]
                real = abs(r["delta_mean"]) > r["delta_std"]
                ax.errorbar(r["delta_mean"], i, xerr=r["delta_std"], fmt=MARKERS[0], ms=4.5,
                            color=SERIES[0], mfc=SERIES[0] if real else SURFACE, capsize=2,
                            lw=1)  # fmt: skip
            ax.axvline(0, color=AXIS, lw=1)
            ax.set_xlabel("Δ CV macro-F1 vs base")
            ax.grid(axis="y", visible=False)
        base = ablations["settings"][ablations["base"]]
        axes[0].set_yticks(
            y, [diff_label(s["features"], base["features"]) for s in settings], fontsize=7.5
        )
        fig.suptitle(f"Ablation: change vs base ({base['label']}), paired mean ± std over folds;"
                     "\nhollow marker = within fold noise (|mean| ≤ std)",
                     x=0.01, ha="left", fontweight="bold", fontsize=10, color=INK)  # fmt: skip
        return fig

    return _draw(draw)


# ---------------------------------------------------------- tm-curves.json figures


def tm_curves(curves: dict) -> object:
    """Held-out and training macro-F1 per epoch, one panel per TM setting, chosen epoch marked."""

    def draw():
        settings = list(curves["settings"].items())
        fig, axes = _new_figure(3.8 * len(settings), 3.3, ncols=len(settings), sharey=True)
        for ax, (name, s) in zip(axes, settings, strict=True):
            sm = s["summary"]
            epochs = list(range(1, len(sm["val_mean"]) + 1))
            lo = [m - d for m, d in zip(sm["val_mean"], sm["val_std_folds"], strict=True)]
            hi = [m + d for m, d in zip(sm["val_mean"], sm["val_std_folds"], strict=True)]
            ax.fill_between(epochs, lo, hi, color=SERIES[0], alpha=0.12, lw=0)
            ax.plot(epochs, sm["val_mean"], color=SERIES[0], lw=1, alpha=0.45)
            ax.plot(epochs, sm["val_smoothed"], color=SERIES[0], lw=2,
                    label="held-out (smoothed; band ± std over folds)")  # fmt: skip
            ax.plot(epochs, sm["train_mean"], color=SERIES[1], lw=1.5, ls="-", label="train")
            e = sm["chosen_epoch"]
            ax.axvline(e, color=INK_2, lw=1)
            ax.text(e, 0.0, f" epoch {e}", transform=ax.get_xaxis_transform(), color=INK_2,
                    fontsize=8, va="bottom")  # fmt: skip
            ax.axhline(TARGET_F1, color=MUTED, lw=1, label=f"target {TARGET_F1}")
            params = s["params"]
            title = (f"{name}: {params.get('n_clauses')} clauses/language, "
                     f"T={params.get('T')}, s={params.get('s')}")  # fmt: skip
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("epoch")
        axes[0].set_ylabel("macro-F1")
        axes[0].set_ylim(bottom=max(0.0, min(min(s["summary"]["val_mean"]) for _, s in settings)
                                    - 0.05), top=1.01)  # fmt: skip
        axes[0].legend(loc="lower right")
        fig.suptitle("TM training curves (CV folds of train, mean over folds x seeds)", x=0.01,
                     ha="left", fontweight="bold", fontsize=10, color=INK)  # fmt: skip
        return fig

    return _draw(draw)


def tm_clause_formation(curves: dict) -> object:
    """Clause growth and stability per epoch: included literals per clause, decisions changed."""

    def draw():
        settings = list(curves["settings"].items())
        fig, axes = _new_figure(7.6, 3.1, ncols=2)
        for slot, (name, s) in enumerate(settings[: len(SERIES)]):
            sm = s["summary"]
            epochs = list(range(1, len(sm["val_mean"]) + 1))
            per_clause = [i / n if n else 0.0 for i, n in
                          zip(sm["included_mean"], sm["nonempty_mean"], strict=True)]  # fmt: skip
            every = max(1, len(epochs) // 10)
            style = dict(color=SERIES[slot], marker=MARKERS[slot], ms=4, markevery=every,
                         label=name)  # fmt: skip
            axes[0].plot(epochs, per_clause, **style)
            axes[1].plot(epochs, sm["changed_mean"], **style)
        axes[0].set_title("literals per non-empty clause", fontsize=9)
        axes[1].set_title("include decisions changed per epoch (log scale)", fontsize=9)
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("epoch")
        axes[0].legend(loc="lower right")
        fig.suptitle("How clauses form during training (mean over folds x seeds)", x=0.01,
                     ha="left", fontweight="bold", fontsize=10, color=INK)  # fmt: skip
        return fig

    return _draw(draw)


# --------------------------------------------------------- tm-results.json figures


def _tm_rows(tm_results: dict) -> list[dict]:
    """Baselines then TM settings as dicts: CV, repeated test and test scores with spreads."""
    rows = []
    for b in tm_results["baselines"]:
        rows.append(dict(name=_label(b["name"]), cv=b["cv_mean"], cv_err=b["cv_std"],
                         rep=b["repeat_mean"], rep_err=b["repeat_std"], test=b["test_f1"],
                         test_lo=b["test_f1"], test_hi=b["test_f1"]))  # fmt: skip
    for name, t in tm_results["tm"].items():
        a = t["aggregate"]
        rows.append(dict(name=f"{name} (TM)", cv=a["cv_mean"], cv_err=a["cv_std_folds"],
                         rep=a.get("repeat_mean"), rep_err=a.get("repeat_std_splits"),
                         test=a["test_mean"], test_lo=a["test_min"],
                         test_hi=a["test_max"]))  # fmt: skip
    return rows


def tm_comparison(tm_results: dict) -> object:
    """TM settings beside the baselines: CV, repeated test, single test (TM: range over seeds)."""

    def draw():
        rows = _tm_rows(tm_results)[::-1]
        fig, (ax,) = _new_figure(6.4, 0.55 * len(rows) + 1.5)
        y = list(range(len(rows)))
        ax.errorbar([r["cv"] for r in rows], [i + 0.15 for i in y],
                    xerr=[r["cv_err"] for r in rows], fmt=MARKERS[0], color=SERIES[0], ms=6,
                    capsize=3, lw=1.5, label="CV, mean ± std over folds")  # fmt: skip
        rep = [(i, r) for i, r in zip(y, rows, strict=True) if r["rep"] is not None]
        if rep:
            ax.errorbar([r["rep"] for _, r in rep], [i - 0.15 for i, _ in rep],
                        xerr=[r["rep_err"] for _, r in rep], fmt=MARKERS[1], color=SERIES[1],
                        ms=6, capsize=3, lw=1.5, label="repeated test, mean ± std")  # fmt: skip
        for i, r in zip(y, rows, strict=True):
            if r["test_hi"] > r["test_lo"]:
                ax.plot([r["test_lo"], r["test_hi"]], [i, i], color=INK_2, lw=1, alpha=0.6)
        ax.scatter([r["test"] for r in rows], y, marker="|", s=120, color=INK_2, zorder=3,
                   label="single test (TM: mean, line = range over seeds)")  # fmt: skip
        ax.axvline(TARGET_F1, color=MUTED, lw=1)
        ax.text(TARGET_F1, len(rows) - 0.45, f" target {TARGET_F1}", color=MUTED, fontsize=8,
                va="bottom")  # fmt: skip
        ax.set_yticks(y, [r["name"] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("macro-F1")
        seeds = len(next(iter(tm_results["tm"].values()))["seeds"]) if tm_results["tm"] else 0
        ax.set_title(f"TM vs baselines, same splits (TM: mean over {seeds} seeds)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=1)
        return fig

    return _draw(draw)


def tm_per_language(tm_results: dict) -> object:
    """Test F1 per language: baselines and TM settings (TM averaged over seeds)."""

    def draw():
        langs = tm_results["languages"]
        names, values = [], []
        for b in tm_results["baselines"]:
            names.append(_label(b["name"]))
            values.append([b["per_language_f1"][g] for g in langs])
        for name, t in tm_results["tm"].items():
            names.append(f"{name} (TM)")
            values.append([t["aggregate"]["per_language_f1"][g] for g in langs])
        # All scores here are high: start the scale just below the lowest one so differences
        # between strong models stay visible (the printed values are exact either way).
        low = min(min(row) for row in values)
        vmin = math.floor((low - 0.05) * 10) / 10
        fig, (ax,) = _new_figure(6.4, 0.42 * len(names) + 1.5)
        image = _heatmap(ax, values, names, langs, vmin, 1.0, ".2f")
        fig.colorbar(image, ax=ax, shrink=0.8, label="test F1").outline.set_visible(False)
        ax.set_title("Test F1 per language: TM vs baselines")
        return fig

    return _draw(draw)


# ------------------------------------------------------------------------ all figures

RESULTS_FIGURES: dict[str, Callable[[dict], object]] = {
    "dataset_composition": dataset_composition,
    "baseline_comparison": baseline_comparison,
    "per_language_f1": per_language_f1,
    "confusion_best": confusion,
    "resources": resources,
}
ABLATION_FIGURES: dict[str, Callable[[dict], object]] = {
    "ablation_vocabulary": ablation_vocabulary,
    "ablation_ngrams": ablation_ngrams,
    "ablation_deltas": ablation_deltas,
}
TM_RESULT_FIGURES: dict[str, Callable[[dict], object]] = {
    "tm_comparison": tm_comparison,
    "tm_per_language": tm_per_language,
}
CURVE_FIGURES: dict[str, Callable[[dict], object]] = {
    "tm_curves": tm_curves,
    "tm_clause_formation": tm_clause_formation,
}


def load_sidecar(path: str | Path, schema_prefix: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sidecar not found: {path} (re-run the command that makes it)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not str(data.get("schema", "")).startswith(schema_prefix):
        raise ValueError(f"{path}: expected schema {schema_prefix}*, got {data.get('schema')!r}")
    return data


def render_all(
    results_path: str | Path | None,
    ablations_path: str | Path | None,
    out_dir: str | Path,
    curves_path: str | Path | None = None,
    tm_results_path: str | Path | None = None,
) -> list[Path]:
    """Draw every figure whose sidecar is given; returns the written PNG paths."""
    _mpl()  # fail early with the install hint
    written = []
    jobs = []
    if results_path:
        jobs.append((load_sidecar(results_path, "codelangtm.results/"), RESULTS_FIGURES))
    if ablations_path:
        jobs.append((load_sidecar(ablations_path, "codelangtm.ablations/"), ABLATION_FIGURES))
    if curves_path:
        jobs.append((load_sidecar(curves_path, "codelangtm.tm-curves/"), CURVE_FIGURES))
    if tm_results_path:
        jobs.append((load_sidecar(tm_results_path, "codelangtm.tm-results/"), TM_RESULT_FIGURES))
    for data, figures in jobs:
        for name, fn in figures.items():
            written.append(save_png(fn(data), Path(out_dir) / f"{name}.png"))
    return written
