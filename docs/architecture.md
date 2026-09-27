# Architecture

```
Code snippet
  -> Feature extraction & binarization   (char n-grams, M selected per language; see Features)
  -> Literal expansion                   (x_k and NOT x_k -> 2M literals)
  -> Multi-class TM engine (TMU)         (bitwise clause evaluation)
  -> Vote summation                      (positive clauses - negative clauses)
  -> Predicted language (argmax)
```

## Modules (`src/codelangtm/`)

| Stage | Module | Command |
| --- | --- | --- |
| Snippet schema, JSONL I/O | `data.py` | |
| Collection (GitHub, permissive licenses) | `github.py`, `windows.py`, `syntax.py` | `codelangtm collect github` |
| Cleaning: label checks, dedup | `labels.py`, `dedup.py` | |
| Dataset build: stable repo split, CV folds, manifest | `build.py`, `splits.py` | `codelangtm data build` |
| Dataset audit | `audit.py` | `codelangtm data audit` |
| Features | `features.py` | |
| Experiment configs (YAML) | `config.py`, `configs/*.yaml` | |
| Classical baselines | `baselines.py` | `codelangtm baselines` |
| Feature ablations | `ablations.py` | `codelangtm ablate` |
| Label-issue and shortcut checks | `diagnostics.py` | `codelangtm diagnose` |
| Report figures (optional `viz` extra) | `figures.py` | `codelangtm report` |
| TM classifier, TMU-free state, model files | `model.py` | (used by `tm-curve`, and `baselines`/`tm-train` from B3) |
| TM experiment config, training curves | `config.py` (`TMConfig`), `configs/tm.yaml`, `curves.py` | `codelangtm tm-curve` |
| Rules, C export | `rules.py`, `export_c.py` | stubs (B6, M6) |

Data flow: `data/raw/` (collected, gitignored) → `data/processed/` (train/test/wild, `folds.json`, `dataset.json`) → generated reports in `docs/` (`results.md`, `ablations.md`, `tm-curves.md`), each with a JSON sidecar (`results.json`, `ablations.json`, `tm-curves.json`: same runs, machine-readable, read by the figure code) → `docs/figures/*.png` (`figures.py`: matplotlib imported only when drawing, fixed style and font, PNGs without metadata so redrawing the same sidecar gives identical bytes) → embedded in the hand-written [report.md](report.md).

## Features
`Binarizer` (scikit-learn transformer; refit inside every CV fold):
- **Candidates:** character n-grams of the configured sizes (default 2 and 3); optionally whole identifier/keyword tokens (`word_tokens`, shown as `word:NAME`).
- **Structural delimiters** (`;`, `{`, `->`, `#`, `::`, tabs, 4-space indent, comment tokens) always included when `use_delimiters` is on.
- **Selection of the M slots** (`selection`):
  - `frequency`: top M by document frequency (code default, unsupervised);
  - `chi2`: top M by chi² association with the language labels;
  - `class_balanced`: languages take turns picking their most distinctive feature (P(g | language) − P(g | other languages)).
- **Frozen for M3** (`configs/baselines.yaml`): `class_balanced`, M=500, 2+3-grams, no forced delimiters, no word tokens. Label-aware selection lifts CV macro-F1 from 0.917 to ~0.96 ([ablations.md](ablations.md)).
- **Literals:** `L = [x_1..x_M, NOT x_1..NOT x_M]`.
- **Speed:** `transform` matches 1-3 character terms with direct-address tables over code points (identical to substring tests, fuzz-verified); lookup tables are derived from the vocabulary and never pickled.

## Evaluation
- Repo-grouped splits: no repository in both train and test; stable per-language hash split, so re-collecting one language does not move others ([dataset-card.md](dataset-card.md)).
- Model selection and ablations use 5-fold repo-grouped CV on train only. Test is scored once per model, plus a repeated-split test score (10 further balanced splits) to show split variance.

## Model
`tmu.models.classification.vanilla_classifier.TMClassifier`, N_c clauses per class. Net score
`V_m(X) = sum C+_{m,j}(X) - sum C-_{m,j}(X)`; highest wins.
Feedback: Type I (pattern discovery, erasure with prob 1/s), Type II (false-alarm correction), gated by threshold T.

TMU 0.8.3 facts checked in the B1 spike ([issues-and-fixes](issues-and-fixes.md) M8):
- `number_of_clauses` is per class. The first half of the clauses start with weight +1 (vote for the class), the second half with -1 (vote against). With `weighted_clauses=True`, training moves the weights (seen range -20 to +29).
- `fit(X, Y)` runs **one epoch**; state persists across calls. X must be `uint32` (the `Binarizer` output is). TMU adds the negated literals itself, so literal `j < M` is `x_j` and literal `M + j` is `NOT x_j`.
- Prediction: class sum = `weights · clause_outputs` (not clipped); argmax, with ties going to the lower class id. A clause outputs 1 when none of its included literals is violated **and it includes at least one literal**, so empty clauses output 0 at prediction. A NumPy re-implementation from `clause_banks[c].get_literals()` and `weight_banks[c].get_weights()` matches TMU's class sums exactly (all 696 train snippets, two model sizes); counting empty clauses as firing does not.
- TMU's default incremental clause evaluation gives the same sums as the plain path, and its cache is reset by every training update, so predicting between epochs is safe.
- Training is deterministic for a given `seed` (identical per-epoch scores across separate runs). `seed=0` makes TMU hang (all-zero xorshift128+ state; [issues-and-fixes](issues-and-fixes.md) E6), so `TMLanguageClassifier` requires `seed >= 1`.

## Model files (`codelangtm.tm/1`)
`TMLanguageClassifier` (`model.py`) is a scikit-learn estimator: TMU trains it, and after every `fit`/`partial_fit` its clauses are copied into a `TMState` (plain NumPy arrays). All prediction goes through `TMState`, which gives exactly TMU's class sums, so a saved model needs no TMU. `save_model` / `save_pipeline` write **one JSON file** that holds everything needed to go from raw code to a prediction; `load_model` reads it back (`TrainedModel.predict(texts)`). The C export and the web demo will read this same file.

```json
{
  "format": "codelangtm.tm/1",
  "literal_encoding": "literal id j < n_features means has(feature j); id n_features + j means NOT has(feature j)",
  "classes": ["cpp", "go", "..."],
  "binarizer": {"n_features": 500, "ngram_sizes": [2, 3], "selection": "class_balanced", "...": "...", "vocabulary": ["::", "fn", "..."]},
  "tm": {
    "n_features": 500,
    "params": {"n_clauses": 400, "T": 100, "s": 5.0, "seed": 1, "epochs_trained": 50, "...": "..."},
    "clauses": {"cpp": {"weights": [3, -2, "..."], "literals": [[12, 640], [], "..."]}, "...": "..."}
  },
  "meta": {"dataset": "...", "config_hash": "..."}
}
```

- **Clauses** are listed per class in TMU's order: the first half start as "for" clauses (positive weight), the second half as "against" (negative weight). `literals` holds the ids of the included literals only (sparse); an empty list is an empty clause, which never fires.
- **Prediction:** a clause fires when every included `has(j)` feature is present and every included `NOT has(j)` feature is absent. A class's sum is the total weight of its firing clauses; the highest sum wins, and ties go to the class listed first.
- **Size:** the 400-clause model on the frozen features is about 160 KB as compact JSON (3,200 clauses, ~36,000 included literals, measured on one CV fold).

## Install notes
- TMU builds a C extension; Python 3.12 is pinned. It installs natively on Windows (checked with TMU 0.8.3); WSL2 or Docker is only a fallback.
- TMU 0.8.x breaks on NumPy 2, so the `tm` extra pins `numpy<2` and `scipy<1.14`.
- Always `uv sync --extra tm --extra collect --extra viz`: a plain `uv sync` removes the extras.
- `viz` (matplotlib) is only needed for figures; the package and CLI import without it.

## Targets
Macro-F1 >= 96% over 8 languages; < 0.1 ms/snippet; < 500 KB model.

Every model, baseline or TM, is compared on accuracy (CV, test, repeated test) and on the same resource metrics: training wall/CPU time, peak memory, model size, latency and throughput ([results.md](results.md), protocol in the roadmap under M3).
