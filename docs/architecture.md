# Architecture

```
Code snippet
  -> Feature extraction & binarization   (char n-grams + structural delimiters, top M selected)
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
| TM model, rules, C export | `model.py`, `rules.py`, `export_c.py` | stubs (M3, M5, M6) |

Data flow: `data/raw/` (collected, gitignored) → `data/processed/` (train/test/wild, `folds.json`, `dataset.json`) → generated reports in `docs/` (`results.md`, `ablations.md`).

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

## Evaluation
- Repo-grouped splits: no repository in both train and test; stable per-language hash split, so re-collecting one language does not move others ([dataset-card.md](dataset-card.md)).
- Model selection and ablations use 5-fold repo-grouped CV on train only. Test is scored once per model, plus a repeated-split test score (10 further balanced splits) to show split variance.

## Model
`tmu.models.classification.vanilla_classifier.TMClassifier`, N_c clauses per class. Net score
`V_m(X) = sum C+_{m,j}(X) - sum C-_{m,j}(X)`; highest wins.
Feedback: Type I (pattern discovery, erasure with prob 1/s), Type II (false-alarm correction), gated by threshold T.

## Install notes
- TMU builds a C extension; Python 3.12 is pinned. It installs natively on Windows (checked with TMU 0.8.3); WSL2 or Docker is only a fallback.
- TMU 0.8.x breaks on NumPy 2, so the `tm` extra pins `numpy<2` and `scipy<1.14`.
- Always `uv sync --extra tm --extra collect`: a plain `uv sync` removes the extras.

## Targets
Macro-F1 >= 96% over 8 languages; < 0.1 ms/snippet; < 500 KB model.

Every model, baseline or TM, is compared on accuracy (CV, test, repeated test) and on the same resource metrics: training wall/CPU time, peak memory, model size, latency and throughput ([results.md](results.md), protocol in the roadmap under M3).
