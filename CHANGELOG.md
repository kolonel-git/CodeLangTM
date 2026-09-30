# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Language selection for experiments (Stage B, B-S4): `languages:` in the baselines and ablations YAML configs and `--languages auto|core|all|python,go` on `baselines`, `ablate` and `diagnose`. `auto` (default) evaluates every language in the data, `core` the original 8 on the same CV folds (each snippet keeps its fold), `all` the 14; a requested language without data is an error, and the list is recorded in every report and JSON sidecar (`meta.languages`). `curves`, `tm-results`, `tm-train` and `errors` use `auto`. Helpers `resolve_languages`, `restrict_dataset`, `parse_language_spec` (`data.py`) and `load_evaluation_data` (`baselines.py`).
- `collect github` searches SQL under its four GitHub language names (`SQL`, `TSQL`, `PLpgSQL`, `PLSQL`): the plain `SQL` class holds only about 70 repositories with 10+ stars. The manifest records `search_languages`.
- `collect github --pages N`: read N result pages (100 repos each) per star band; default 1 (unchanged). Recorded in the manifest.
- Stage B, B-S2, embedded-language policy: HTML windows with more than 50% of lines inside `<script>`/`<style>` and PHP windows with more than 50% HTML-tag lines are set aside per window as evaluation-only **hard examples** instead of being dropped or trained on. The collector writes them to `data/hard/github/` (`--hard-out`) and tries up to 3 windows per file; `data build` reads `data/hard/` (`--hard-source`), writes `hard.jsonl` and a `hard` block in `dataset.json`, and never lets them change train, test or folds. `labels.check_label(...).hard`, `filter_labels(..., hard=)`.
- Stage B, B-S1: six stretch languages (C, C#, TypeScript, Kotlin, PHP, Ruby) in the collector, label checks and comment/string scanner; `CORE_LANGUAGES`, `STRETCH_LANGUAGES`, `ALL_LANGUAGES` and `languages_present`. `collect github` defaults to all 14 languages; `data build` and `data audit` accept all 14 and cover the languages present. The collector samples `.h` for C and C++ and skips `.jsx`/`.tsx`, `.d.ts`, generated C#/protobuf files and `obj/`, `bin/`. `LANGUAGES` (evaluation default) stays the 8 core languages until dataset v6.
- `codelangtm errors` (`errors.py`): out-of-fold predictions for every training snippet (the TM for every seed, the baselines on the same folds, with `evaluate_model`'s exact fold pipeline; per-fold scores are checked against `tm-results.json`), then confusable pairs (with their Spearman correlation to the clause inspector's language overlap), hard vs unlucky errors across seeds, exact McNemar tests TM vs each baseline (per seed and seed majority vote), which languages receive errors, and verdict-rule hypotheses for the top receiver (evidence, margins, rejection strength, generic signatures, pulling rules, embedded `<script>`/`<style>` code in HTML). Writes `docs/errors.md` + `errors.json` (snippet references only) and a local review file with code and fired clauses (`data/processed/errors-review.md`). Figures `errors_confusion`, `errors_receivers`, `errors_embedded` via `codelangtm report --errors`.
- Clause inspector (`rules.py`): `codelangtm clauses` turns every clause of a model file into a readable rule (`has("fn ") AND NOT has("\t}")`, whitespace and quotes escaped, word features as `word("X")`) and writes `docs/clauses.md` + `clauses.json` (every clause, one per line, with coverage / false fires / precision on the training set). It also reports per-language summaries (empty, duplicate and narrow clauses), signature n-grams (usage × log lift), language overlap (Jaccard), top 20 "for" clauses per language, stability across the other seeds' models (top-10 overlap and score correlation), and a formation replay: the official model is retrained epoch by epoch, the Jaccard similarity to the final clauses and each signature n-gram's share are tracked, and the replay must end at the saved model. The command refuses a dataset whose hashes differ from the model's, and independently checks that every rule evaluated from its text equals the model's clause outputs. Figures `clause_signatures`, `clause_overlap`, `clause_shapes`, `clause_formation_replay` are drawn via `codelangtm report --clauses`.
- `codelangtm explain --file FILE|-`: votes per language and the strongest clauses that fired for the top two languages (long rules shortened), from the model file alone (no TMU).
- `codelangtm resources` (`resources.py`): process-level resources, each job in a fresh Python process (fit: wall/CPU time, memory before and peak; predict: model load time, per-snippet binarize/predict/total latency median and p95, batch throughput), repeated with the median kept; process memory from the OS (Windows working set via `GetProcessMemoryInfo`, `getrusage` elsewhere). Writes `docs/resources.md` + `.json`; figure `process_resources.png` via `codelangtm report --resources`.
- `codelangtm tm-train` (one TM from `configs/tm.yaml`, trained on all of train, saved as a model file) and `codelangtm tm-select` (copies the seed with the median CV score from a `tm-results` run to `models/tm.json`, recording the rule and every seed's CV score in the model's metadata).
- `codelangtm tm-results` (`tm_results.py`): every TM setting x seed through the baselines' exact protocol (`evaluate_model`), baselines re-run beside them, corrected resampled t-test (Nadeau & Bengio) per fold and per repeated split; writes `docs/tm-results.md` + `.json` and the final models (one JSON per setting x seed) to `models/` (gitignored). Figures `tm_comparison.png`, `tm_per_language.png` via `codelangtm report --tm-results`.
- `evaluate_model(keep_pipeline=True)` returns the final fitted pipeline (`ModelResult.pipeline`, never written to reports).
- TM experiment config (`TMConfig`, `configs/tm.yaml`: seeds, TM settings, curve rule; features must equal the frozen baseline features).
- `codelangtm tm-curve` (`curves.py`): TM training curves on CV folds of train (every fold x seed, one epoch at a time), clause-formation statistics per epoch, smoothed epoch rule (`epoch_rule`: `plateau` or `max`, the smoothed peak); writes `docs/tm-curves.md` + `.json`. `codelangtm report --curves` draws `tm_curves.png` and `tm_clause_formation.png`.
- `TMLanguageClassifier` (`model.py`): Tsetlin Machine as a scikit-learn estimator (`fit`, `partial_fit`, `predict`, `decision_function`, `predict_tmu`), TMU imported lazily and without its pycuda noise.
- `TMState`: trained TM as NumPy arrays, predictions identical to TMU's; one-file JSON model format `codelangtm.tm/1` with `save_model`, `save_pipeline`, `load_model` (documented in `docs/architecture.md`).
- `Binarizer.to_dict` / `Binarizer.from_dict` (used by the model file; `save`/`load` unchanged).
- Project scaffold: package layout, binarizer, CLI stub, CI, docs.
- `Snippet` schema with JSONL I/O (`data.py`).
- Line-window extractor with low-signal filter (`windows.py`).
- Exact + MinHash near-duplicate removal (`dedup.py`).
- Label sanity checks (`labels.py`): extension/content agreement, minified/binary, required markers, template-heavy SQL/HTML, cut comments/strings.
- Group-by-repo splits and stratified group k-fold (`splits.py`).
- GitHub collector with license allowlist, star bands, rate limiting and caching (`github.py`, `codelangtm collect github`).
- Dataset build: leak-free train/test/wild + CV folds + manifest (`build.py`, `codelangtm data build`).
- Dataset audit: stats, quality flags and review samples (`audit.py`, `codelangtm data audit`).
- Comment/string scanner (`syntax.py`) so windows never start or end mid-comment.
- Dataset card (`docs/dataset-card.md`, now Stage A v5) and source ledger entries.
- Classical baselines with repo-grouped CV and generated `docs/results.md` (`baselines.py`, `codelangtm baselines`).
- Data diagnostics: confident-learning label-issue candidates and shortcut probe (`diagnostics.py`, `codelangtm diagnose`).
- YAML experiment configs (`config.py`, `configs/baselines.yaml`, `codelangtm baselines --config`): strict validation, CLI overrides, settings hash recorded in `docs/results.md`. New dependency `pyyaml`.
- Feature ablation runner (`ablations.py`, `configs/ablations.yaml`, `codelangtm ablate`): train-only repo-grouped CV per feature setting, paired Δ vs base, per-language F1, binarize cost; generates `docs/ablations.md`.
- `Binarizer` options `selection` (`frequency`, `chi2`, `class_balanced`), `min_df` and `word_tokens` (whole identifier/keyword features, shown as `word:NAME`); configurable in YAML. Defaults keep the previous behaviour.
- Resource metrics in `docs/results.md` for every model: fit CPU time, peak memory (binarizer and classifier separately), size split into vocabulary and classifier, throughput.
- JSON sidecars for generated reports: `codelangtm baselines` writes `results.json` and `codelangtm ablate` writes `ablations.json` next to the markdown (same runs, machine-readable). Report metadata now includes per-language dataset composition (snippets and repos per split).
- `codelangtm report`: writes the report figures to `docs/figures/`; hand-written results report `docs/report.md`.
- Documentation link check (`tests/test_docs.py`): relative links, images and anchors in all markdown must resolve.
- Report figures (`figures.py`): 8 deterministic PNGs drawn from the JSON sidecars (dataset, baselines, per-language F1, confusion, resources, ablations).
- Optional `viz` extra (matplotlib) for figures and the report; `docs/figures/` for generated plots.

### Fixed
- Evaluation commands on data with more languages than the 8 core ones no longer report only those 8: `evaluate_model` refuses a dataset that has languages outside the evaluation list, and every command now derives its languages from the data (`auto`). Results on the 8-language v5 data are unchanged (scores, folds, confusion matrices and repeated splits identical; config hash unchanged for `languages: auto`).
- Collector: an HTTP 429 without `Retry-After` or rate-limit headers (secondary rate limit) is retried after 60 s x attempt instead of skipping the repository.
- C label check: C++ constructs (`constexpr`, `nullptr`, casts, `enum class`, `::` outside comments, `<cstdio>`-style includes, access specifiers, `using X =`) and Objective-C markers (`@interface`, `#import`) now mark a `.c`/`.h` file as not C; the same C++ markers decide whether a `.h` is C++.
- `resources.memory_mb` on Linux: the peak now comes from the same `/proc/self/status` snapshot as the current value (`VmHWM`), since `getrusage`'s lazily updated peak could be below the current value and failed CI after PR #7 (fixed in PR #8); the peak is never reported below the current value.
- Pickling a `Binarizer` or `TMLanguageClassifier` no longer modifies the original object (Python 3.11+ `__getstate__` returns the live dict).
- HTML label check accepted comparisons (`i < x`) as tags; HTML windows that are mostly inline script/style are now dropped (embedded-language rule).
- TMU crash on NumPy 2: `tm` extra pins `numpy<2`, `scipy<1.14`.
- Windows splitting block comments, docstrings and multi-line strings (4.6% of Stage A v2 snippets).

### Changed
- The HTML rule "fewer than 20% of lines with a tag means mostly embedded" is now "more than 50% of lines inside `<script>`/`<style>`" (`embedded_share`, moved from `errors.py` to `labels.py`); HTML windows with few tags that are not embedded code are dropped as "too little markup". Rebuilding v5 from `data/raw/` now gives 75 HTML windows plus 25 hard ones; the published v5 numbers used the old rule.
- Report (M3 wrap-up): key findings and a targets scorecard up front; section 7 in story order (training curves, head-to-head, resources, inside the model, errors); statements from earlier steps corrected (the "Python sink", speed claims, pending reviews); limitations extended. README shows a rule the model really learned and measured values next to the targets.
- CI installs the `tm` and `viz` extras so TM and figure tests run in CI.
- `Binarizer.transform` is 5.4-6.0× faster (direct-address lookup of 1-3 character terms instead of building substring sets) with identical output; verified by fuzz tests against the `term in snippet` definition.
- `data build` uses a stable hash-based per-language repo split (`stable_split`, `--salt`) instead of StratifiedGroupKFold; re-collecting one language no longer reshuffles other languages' test repos.
- `codelangtm baselines` reports repeated test macro-F1 over 10 extra balanced splits (`--repeats`).
- `Binarizer` is a scikit-learn transformer (refit per CV fold), with alphabetical tie-breaking, JSON save/load, a `use_delimiters` switch and set-based transform. When `n_features` is smaller than the delimiter list, delimiters are now truncated too.
- `configs/baselines.yaml` now holds the feature config frozen for M3: `class_balanced` selection, M=500, 2+3-grams, no forced delimiters. Baseline CV macro-F1 rises from 0.917 to 0.963 (best model now Naive Bayes); latency 0.31 → 0.19 ms/snippet. Code defaults are unchanged.
- Dependencies: removed unused `pandas`; declared `scipy` (used directly by `features.py`).
