# CodeLangTM — Progress Log

Living tracker. Update after every work session. Planning detail lives in [docs/roadmap.md](docs/roadmap.md); problems and how they were solved live in [docs/issues-and-fixes.md](docs/issues-and-fixes.md).

**Last updated:** 2026-09-27
**Current milestone:** M3 — TM training (branch A merged, PR #6; branch B `feat/tm-training`: B1-B5 done; B6 clause inspector next)
**Overall:** M0 complete, M1 Stage A complete (dataset v5: 869 snippets, stable split, [dataset card](docs/dataset-card.md)), M2 done (PR #5 merged): feature config frozen (label-aware selection, [ablations](docs/ablations.md)), binarization 5-6× faster; bar to beat = Naive Bayes CV macro-F1 0.963, repeated test 0.968 ([results](docs/results.md)); M3 in progress: branch A merged (results [report](docs/report.md) with figures), branch B: the 400-clause TM matches Naive Bayes and logistic regression (repeated test 0.966 vs 0.968, p = 0.82; test 0.958 vs 0.959) with a 166 KB model

## Milestone overview

| Milestone | Status | Notes |
| --- | --- | --- |
| M0 Foundations | Done | Scaffold, CI, docs, roadmap |
| M1 Data pipeline | Stage A done | v5: 869 snippets, 196 repos, stable split; Stage B items deferred |
| M2 Features & baselines | Done | Parts 1 and 2 merged (PR #5: configs, ablations, frozen features, resource metrics, faster binarization) |
| M3 TM training | In progress | Branch A merged (PR #6: report + figures); branch B `feat/tm-training`: B1-B4 done (TM matches the baselines) |
| M4 Tuning & compression | Planned | |
| M5 Explainability | Planned | |
| M6 Deployment & benchmarks | Planned | |
| M7 Showcase & release | Planned | |

## Next steps (in order)

M0-M2 are finished; each item is recorded in the done log below and ticked in [docs/roadmap.md](docs/roadmap.md).

- [x] M3 branch A `feat/report`: A0 housekeeping, A1 JSON sidecars, A2 `figures.py`, A3 `codelangtm report` + [docs/report.md](docs/report.md)
- [x] Push `feat/report`, PR #6 merged; CI green with TMU and matplotlib installed (TMU builds on the Ubuntu runner)
- [x] B1 spike: timing, NumPy-vs-TMU equivalence, first CV numbers (issues-and-fixes M8)
- [x] B3 main TM setting chosen: N_c=400, T=100, s=5 (planned 100/30/3.5 kept as a reference row)
- [x] Direction agreed (see done log 2026-09-27): portfolio + later research write-up, research-ready rigor, Stage B after M3, CLI + C runtime + web demo, concepts guide
- [x] B2 `TMLanguageClassifier`, `TMState`, JSON model file
- [x] B4 training curves (run before B3): epochs 120 (400 clauses) and 142 (100 clauses), smoothed peak
- [x] B3 full protocol: TM 400 matches the baselines (no significant difference); TM 100 significantly behind logistic regression
- [x] B4b `tm-train` + `tm-select`: official model = tm_400 seed 4 (median CV)
- [x] B5 process-level resources (`codelangtm resources`)
- [ ] M3 branch B `feat/tm-training`: B6 `TMLanguageClassifier` → B3 config + protocol → B4 curves, B4b `tm-train` → B5 resources → B6 clause inspector → B6b error analysis → B7 report (details in the roadmap, M3)
- [ ] M3/M4: use repeated splits for the final TM vs baselines comparison (5 seeds, paired significance test)
- [ ] After M3, before M4: Stage B (The Stack / CodeSearchNet loaders, stretch languages, embedded-language policy), wild set collected by hand (StackOverflow / blogs / docs), short-snippet evaluation (roadmap Future)

## Blockers / open questions
- None.

## Done log

### 2026-09-28 — M3 branch B, B5: process-level resources
- Choice (start of step): full protocol.
- `resources.py` + `codelangtm resources`: every measurement in a fresh Python process doing one job (`python -m codelangtm.resources fit|predict`), 3 repeats, median kept. Fit: wall/CPU time, process memory before training and at its peak. Predict: load the saved model (TM: the JSON model file, NumPy only, no TMU; baselines: pickled pipeline), then every test snippet one at a time, 3 passes, binarize and predict timed separately (median, p95), plus one batch pass. Process memory from the OS: Windows `GetProcessMemoryInfo` (working set), `getrusage` elsewhere; includes C-extension memory that `tracemalloc` misses.
- Models: Naive Bayes, logistic regression, TM 400 seed 4 (official), TM 100 seed 1 (its median-CV seed). 1 min 14 s in total.
- **Results:**
  - Training memory added is about 76 MB for every model: the binarizer's candidate table dominates (matches the M2 tracemalloc finding, issues-and-fixes M7). Training time: TM 400 7.7 s, TM 100 5.9 s, NB 0.35 s, LR 0.41 s.
  - One snippet, median (p95): LR 0.089 (0.109), TM 100 0.199 (0.288), NB 0.215 (0.251), TM 400 0.343 (0.471) ms. Only LR meets < 0.1 ms per snippet in Python. The TM's cost is the classifier (0.28 ms): the dense NumPy matrix (500 x 3,200) is read for every snippet although clauses include ~12 literals each; NB pays scikit-learn's per-call input checks.
  - Batch: 0.031-0.057 ms per snippet (TM 400: ~17,500 snippets/s).
  - Binarize is identical code for all models yet measured 0.033-0.056 ms: the measurement noise floor on this machine.
- Implication for M6: the C runtime should evaluate only included literals per clause (sparse); single-snippet latency is the number to beat.
- Report: section 7.2 added (resources), section 6 points to it; concepts explains process memory, median/p95 and fresh processes.
- Tests: 322 passing (7 new in `tests/test_resources.py`, figure and docs tests extended: current memory grows when 80 MB is touched, fit and predict jobs in fresh processes, failing job surfaces stderr, median-seed model choice, end-to-end CLI with a tiny TM, missing sidecar).

### 2026-09-28 — M3 branch B, B4b: one-command training and the official model
- Choice (start of step): both a `tm-train` command and a documented rule for which B3 model is official.
- `codelangtm tm-train --setting tm_400 --seed 4`: trains one TM from `configs/tm.yaml` on all of train (no CV, no repeated splits; 7.9 s) and saves the model file; the test score is printed for information only.
- `codelangtm tm-select`: the official model `models/tm.json` is the seed whose CV score is the median of the 5 (no test data). For `tm_400` the CV scores are 0.9461 (seed 5), 0.9488 (3), **0.9499 (4)**, 0.9505 (1), 0.9520 (2), so seed 4 (its test score, 0.954, played no part). The rule and all CV scores are stored in the model's metadata.
- Check: retraining seed 4 with `tm-train` gives exactly B3's saved model (every clause and the vocabulary identical; test 0.953 both times). The official model is reproducible from the config in one command.
- Tests: 314 passing (4 new: median rule incl. even counts and ties, selection copies the right model with metadata, `train_final` reproduces the protocol's model, CLI).

### 2026-09-28 — M3 branch B, B3: TM vs baselines, full protocol
- Choices (start of step): full protocol for both TMs over 5 seeds; save the final models. Session paused while the run finished and resumed afterwards.
- `tm_results.py` + `codelangtm tm-results`: each TM setting x seed runs `baselines.evaluate_model`, the exact code used for the baselines (5 CV folds, one fit on all of train scored once on test, 10 repeated splits, resource measurements); Naive Bayes and logistic regression re-run beside them. Corrected resampled t-test (Nadeau & Bengio) on per-fold and per-split differences, the TM averaged over seeds; test/train ratios 0.25 (CV) and the measured one for the repeated splits. Final models saved to `models/<setting>_seed<k>.json`; a test reloads them and reproduces the reported test F1 exactly.
- **Result (dataset v5, seeds 1-5, 24 min 36 s):**
  - TM 400 (120 epochs): CV 0.949 ± 0.023, test 0.958 (seeds 0.947-0.966), repeated 0.966 ± 0.009. Naive Bayes 0.963 / 0.959 / 0.968; logistic regression 0.953 / 0.971 / 0.968.
  - TM 400 − Naive Bayes: repeated −0.001 [−0.012, +0.010] p = 0.82; CV −0.013 [−0.056, +0.030] p = 0.44. No significant difference from either baseline.
  - TM 100 (142 epochs): 0.933 / 0.946 / 0.949; significantly behind logistic regression on the repeated splits (−0.019 [−0.034, −0.004], p = 0.016).
  - The seed spread is small for the 400-clause TM (std 0.002 CV, 0.007 test, 0.004 repeated).
- Size and speed (this run, loaded machine): TM 400 model file 166 KB (3,200 clauses, 12.1 literals each, 45 empty), 0.081 ms/snippet end to end (TMU's own predict 0.131), fit 7.7 s; TM 100 34 KB, 0.060 ms. Naive Bayes 0.038 ms, 67 KB.
- Error pattern for B6b: summed over 5 seeds, Python attracts snippets from other languages (Java → Python 8, C++ → Python 3, Rust → Python 3); also Rust → C++ 5, HTML → JavaScript 5.
- Figures: `tm_comparison.png`, `tm_per_language.png` (colour scale starts just below the lowest score so differences between strong models stay visible; the legend is stacked so it is not cut off). Report section 7 restructured: 7.1 head-to-head, 7.2 curves, 7.3 still to come; concepts explains p-values and intervals.
- Tests: 310 passing (8 new in `tests/test_tm_results.py`, figure and docs tests extended: the t-test equals scipy's paired t-test without correction and widens with it, hand-checked t, edge cases, protocol shapes, saved models reproduce the test score, significance rows, JSON/markdown, CLI).

### 2026-09-27 — M3 branch B, B4: TM training curves (run before B3)
- Choices (start of step): B4 before B3 so B3 uses a CV-chosen epoch count; 150 epochs; epoch rule = first epoch where the smoothed held-out curve (11-epoch centred moving average, mean over folds x seeds) is within 0.005 of its maximum. Significance test for B3 chosen: Nadeau-Bengio corrected resampled t-test; TM results go to `docs/tm-results.md`.
- New: `TMConfig` + `configs/tm.yaml` (seeds 1-5, settings `tm_400` and `tm_100`, curve rule; a test keeps its features equal to the frozen baseline features); `curves.py` + `codelangtm tm-curve` (5 folds x 5 seeds x 150 epochs per setting; per epoch: held-out and train macro-F1, epoch time, included literals, non-empty clauses, include decisions changed); `docs/tm-curves.md` + `.json` (every raw run kept); figures `tm_curves.png`, `tm_clause_formation.png`; `codelangtm report --curves`.
- **Result (CV only):** 400 clauses levels off at epoch 19 (held-out 0.946) and peaks at epoch 120 (smoothed 0.951; last 20 epochs 0.948); 100 clauses levels off at 44 (0.928) and peaks at 142 (0.933). Naive Bayes on the same folds 0.963. Training F1 reaches 0.99 by epoch 5 and 1.000 by about 20: the TM fits the training data completely.
- **Epoch rule changed after seeing the curves (user decision, 2026-09-28):** from the plateau rule (19 / 44 epochs) to the smoothed peak (120 / 142 epochs): about 6× the training time for about 0.005 CV F1. Done before any test-set look; new config option `curves.epoch_rule: plateau | max`, both epochs reported in `tm-curves.md`. Recorded in the report because it makes the CV score at the chosen epoch slightly optimistic; the test set and repeated splits in B3 are the independent check.
- **Surprise: B1's 0.962 was a lucky seed.** Seeds 1-5 give 0.945-0.950 each; seed 42 through the final code gives 0.962 again (one fold 0.991), so the code is consistent. Recorded as a correction in issues-and-fixes M8; the report now says the TM trails Naive Bayes by about 1.5 points on CV.
- Clause formation (400 clauses): literals per non-empty clause 9.0 → 12.7; include decisions changed per epoch fall from ~29,000 (epoch 1) to ~3,000 (epoch 20) and ~56 (epoch 150).
- Fixed during the step: the moving average first used a shrinking one-sided window at the curve's start (epoch 1 showed 0.875 instead of its real 0.664); it now shrinks symmetrically, and the curves were rerun (identical F1 values, same chosen epochs: training is deterministic). A test's expected epoch was wrong by hand arithmetic (the code was right).
- Run time: 7 min 41 s for both settings (0.05 s/epoch for 400 clauses including copying the clauses out of TMU, 0.03 s for 100). The final rerun with `epoch_rule: max` gave identical F1 values but 0.16 / 0.11 s per epoch: the machine was busier, the same ±50%+ timing swings as before; B5 measures resources properly.
- Tests: 299 passing (20 new: config parsing, smoothing and epoch rule, curve runs, determinism, no test/held-out text in any binarizer, JSON/markdown, CLI, curve figures).

### 2026-09-27 — M3 branch B, B2: TM classifier and TMU-free model
- Choices (asked at the start of the step): sparse JSON model file; one bundled file (vocabulary + clauses + metadata); simple NumPy matrix inference, with speed left to the C runtime (M6).
- `TMLanguageClassifier` (`model.py`): scikit-learn estimator over the binary features.
  - `fit` trains `epochs` epochs from scratch; `partial_fit` adds one epoch (for the B4 curves); `decision_function` returns TMU's class sums; `predict_tmu` calls TMU for comparison.
  - Labels are encoded and checked (unknown labels rejected); TMU is imported only on training, without its pycuda traceback (E5).
  - Defaults: N_c=400, T=100, s=5, weighted clauses, 50 epochs (B4 sets epochs), seed 1.
- `TMState`: the trained model as arrays (clause → class, weight, included literal ids in CSR layout, empty clauses kept). One matrix product decides which clauses fire. `save_model` / `save_pipeline` / `load_model` write and read the one-file JSON format `codelangtm.tm/1` (documented in architecture); `TrainedModel.predict(texts)` goes from raw code to language without TMU.
- Real-data check (fold 0, 20 epochs): predictions and class sums identical to TMU; pickle and JSON round trips identical; JSON model 159 KB (3,200 clauses, 36,474 included literals, 52 empty); NumPy predict 0.031 vs TMU 0.070 ms/snippet (single run).
- Problems found and fixed:
  - **Pickling stripped the live model** (Python 3.11+ `__getstate__` returns the live dict; W6). `Binarizer` had the same pattern.
  - **TMU hangs forever with `seed=0`** (all-zero xorshift128+ state; E6). Seeds must now be >= 1.
  - TMU's own NumPy deprecation warnings are filtered in pytest (for `tmu.*` modules only).
- Tests: 279 passing (19 new in `tests/test_model.py`). They cover:
  - hand-built clauses with known answers (empty clauses, negated literals, ties);
  - NumPy vs TMU equality on a trained model (a mutation that lets empty clauses fire breaks it);
  - same seed → same model; `partial_fit` × 2 equals `fit` with 2 epochs;
  - pickling; JSON round trip and validation; pipeline + clone;
  - quiet TMU import; install hint without TMU.

### 2026-09-27 — Direction agreed, B3 setting chosen, concepts guide
- Branch `feat/tm-training` created from the updated `main`; B1 findings committed.
- Decisions (alignment questions):
  - **Purpose:** portfolio piece first, convertible into a research write-up later.
  - **Rigor from now on:** every TM number is a mean over 5 seeds with spread; paired significance test against Naive Bayes on the same folds; raw per-seed runs saved as JSON.
  - **Accuracy vs readability:** balance both. Report the accurate TM (N_c=400, T=100, s=5; option A for B3) and, in M4, the smallest model within ~1 F1 point, with the curve between them.
  - **Data:** stay on v5 through M3; Stage B (more data, stretch languages) after M3 and before M4 tuning.
  - **End product:** CLI + zero-dependency C runtime, plus a web demo. The demo technology is decided later; the saved model format must stay portable and documented.
  - **Check-ins:** at the start and end of each step.
  - **Explanations:** new plain-language [docs/concepts.md](docs/concepts.md), linked from the README and the report.
- Docs updated for these decisions: roadmap (Principles, Stage B timing, B2 portable format, B3 multi-seed + test, M4 smallest model, M6 web demo, M7 research write-up), README, report, CLAUDE.md, CONTRIBUTING, issues-and-fixes M8.

### 2026-09-27 — M3 branch B, B1: TMU spike (throwaway scripts, not committed)
- `feat/report` merged (PR #6); CI green, so TMU compiles on the Ubuntu runner and the figure tests pass there.
- Read TMU 0.8.3's classifier and clause bank source first: prediction is `weights · clause outputs` per class; the incremental prediction cache resets on every training update, so predicting between epochs is safe.
- Speed: 0.03-0.04 s per epoch on ~550 snippets (first epoch 0.06-0.12 s); TMU predict 0.07-0.09 ms/snippet (a Python loop per sample). The whole CV + test + 10 repeated splits protocol at 100 epochs is about a minute, so B3 keeps the full protocol.
- Equivalence: a NumPy re-implementation matches TMU's class sums exactly when empty clauses output 0 (off by up to 21 otherwise); incremental and plain TMU prediction agree; the same seed gives identical runs.
- Accuracy (5 CV folds, test not used): planned N_c=100/T=30/s=3.5 ends at 0.925-0.940 (3 seeds), Naive Bayes 0.963 on the same folds; N_c=400/T=100/s=5 reaches 0.962. Held-out F1 is noisy (±0.02 epoch to epoch), so B4's epoch rule needs smoothing. Details: issues-and-fixes M8; TMU facts: architecture, Model.
- Found TMU's harmless pycuda traceback at import (issues-and-fixes E5).

### 2026-09-27 — M3 branch A, A3: `codelangtm report` and the written report
- `codelangtm report [--results] [--ablations] [--out]` draws the 8 figures from the JSON sidecars into `docs/figures/` (committed). Missing sidecar, wrong schema or missing matplotlib: exit 1 with a clear message.
- `docs/report.md`: hand-written narrative for readers new to the project: goal and how a TM classifies, data, evaluation protocol (repo split, CV, repeated test, macro-F1), baselines with figures, ablation findings (label-aware selection), resources, TM section (placeholder until B7), limitations, reproduction commands. Every number was checked against `results.json` / `ablations.json`; four draft claims were corrected in the process (vocabulary-size gain up to +0.011 not < 0.01; word tokens now 25% slower, not 10%; frequency ranking catches up only for logistic regression; unverified explanations of individual errors removed).
- New `tests/test_docs.py`: every relative link, image and `#anchor` in all tracked markdown must resolve (checked to fail on a missing file, image and anchor), and `report.md` must embed every figure.
- Docs audit: README (install line with `viz`, `report` command, pipeline line matches the frozen features, M3 status, link to the report), CONTRIBUTING (install, figures, report.md upkeep), CLAUDE.md (one install line, `report` command, correct raw-data path, docs test), architecture (pipeline, install), roadmap (M2 marked done; M3 lists steps B1-B7 and the clause inspector; M5 notes what moved to M3), PR template (regenerate figures, update report numbers), this file (finished M0-M2 checklist collapsed into the done log).
- Observation: with the faster binarizer, delimiters are now ~75% of binarize time (0.114 vs 0.029 ms/snippet with them off), up from ~40% before; another reason they stay off (issues-and-fixes M2).
- Tests: 259 passing (20 new: link checks for each markdown file, report figure check, slug check, `report` CLI).

### 2026-09-27 — M3 branch A, A2: figures
- `src/codelangtm/figures.py` draws 8 PNGs from the JSON sidecars: dataset composition, baseline comparison (CV, repeated test, single test, 0.96 target), per-language test F1 heatmap, confusion matrix of the best model (colour = share of the true language, text = counts), resources (fit time, latency, size as three separate panels, never a dual axis), and three ablation figures (F1 vs vocabulary size per selection method, F1 per n-gram set, paired Δ vs base for all 30 settings with within-noise results hollow).
- Colours: three categorical slots from a palette checked with a colour-vision-deficiency validator (all pairs pass); each selection method keeps its colour in every figure; series also differ by marker shape and legend. Magnitudes use one blue ramp.
- matplotlib is imported only when drawing; without the `viz` extra the package and CLI still import and drawing fails with "run `uv sync --extra viz`" (tested in a subprocess with matplotlib blocked).
- Deterministic output: same sidecar, same bytes (no PNG metadata, fixed font/size/dpi, style applied at save time too); issues-and-fixes W5.
- Checked every figure by eye on the v5 sidecars; fixed a legend covering a label, a clipped legend, `6.99e+03` size labels, and unreadable ablation labels (now only the options that differ from base).
- Figures are not committed yet: A3's `codelangtm report` writes them to `docs/figures/`.
- Tests: 239 passing (9 new).

### 2026-09-27 — M3 branch A, A1: JSON sidecars
- `codelangtm baselines` and `codelangtm ablate` now also write `docs/results.json` / `docs/ablations.json` next to the markdown (path = markdown path with `.json`). Same runs as the markdown, floats kept to 6 decimals, schema tags `codelangtm.results/1` and `codelangtm.ablations/1`. The figure code (A2) reads these instead of parsing markdown.
- `results.json`: meta, language order, best by CV, and per model the CV folds, test, repeated-split scores, per-language F1, confusion matrix and resources.
- `ablations.json`: each unique setting stored once (base = id 0) with its features, vocabulary used, binarize time and per model the fold scores, paired Δ vs base and per-language F1; studies list their settings by id; best setting per model.
- Shared `dataset_meta` gained `composition` (snippets and distinct repos per language for train/test/wild), and `data_dir` is written with `/` on every OS.
- Regenerated both reports on v5: all F1 scores and deltas identical to the digit; only timings changed. Ablation binarize times now reflect the faster binarizer (base 0.11-0.12 ms/snippet, was ~0.30 when the ablations were last run).
- Tests: 230 passing (sidecar vs in-memory results, composition counts, stable JSON writing, CLI writes both files).

### 2026-09-25 — M3 branch A, A0 housekeeping (`feat/report`)
- `feat/ablations` merged (PR #5); M3 plan approved, work split into branch A (report) and branch B (TM training).
- Short-snippet evaluation moved from the M3 roadmap to Future (roadmap, dataset card limitation, this file).
- New optional extra `viz` (matplotlib) for figures; core code stays importable without it. CI now runs `uv sync --extra tm --extra viz` so TM and figure tests run there (TMU build on the runner is checked on the first PR).
- `docs/figures/` created for generated figures.
- Tests: 226 passing, ruff clean.

### 2026-09-25 — M2: faster binarization
- Profile: 92% of `Binarizer.transform` was building substring sets; the vocabulary lookups were 8%.
- Fix: code-point arrays + direct-address tables for 1-3 character terms (NumPy), set method kept for longer terms and word features, binary-search fallback for huge alphabets; lookup tables derived from the vocabulary and not pickled. Details and numbers: issues-and-fixes M2.
- Correctness: output identical to `term in snippet` (fuzz tests incl. non-BMP, unseen characters, lengths 1-4, words, delimiters, fallback path; mutation check fails 5 tests). Baseline scores unchanged to the digit.
- Speed: 5.4-6.0× faster over 5 interleaved runs (same process, same load). Absolute timings on this machine swing ±50% with background load (0.078-0.099 ms/snippet for transform during the measurement), so the < 0.1 ms end-to-end target is not yet confirmed; M6 benchmarks it on a quiet machine.
- Tests: 226 passing.
- M2 is now complete apart from the PR.

### 2026-09-25 — M2 part 2 step 4: frozen feature config, new baseline bar, resource metrics
- **Delimiters check** (`delimiters_x_selection` study): with label-aware selection, turning delimiters off changes LR by -0.006 and Naive Bayes by +0.003 (within noise) and cuts binarize time 44% (0.312 → 0.176 ms/snippet).
- **Frozen feature config** (in `configs/baselines.yaml`, guarded by a test): `class_balanced`, M=500, 2+3-grams, delimiters off, word tokens off, `min_df` 1. Reasons: top of the fold-noise band, every language gets its own features (cleaner TM rules), M=500 keeps the TM small (larger M adds < 0.005), everything else adds nothing. Code defaults unchanged (old behaviour), so earlier runs stay reproducible from flags.
- **New baseline bar** (dataset v5, frozen features, the one test look): Naive Bayes best by CV 0.963 ± 0.005, test 0.959, repeated test 0.968 ± 0.009; LR CV 0.953, test 0.971, repeated 0.968 ± 0.017; SVM 0.946 / 0.964; RF 0.950 / 0.944; DT 0.843. The ≥ 0.96 target is reached by simple models: the TM must match it, and its case rests on interpretability, size and speed.
- **Resource metrics** in `docs/results.md`, same for every model so the TM is just another row: fit wall and CPU time, peak memory (binarizer and classifier separately), size and its vocabulary part, latency and throughput. First version was misleading (memory tracing inflated CPU time and hid classifier differences); fixed before commit, issues-and-fixes M7. Process-level memory and the full TM resource protocol are in the roadmap under M3.
- Finding: the binarizer dominates everything (72 MB peak, ~0.18 ms/snippet) while the classifier needs ≤ 4 MB and ~0.01 ms. Faster binarization is now the main latency lever.
- Tests: 219 passing.

### 2026-09-24 — M2 part 2 step 3: label-aware selection, word tokens, larger M
- `Binarizer` options: `selection` (`frequency` default, `chi2`, `class_balanced`), `min_df`, `word_tokens`. Defaults reproduce the old behaviour exactly (baselines rerun: identical numbers). Labels reach the binarizer only from each fold's training part (spy test).
- Word features are stored with an internal marker so they never collide with n-grams; reports show them as `word:SELECT`.
- 29-setting ablation on v5 (CV only). Main result: label-aware selection at M=500 lifts LR 0.917 → 0.958 (chi2) / 0.959 (class_balanced), Naive Bayes 0.857 → 0.952 / 0.960. Paired Δ ≈ 3× its fold std: a real effect.
- SQL F1 0.87 → 0.99 (M5 fixed by selection; SQL's first picks are `SELECT` fragments). JavaScript 0.86 → 0.92, C++ 0.88 → 0.92.
- No gain, within noise: M=1000/2000 once selection is on, `min_df` 5/20, word tokens (+10% binarize time).
- Best single setting by CV: chi2, M=2000 (LR 0.964), but top settings are all within fold noise (0.956-0.964); choosing among them is step 4.
- Tests: 216 passing.

### 2026-09-24 — M2 part 2 step 2: ablation runner and first results
- `ablations.py` + `codelangtm ablate [--study NAME] [--config] [--out]`, driven by `configs/ablations.yaml`. Each study varies feature options (every combination), the rest from the base; each unique setting is evaluated once.
- Train-only, repo-grouped 5-fold CV; the test set is never used (ablations choose features). Binarizer fit once per fold per setting and shared by all models; a test proves this gives exactly the Pipeline's CV scores.
- Reports paired Δ vs base (mean ± std of per-fold differences), per-language out-of-fold F1, vocabulary size used, binarize ms/snippet.
- `baselines.dataset_meta` extracted and shared by both reports.
- Sanity check: base row equals the baselines (LR 0.917, NB 0.857).
- Findings (LR): M matters most (M=100 -0.131, M=1000 +0.014, still rising); bigrams alone beat 2+3 (0.944, +0.027 ± 0.021) because frequency ranking fills slots with generic 3-grams like `ing`, `ion` (issues-and-fixes M6); delimiters +0.012 ± 0.014 (within noise) at ~40% of binarize time (M2 entry).
- No feature config frozen yet: step 3 tests discriminative selection and keyword features first.
- Tests: 197 passing.

### 2026-09-24 — M2 part 2 step 1: YAML experiment configs
- `feat/baselines` merged (M2 part 1). New branch `feat/ablations`.
- `config.py`: `FeatureConfig` (M, n-gram sizes, delimiters) and `BaselinesConfig` (data, out, models, seed, repeats, features). Strict loading: unknown keys and wrong types are errors, so a typo cannot silently fall back to a default.
- `configs/baselines.yaml` documents the current setup; `codelangtm baselines --config configs/baselines.yaml` regenerates `docs/results.md`. CLI flags still work and override the file.
- `results.md` now records the config file, which flags overrode it, and a hash of the effective settings, plus the full binarizer options (n-gram sizes and delimiters were hard-coded before).
- Dependency: `pyyaml>=6`.
- Check: rerun from the config reproduces v5 exactly (LR CV 0.917 ± 0.008, test 0.943, repeated 0.931 ± 0.007). Tests: 177 passing.

### 2026-09-24 — Stable split and repeated-split reporting (dataset v5)
- Decision (option A after weighing A/B/C, see issues-and-fixes M4): stable hash-based per-language split + repeated test splits for baselines now and for the final TM comparison.
- Rejected pure hashing after simulating it on v4 (Java 2 test repos, Python 9). Chosen: per language, hash-ordered repos, first round(n × 0.2) to test; rest cut into 5 equal folds by a second hash.
- `splits.py`: `stable_split`, `StableSplit`, `repo_languages`; `data build` uses it (`--salt`, default `codelangtm-v1`); manifest records `split: stable-hash`.
- `baselines.py`: `repeated_split_f1`; `--repeats 10` (default) → "Repeated test macro-F1" column (mean ± std, min-max).
- Tests: 155 passing; properties tested: exact per-language balance, isolation between languages, at most one repo moved per added/removed repo (50 randomised trials).
- v5 results: LR CV 0.917 ± 0.008, test 0.943, repeated test 0.931 ± 0.007 (CV and repeated test now agree). Each language has 5 test repos (SQL 4).

### 2026-09-24 — Dataset v4 and test-variance finding
- HTML re-collected with the embedded-language rule: 100 snippets from 25 repos (13 repos skipped). Dataset v4: 869 snippets (train 696, test 173).
- Baselines on v4: logistic regression still best by CV (0.923 ± 0.014, was 0.920); test fell 0.951 → 0.896.
- Investigated: the HTML change reshuffled test repos for every language. Same data and model over 10 split seeds: test macro-F1 0.870-0.978 (mean 0.929 ± 0.030). Test drop is split luck, not a regression. CV is the primary metric. (issues-and-fixes M4)
- Diagnose on v4: 5 of 696 candidates, all correctly labelled. 3 are repetitive list-like code predicted as SQL: SQL is partly learned as "data rows" because n-grams miss keywords. (issues-and-fixes M5)
- Dataset card updated to v4 (counts, drops, hashes, history, split variance, new limitations); ledger, README, roadmap updated.

### 2026-09-24 — HTML embedded-language rule
- `labels.py`: `HTML_TAG` now requires a real tag (not `i < x` or `a<b` comparisons); HTML windows with < 20% markup lines dropped as `mostly embedded script/style`.
- On v3 raw data: exactly the 9 predicted HTML windows dropped (5 mostly script, 4 no real tags); 0 other languages affected. Issues-and-fixes D5 closed.
- Tests: 145 passing.

### 2026-09-24 — M2 data checks with the baseline
- `diagnostics.py` + `codelangtm diagnose`: out-of-fold logistic-regression probabilities (train only) → confident-learning label-issue candidates; shortcut probe = top features per language with repo spread and cross-language share. Review file `data/processed/diagnostics.md`.
- Result: 6 of 689 candidates (0.9%). 3 HTML windows are inline `<script>` code (tag check fooled by `<` comparisons), 1 JS window is mostly an HTML template, 2 are correct but hard (Rust FFI mirroring C, bare Java interface).
- Shortcut probe: none; top features are real syntax in 13-20 repos; test idioms not among them.
- Measured: 9 of 92 HTML windows have < 20% markup lines. Fix proposed, decision pending. Details: issues-and-fixes M3, D5.
- Tests: 140 passing.

### 2026-09-24 — M2 baselines
- `features.py`: `Binarizer` is now a scikit-learn transformer so the vocabulary is refit inside each CV fold (fitting it once on all train would leak validation n-grams). Alphabetical tie-break makes the vocabulary independent of input order; JSON save/load; `use_delimiters` switch for ablations; transform uses n-gram set lookups instead of substring search.
- `baselines.py` + `codelangtm baselines`: 5 models as `Pipeline([Binarizer, model])`, 5-fold repo-grouped CV (model selection) then one test evaluation; macro-F1, accuracy, per-language F1, confusion matrix, fit time, end-to-end latency, pickled size → generated `docs/results.md` with dataset hashes and library versions.
- Decision: Bernoulli NB instead of Multinomial NB (features are presence flags, not counts).
- Results (dataset v3, M=500): logistic regression best by CV (0.920 ± 0.010, test 0.951); linear SVM 0.905; random forest 0.897 (10.8 MB); naive Bayes 0.853; decision tree 0.724 (SQL F1 0.22).
- Findings: main confusion is C++ → Rust (4 of 21 C++ test snippets; shared `::`, `->`, braces); Go scores 1.00 everywhere (distinctive syntax, gofmt tabs); test F1 > CV for most models because test has only 39 repos, so CV is the more reliable number.
- Latency: binarization takes ~0.31 ms/snippet, essentially all of the end-to-end time; model inference is negligible. The < 0.1 ms target depends on feature extraction speed (M6 C export).
- Tests: 134 passing, all offline; includes a spy test proving the vocabulary never sees test or held-out fold snippets.

### 2026-09-24 — Dataset card, M1 Stage A closed
- Manual re-review of v3 audit samples: no problems found.
- `docs/dataset-card.md` (Datasheets for Datasets structure): composition per language, collection and cleaning steps with drop counts, splits, licensing (MIT 505, Apache-2.0 315, BSD 41), known biases (SQL star floor 10 and imbalance, test-file share up to 40% for Go, 20-50 line windows only), intended use, reproduction commands, v3 file hashes.
- `docs/data-sources.md` ledger rows for the two GitHub collection runs; roadmap M1 ticked, Stage B items marked deferred; README status updated.
- Stage A target was 1,000 snippets; reached 861 (SQL limited by permissive repos). Accepted and documented.

### 2026-09-24 — Stage A data quality: templates, audit, cut comments (M1)
Details and numbers in [docs/issues-and-fixes.md](docs/issues-and-fixes.md) (D1-D4, W3).
- Stage A v1: 829 snippets; SQL only 40 from 10 repos. SQL top-up with `--min-stars 10` → 77 from 21 repos.
- Template filter (`labels.py`): SQL/HTML windows > 30% Jinja/Liquid/ERB lines dropped (21: HTML 15, SQL 6).
- `audit.py` + `codelangtm data audit`: per-language stats (repo share, comment ratio, test-file share, licenses), flags, and `data/processed/audit.md` with 10 seeded samples per language + GitHub permalinks. Header shows generation time.
- Manual review found windows cutting through comments/docstrings (40 of 866). Added `syntax.py` scanner; windows now snap to boundaries outside comments/strings; label check flags cut snippets. Re-collected → v3: 861 snippets, 0 drops, audit flags: none.
- Watch: Go test-file share 40% (idiomatic `_test.go`), SQL still smallest class (77 vs 124).
- Tests: 117 passing, ruff clean.

### 2026-09-24 — Dataset build command (M1)
- Collector smoke test passed and pushed (`feat/github-collector`).
- `build.py`: `build_dataset` = load all `.jsonl` under sources → label check → cross-source dedup (training data first, so wild copies of training snippets are dropped) → group-by-repo train/test → k-fold ids for train → leakage checks (train/test, main/wild) → thin-language warnings.
- Writes `data/processed/{train,test,wild}.jsonl`, `folds.json`, `dataset.json` (counts, drops, params, SHA-256 per file).
- CLI: `codelangtm data build [--source DIR] --out data/processed --test-size 0.2 --folds 5 --seed 0`.
- Tests: 70 passing, ruff clean.

### 2026-09-24 — GitHub collector (M1)
- PR #1 merged (schema, windows, dedup, labels, splits).
- Created read-only fine-grained GitHub token, stored in `GITHUB_TOKEN` user env var (verified: 5000/hr limit).
- `github.py`: `GitHubClient` (token from env, rate-limit + 5xx retry, plain 403 fails fast, commit-pinned disk cache), `search_repos` (license allowlist, star bands, round-robin), `candidate_files` (extension + vendored/generated/minified/size filters), `collect_repo` (1 random window per file, label check), `collect_language` (repo caps, skips empty/broken repos, dedup, report).
- CLI: `codelangtm collect github [--language X] --repos 25 --per-repo 5 --min-stars 50` writes `<lang>.jsonl` + `<lang>.manifest.json`.
- Deps: `httpx` in new `collect` extra (+ dev group). `.gitignore`: `data/cache/`, `.env`.
- Tests: 63 passing, all offline via `httpx.MockTransport`; includes a test that the token never appears in outputs.
- Note: use `uv sync --extra tm --extra collect`; plain `uv sync` removes TMU from the venv.

### 2026-09-24 — Window extractor, dedup, label check, group split (M1)
- `windows.py`: `extract_windows` (seeded, non-overlapping 20-50 line windows) + `is_low_signal` (blank / license / comment-only windows dropped).
- `dedup.py`: `dedup` = exact (whitespace-normalized SHA-1) + near-duplicate (MinHash LSH, verified by Jaccard >= 0.8 on 5-token shingles). 3,000 snippets in ~3s.
- `labels.py`: `check_label` / `filter_labels` — binary, minified, non-ASCII, extension mismatch (`.h` decided by content), other-language red flags, required HTML/SQL markers; wild snippets skip the extension check. Returns drop-reason counts.
- `splits.py`: `group_split`, `group_kfold` (StratifiedGroupKFold by repo), `check_no_leakage`, `separate_wild`.
- Tests: 50 passing, ruff clean.
- Watch on real data: dedup threshold (boilerplate-heavy languages), JS red flag drops Flow-typed JS, SQL keyword rule drops pure INSERT-value windows.

### 2026-09-24 — TMU check + snippet schema (M1 start)
- `uv sync --extra tm` installed `tmu 0.8.3`. First toy run crashed with numpy 2.5 (`OverflowError: Python integer -1 out of bounds for uint32`).
- Fix: `tm` extra now pins `numpy<2` and `scipy<1.14` (numpy 1.26.4 + scipy 1.13.1). Toy XOR training works (93.5% acc, CPU backend; pycuda warning harmless).
- Added `src/codelangtm/data.py`: `Snippet` frozen dataclass (fields per schema), validation (known language, 20-50 line window rule left to the extractor), JSONL `save_snippets` / `load_snippets`, `group_key` = repo.
- Added `tests/test_data.py`.
- Ruff clean, pytest passing.

### 2026-09-24 — Project setup (M0)
- Wrote project brief into structure; chose Public + MIT license.
- Python via `uv`, pinned to 3.12 (system has 3.14; TMU needs C compiler and likely lacks 3.14 wheels). TMU kept as optional extra `tm`.
- Package `src/codelangtm/`:
  - `features.py`: `Binarizer` (top-M char 2/3-grams + delimiters), `expand_literals` (2M literals)
  - `model.py`: `TMLanguageClassifier` wrapper over TMU `TMClassifier` (fit/predict)
  - `rules.py`: `format_rule` done; `extract_rules` stub
  - `export_c.py`: stub
  - `cli.py`: `--version` only
- `tests/test_smoke.py`: 3 tests passing (version, CLI, binarizer + literals)
- Repo files: `.gitignore` (data ignored), `.gitattributes`, `.editorconfig`, `LICENSE`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `CLAUDE.md`
- GitHub: CI workflow (ruff + pytest), issue and PR templates
- Docs: `architecture.md`, `roadmap.md`, `data-sources.md`, `github-setup.md`
- User pushed to `github.com/kolonel-git/CodeLangTM` (fixed non-fast-forward rejection by pulling remote initial commit).
- Fixed stale `.venv` error (`uv python install 3.12`, `uv sync --reinstall`).

### 2026-09-24 — Roadmap refinement
Decisions made (recorded in roadmap):
- **Data:** GitHub permissive repos + public datasets (The Stack, CodeSearchNet); no synthetic generator
- **Size:** staged, 1,000 then 10,000+
- **Snippets:** contiguous 20-50 line windows
- **Languages:** 8 core + stretch set (C, C#, TypeScript, Kotlin, PHP, Ruby)
- **Eval:** group split by repo + stratified k-fold + held-out "wild" test set
- **Baselines:** NB, Decision Tree, Logistic Regression, linear SVM, Random Forest
- **Features:** ablation over M, n-gram sizes, token-level features
- **Reproducibility:** YAML configs, fixed seeds, results in `docs/results.md`
- **Tuning:** grid over s/T/N_c, then Optuna refinement
- **Explainability:** rules, `--explain`, rule metrics, HTML report and gallery
- **Deployment:** pure C export + CLI (VS Code ext, WASM go to Future)
- **Benchmarks:** latency + model size only
- **Workflow:** feature branches + PRs; GitHub Projects dropped in favor of this file
- **Showcase:** README results, blog write-up, demo notebook
- **Timeline:** no deadlines

## Results log
Record each run here: date, config, dataset version, macro-F1 (CV / test / wild), latency, model size. Full tables in [docs/results.md](docs/results.md).

| Date | Experiment | Dataset | Macro-F1 (CV / test) | Notes |
| --- | --- | --- | --- | --- |
| 2026-09-24 | Baselines, M=500, 2/3-grams + delimiters | v3 | LR 0.920 / 0.951 (best) | SVM 0.905, RF 0.897, NB 0.853, DT 0.724; 0.31 ms/snippet (binarization-bound) |
| 2026-09-24 | Baselines, same config | v4 | LR 0.923 / 0.896 (best) | SVM 0.913, RF 0.907, NB 0.876, DT 0.726; test not comparable to v3 (split reshuffled; seed spread 0.870-0.978) |
| 2026-09-24 | Baselines, same config, stable split + 10 repeated splits | v5 | LR 0.917 / 0.943; repeated 0.931 ± 0.007 (best) | SVM 0.908 (rep 0.922), RF 0.897 (rep 0.924), NB 0.857 (rep 0.880), DT 0.729 (rep 0.745) |
| 2026-09-24 | Ablations (`configs/ablations.yaml`, 11 settings, CV only) | v5 | LR best: M=500, bigrams only, 0.944 / - | M=1000 2+3: 0.931; delimiters off: 0.905; NB best n=4: 0.897; see [docs/ablations.md](docs/ablations.md) |
| 2026-09-24 | Ablations + selection, word tokens, M=2000 (29 settings, CV only) | v5 | LR: class_balanced M=500 0.959, chi2 M=2000 0.964 / - | NB class_balanced M=500 0.960 (was 0.857); word tokens and larger M within noise |
| 2026-09-25 | Baselines, frozen features (class_balanced, M=500, 2+3-grams, no delimiters) | v5 | NB 0.963 / 0.959; repeated 0.968 ± 0.009 (best by CV) | LR 0.953 / 0.971 (rep 0.968), SVM 0.946 / 0.964, RF 0.950 / 0.944, DT 0.843 / 0.832; 0.19 ms/snippet (was 0.31), classifier peak memory 1.4-4.3 MB, binarizer 72 MB |
| 2026-09-28 | B5 process-level resources (fresh process per job, 3 repeats) | v5 | - | one snippet median: LR 0.089, TM100 0.199, NB 0.215, TM400 0.343 ms; batch 0.031-0.057 ms; training memory ~76 MB for all (binarizer); TM 400 fit 7.7 s, 164 KB |
| 2026-09-28 | B3 TM vs baselines, full protocol, seeds 1-5 | v5 | TM 400: 0.949 / 0.958, repeated 0.966 ± 0.009; TM 100: 0.933 / 0.946, repeated 0.949 | NB 0.963 / 0.959 / 0.968; TM 400 − NB repeated −0.001 [−0.012, +0.010] p = 0.82; TM 100 − LR repeated −0.019, p = 0.016; TM 400 166 KB, 0.081 ms/snippet |
| 2026-09-27 | B4 TM training curves, 5 folds x seeds 1-5, 150 epochs (CV only) | v5 | tm_400 (N_c=400/T=100/s=5): 0.951 at epoch 120 (levels off at 19: 0.946) / -; tm_100: 0.933 at epoch 142 (44: 0.928) / - | NB same folds 0.963; B1's 0.962 was seed 42 (lucky); 0.05 / 0.03 s/epoch |
| 2026-09-27 | B1 spike: TMU, frozen features, 150 epochs (CV only) | v5 | N_c=100/T=30/s=3.5: 0.925-0.940 / -; N_c=400/T=100/s=5: 0.962 / - | last-20-epoch mean over 5 folds; NB same folds 0.963; 0.03-0.04 s/epoch; TMU predict 0.07-0.09 ms/snippet; N_c=400 clauses 400 KB dense / ~80 KB sparse |
| 2026-09-25 | Same, after faster binarization | v5 | identical scores | 0.08-0.13 ms/snippet end-to-end (linear models / NB) on a loaded machine; binarize alone 5.4-6.0× faster than before (interleaved A/B); RF 0.75 |

## Targets
Macro-F1 >= 96% (8 languages) · < 0.1 ms per snippet · < 500 KB model.

## Session template
```
### YYYY-MM-DD — <topic>
- Done:
- Decisions:
- Problems / fixes:
- Next:
```
