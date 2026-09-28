# Roadmap

Ordered milestones, no fixed deadlines. The project is a portfolio piece built so that it can later become a research write-up: results are reported with research-level care from M3 on (see Principles). Each milestone maps to a GitHub milestone; each task checkbox becomes an issue. Work happens on feature branches merged via PR.

## Principles
- **Real data only.** No synthetic snippet generators. Every snippet records its source repo and license.
- **No leakage.** Splits are grouped by source repo; a separate "wild" test set comes from sources never seen in training.
- **Reproducible.** YAML configs, fixed seeds, results committed to `docs/results.md`; every number regenerates from one command.
- **Honest claims.** Targets (macro-F1 >= 96%, < 0.1 ms/snippet, < 500 KB) are reported as measured, including misses.
- **Research-ready results (from M3).** Every reported TM number is a mean over 5 seeds with its spread; headline comparisons with the baselines use a paired significance test on the same folds; the raw per-seed, per-fold runs are saved as JSON next to the reports. Design choices are made on CV only.
- **Accuracy and readability together.** Report the most accurate TM and the smallest TM within about 1 F1 point of it, and the curve between them (M4).
- **Plain-language docs.** New concepts are explained in [concepts.md](concepts.md).

## M0 — Foundations (done)
- [x] Package layout, `uv` env pinned to Python 3.12, MIT license
- [x] CI (ruff + pytest), issue/PR templates, docs skeleton
- [x] Binarizer + literal expansion prototype

## M1 — Data pipeline
**Goal:** a clean, licensed, leak-free dataset with documented provenance.

Languages: 8 core (Python, C++, Java, JavaScript, Rust, Go, SQL, HTML). Stretch set added in Stage B: C, C#, TypeScript, Kotlin, PHP, Ruby, to stress confusable pairs (JS/TS, C/C++, Java/C#).

**Status:** Stage A complete (dataset v5, 869 snippets, stable split, see [dataset-card.md](dataset-card.md)). Items marked *Stage B* are deferred: they are scheduled **after M3 and before M4 tuning** (decided 2026-09-27), so the TM is first compared on v5 and then tuned on the larger dataset.

- [x] Define snippet record schema: `text`, `language`, `repo`, `commit`, `path`, `license`, `source`, `start_line`, `end_line` (see [data-sources.md](data-sources.md))
- [x] Collector: GitHub API, permissive licenses only (MIT / Apache-2.0 / BSD)
- [ ] *Stage B:* loaders for public datasets (The Stack, CodeSearchNet) with license filter
- [x] Window extractor: contiguous 20-50 line windows from real files; skip near-empty / license-header-only windows; never cut comments/strings
- [x] Dedup: exact hash + near-duplicate (MinHash or shingle Jaccard)
- [x] Label sanity check (extension vs content; drop mislabeled files, e.g. `.h` C vs C++; template-heavy SQL/HTML)
- [x] Splits: group-by-repo train/test + 5-fold repo-grouped CV (stable hash-based per language since v5)
- [x] Dataset audit: per-language stats, flags, manual sample review
- [ ] *Stage B:* wild test set from unseen sources (StackOverflow, blogs, official docs); license/attribution logged
- [x] **Stage A:** target 1,000 snippets (~125/language); reached 869 in v4 (SQL limited by available permissive repos)
- [ ] *Stage B:* scale to 10,000+ and add stretch languages
- [x] Dataset card `docs/dataset-card.md`: counts, class balance, length distribution, known biases

**Exit:** `codelangtm data build` reproduces the dataset from configs; no repo appears in both train and test; every row has a license.

## M2 — Features & baselines (done)
**Goal:** justify feature design with ablations; establish the bar the TM must beat.

**Status:** baselines, data checks, stable split, ablations and the frozen feature config done. Bar for the TM (dataset v5, frozen features, see [results.md](results.md)): Naive Bayes CV macro-F1 0.963, repeated test 0.968 ± 0.009; logistic regression 0.953 / 0.968. Label-aware vocabulary selection was the key (0.917 → ~0.96, [ablations.md](ablations.md)); binarization is 5-6× faster. Merged (PR #5). Narrative with figures: [report.md](report.md).

- [x] Harden Binarizer (sklearn transformer refit per fold, deterministic vocabulary, save/load, fast transform)
- [x] Token-level features: whole-word tokens (`word_tokens`) tested, no gain over label-aware n-gram selection
- [x] Ablation runner + first study: M in {100, 250, 500, 1000}; n-grams in {2, 3, 4, mixed}; delimiters on/off ([ablations.md](ablations.md))
- [x] Ablation: label-aware vocabulary selection (chi2, class_balanced; issues-and-fixes M6), word tokens, M=2000, min_df
- [x] Freeze the feature config for M3 and rerun baselines with it (`configs/baselines.yaml`: class_balanced, M=500, 2+3-grams, no forced delimiters)
- [x] Resource metrics in the baselines report (fit CPU, peak memory, size split, throughput), reusable for the TM
- [x] Baselines on identical features: Bernoulli NB (binary features, not Multinomial), Decision Tree, Logistic Regression, linear SVM, Random Forest
- [x] Metrics: macro-F1, per-language F1, confusion matrix, on CV, test, and wild sets (wild when available)
- [x] YAML config system + seed control (`configs/*.yaml`, settings hash in results)
- [x] Auto-generated `docs/results.md`
- [x] Confident-learning check (out-of-fold predictions) + shortcut probe (top features per language)
- [x] Embedded-language rule for HTML (windows that are mostly `<script>`)
- [x] Stable split: per-language hash-based repo assignment; repeated-split test reporting (dataset v5)
- [x] Faster binarization: 5.4-6.0× faster transform, identical output ([issues-and-fixes](issues-and-fixes.md) M2); the < 0.1 ms end-to-end target is confirmed on a quiet machine in M6

**Exit:** ablation table and all five baselines reproducible from one command; best baseline macro-F1 recorded.

## M3 — Tsetlin Machine training
**Goal:** working TMU classifier with fair comparison, and every learned clause inspectable.

**Plan:** branch `feat/report` (report infrastructure, done), then branch `feat/tm-training` (steps B1-B7 below). Order changed on 2026-09-27: B4 (training curves) runs before B3, so the full protocol run uses an epoch count chosen on CV.

- [x] Results report: JSON sidecars for generated reports, deterministic figures (`figures.py`, `codelangtm report` → `docs/figures/`), hand-written [report.md](report.md) (branch `feat/report`)
- [x] Verify TMU install: TMU 0.8.3 builds natively on Windows with `numpy<2` (see [architecture.md](architecture.md) install notes); CUDA optional
- [x] B1 spike (not committed): 0.03-0.04 s/epoch; NumPy prediction matches TMU exactly (empty clauses output 0); planned N_c=100/T=30/s=3.5 reaches ~0.93 CV, N_c=400/T=100/s=5 0.962 with seed 42, which B4 showed to be a lucky seed (0.948 over seeds 1-5; [issues-and-fixes](issues-and-fixes.md) M8)
- [x] B2 `TMLanguageClassifier` (scikit-learn estimator: fit/partial_fit/predict/decision_function) + TMU-free NumPy `TMState` (predict, save/load in a documented, portable format that the C export and the web demo can read) with tests
- [x] B3 (after B4) main run N_c=400, T=100, s=5 (from B1, CV only) plus the planned N_c=100, T=30, s=3.5 as a reference row, through the same CV / test / repeated-split protocol, each over 5 seeds (mean ± std), with a paired significance test against Naive Bayes on the same folds (Nadeau-Bengio corrected resampled t-test) → `docs/tm-results.md` + `.json` (raw per-seed runs included), via `codelangtm tm-results` (a dedicated command instead of `baselines --config`, because TM runs have seeds and settings). Result: TM 400 CV 0.949 / test 0.958 / repeated 0.966 vs Naive Bayes 0.963 / 0.959 / 0.968; no significant difference (repeated splits: −0.001, p = 0.82); TM 100 significantly behind logistic regression (−0.019, p = 0.016)
- [x] B4 `TMConfig` + `configs/tm.yaml`; `codelangtm tm-curve`: training curves (macro-F1 vs epoch on held-out folds and train, 5 folds x 5 seeds, up to 150 epochs, clause formation per epoch); epoch = peak of the 11-epoch smoothed held-out curve (rule `max`; the plateau rule, first epoch within 0.005 of the peak, is reported beside it; no test data) → `docs/tm-curves.md` + `.json` + figures. Result: 400 clauses → 120 epochs (held-out 0.951; levels off at 19 with 0.946), 100 clauses → 142 epochs (0.933; levels off at 44); Naive Bayes 0.963 on the same folds
- [x] B4b `codelangtm tm-train` (train one model from the config on all of train and save it, ~8 s) and `codelangtm tm-select` (the official model `models/tm.json` = the B3 seed with the median CV score, no test data: seed 4 of `tm_400`). Retraining seed 4 reproduces B3's saved model exactly
- [x] B5 resource comparison, same protocol for every model (`codelangtm resources` → `docs/resources.md`: fresh process per job, 3 repeats, median; result: training memory ~76 MB for every model (binarizer-dominated), TM 400 one-snippet latency 0.34 ms in NumPy vs 0.09 ms logistic regression, batch 0.057 ms; model file 164 KB) (baselines already report the first block in [results.md](results.md)):
  - training: wall time, CPU time, peak memory (Python heap for baselines; process-level peak RSS, measured in a subprocess, for both baselines and the TM, since TMU allocates in C), epochs to converge;
  - model: size in KB (pickled, and for the TM also the bit-packed clause size that the C export will use), vocabulary size, number of clauses and average literals per clause (TM) or non-zero weights (linear models);
  - inference: latency (median, p95) and throughput, split into binarize vs predict;
  - reported as measured, including where the TM loses
- [ ] B6 clause inspector (pulled forward from M5): every clause as a readable rule (`docs/clauses.md`, `clauses.json`), per-clause statistics on train, signature features per language, literal-usage heatmap, class-overlap matrix, clause formation over epochs, `codelangtm explain` traces one prediction to the clauses that fired
- [ ] B6b error analysis: confusable pairs, disagreement with Naive Bayes (short snippets moved to Future)
- [ ] B7 TM sections of [report.md](report.md) (results, curves, resources, "how the TM works inside")

**Exit:** TM results in `docs/tm-results.md`, comparable to baselines on the same splits; reported as measured, including where the TM loses; every learned clause is inspectable and any prediction can be traced to the clauses that fired.

## M4 — Tuning & compression
**Goal:** best accuracy per byte and per rule. Runs on the Stage B dataset (after M3).

- [ ] Grid search over s, T, N_c; heatmaps saved to `docs/figures/`
- [ ] Optuna refinement: epochs, drop-clause rate, literal budget (seeded)
- [ ] Drop clause + literal budgeting to prune redundant literals
- [ ] Trade-off curves: macro-F1 vs model size vs average rule length
- [ ] Smallest model within ~1 F1 point of the most accurate one (fewer clauses, literal budget), reported next to it
- [ ] Pick and freeze a release configuration

**Exit:** frozen config in `configs/`; trade-off plots committed.

## M5 — Explainability
**Goal:** make the interpretability claim demonstrable.

Rule extraction, per-prediction explanation and the per-language gallery move into M3 (B6, clause inspector). M5 keeps what builds on them: rule quality on the test set, the HTML report and a polished `predict --explain`.

- [ ] Rule extraction: clauses -> `has("def ") AND NOT has(";")` per class (`rules.extract_rules`); name features via `Binarizer.get_feature_names_out()` so word features read `word("SELECT")`, not the internal marker
- [ ] Per-prediction explanation: `codelangtm predict --explain` shows winning clauses and vote totals
- [ ] Rule quality metrics: length, coverage, precision, overlap between classes
- [ ] HTML report: matched n-grams highlighted in the snippet, votes per language
- [ ] Per-language rule gallery (top rules by coverage/precision)

**Exit:** for any snippet, the report shows exactly which rules drove the prediction.

## M6 — Deployment & benchmarks
**Goal:** zero-dependency runtime meeting size/latency targets, plus a web demo.

- [ ] Pure C export of trained clauses (`export_c.export_c`)
- [ ] Equivalence test: C predictions == Python predictions on the full test set
- [ ] CLI: `codelangtm predict <file|->`
- [ ] Benchmark script: latency per snippet (median, p95) and compiled model size
- [ ] Results vs targets in README (< 0.1 ms, < 500 KB), reported as measured
- [ ] Web demo: paste code, see the prediction, the votes and the clauses that fired. Technology decided later (options: static page with JavaScript inference, WebAssembly from the C runtime, or a hosted Python app); until then the saved model format stays portable and documented

**Exit:** C runtime builds with a plain compiler; benchmarks reproducible.

## M7 — Showcase & release
- [ ] README: results table, figures, rule examples
- [ ] Demo notebook (`notebooks/demo.ipynb`)
- [ ] Write-up / blog post: method, ablations, findings, limitations
- [ ] Research write-up (later): paper-style version of the report, built on the saved multi-seed runs
- [ ] Tag `v0.1.0`, update CHANGELOG

## Future
- Short-snippet evaluation: accuracy on 1-10 line snippets (collect or cut short windows from the same repos, report F1 by snippet length). Skipped in M3 by decision; the current dataset only has 20-50 line windows ([dataset card](dataset-card.md), Known limitations)
- Relational TM over ASTs (Horn clauses)
- Convolutional TM over 2D code layout
- VS Code extension
- FPGA/ASIC via MATADOR
- Federated TM (FedTMOS)
- Sparse TM for vulnerability detection
- Energy measurement (RAPL or proxy)

## Definition of done (per task)
Code + tests + docs updated, CI green, results regenerated if affected, issue closed by PR.
