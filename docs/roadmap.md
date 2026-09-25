# Roadmap

Ordered milestones, no fixed deadlines. Each milestone maps to a GitHub milestone; each task checkbox becomes an issue. Work happens on feature branches merged via PR.

## Principles
- **Real data only.** No synthetic snippet generators. Every snippet records its source repo and license.
- **No leakage.** Splits are grouped by source repo; a separate "wild" test set comes from sources never seen in training.
- **Reproducible.** YAML configs, fixed seeds, results committed to `docs/results.md`; every number regenerates from one command.
- **Honest claims.** Targets (macro-F1 >= 96%, < 0.1 ms/snippet, < 500 KB) are reported as measured, including misses.

## M0 — Foundations (done)
- [x] Package layout, `uv` env pinned to Python 3.12, MIT license
- [x] CI (ruff + pytest), issue/PR templates, docs skeleton
- [x] Binarizer + literal expansion prototype

## M1 — Data pipeline
**Goal:** a clean, licensed, leak-free dataset with documented provenance.

Languages: 8 core (Python, C++, Java, JavaScript, Rust, Go, SQL, HTML). Stretch set added in Stage B: C, C#, TypeScript, Kotlin, PHP, Ruby, to stress confusable pairs (JS/TS, C/C++, Java/C#).

**Status:** Stage A complete (dataset v5, 869 snippets, stable split, see [dataset-card.md](dataset-card.md)). Items marked *Stage B* are deferred and do not block M2.

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

## M2 — Features & baselines
**Goal:** justify feature design with ablations; establish the bar the TM must beat.

**Status:** baselines, data checks, stable split, ablations and the frozen feature config done. Bar for the TM (dataset v5, frozen features, see [results.md](results.md)): Naive Bayes CV macro-F1 0.963, repeated test 0.968 ± 0.009; logistic regression 0.953 / 0.968. Label-aware vocabulary selection was the key (0.917 → ~0.96, [ablations.md](ablations.md)). Faster binarization left.

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
- [ ] Faster binarization (currently ~0.31 ms/snippet, ~100% of end-to-end latency)

**Exit:** ablation table and all five baselines reproducible from one command; best baseline macro-F1 recorded.

## M3 — Tsetlin Machine training
**Goal:** working TMU classifier with fair comparison.

- [x] Verify TMU install: TMU 0.8.3 builds natively on Windows with `numpy<2` (see [architecture.md](architecture.md) install notes); CUDA optional
- [ ] `TMLanguageClassifier` fit/predict/save/load with tests
- [ ] Baseline run: N_c=100, T=30, s=3.5
- [ ] Training curves (accuracy vs epoch); throughput
- [ ] TM vs baselines table on CV / test / wild, using the repeated-split protocol
- [ ] Resource comparison, same protocol for every model (baselines already report the first block in [results.md](results.md)):
  - training: wall time, CPU time, peak memory (Python heap for baselines; process-level peak RSS, measured in a subprocess, for both baselines and the TM, since TMU allocates in C), epochs to converge;
  - model: size in KB (pickled, and for the TM also the bit-packed clause size that the C export will use), vocabulary size, number of clauses and average literals per clause (TM) or non-zero weights (linear models);
  - inference: latency (median, p95) and throughput, split into binarize vs predict;
  - reported as measured, including where the TM loses
- [ ] Error analysis: confusable pairs, short snippets

**Exit:** TM results in `docs/results.md`, comparable to baselines on the same splits.

## M4 — Tuning & compression
**Goal:** best accuracy per byte and per rule.

- [ ] Grid search over s, T, N_c; heatmaps saved to `docs/figures/`
- [ ] Optuna refinement: epochs, drop-clause rate, literal budget (seeded)
- [ ] Drop clause + literal budgeting to prune redundant literals
- [ ] Trade-off curves: macro-F1 vs model size vs average rule length
- [ ] Pick and freeze a release configuration

**Exit:** frozen config in `configs/`; trade-off plots committed.

## M5 — Explainability
**Goal:** make the interpretability claim demonstrable.

- [ ] Rule extraction: clauses -> `has("def ") AND NOT has(";")` per class (`rules.extract_rules`); name features via `Binarizer.get_feature_names_out()` so word features read `word("SELECT")`, not the internal marker
- [ ] Per-prediction explanation: `codelangtm predict --explain` shows winning clauses and vote totals
- [ ] Rule quality metrics: length, coverage, precision, overlap between classes
- [ ] HTML report: matched n-grams highlighted in the snippet, votes per language
- [ ] Per-language rule gallery (top rules by coverage/precision)

**Exit:** for any snippet, the report shows exactly which rules drove the prediction.

## M6 — Deployment & benchmarks
**Goal:** zero-dependency runtime meeting size/latency targets.

- [ ] Pure C export of trained clauses (`export_c.export_c`)
- [ ] Equivalence test: C predictions == Python predictions on the full test set
- [ ] CLI: `codelangtm predict <file|->`
- [ ] Benchmark script: latency per snippet (median, p95) and compiled model size
- [ ] Results vs targets in README (< 0.1 ms, < 500 KB), reported as measured

**Exit:** C runtime builds with a plain compiler; benchmarks reproducible.

## M7 — Showcase & release
- [ ] README: results table, figures, rule examples
- [ ] Demo notebook (`notebooks/demo.ipynb`)
- [ ] Write-up / blog post: method, ablations, findings, limitations
- [ ] Tag `v0.1.0`, update CHANGELOG

## Future
- Relational TM over ASTs (Horn clauses)
- Convolutional TM over 2D code layout
- VS Code extension; WebAssembly browser demo
- FPGA/ASIC via MATADOR
- Federated TM (FedTMOS)
- Sparse TM for vulnerability detection
- Energy measurement (RAPL or proxy)

## Definition of done (per task)
Code + tests + docs updated, CI green, results regenerated if affected, issue closed by PR.
