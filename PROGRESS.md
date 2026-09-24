# CodeLangTM — Progress Log

Living tracker. Update after every work session. Planning detail lives in [docs/roadmap.md](docs/roadmap.md); problems and how they were solved live in [docs/issues-and-fixes.md](docs/issues-and-fixes.md).

**Last updated:** 2026-09-24
**Current milestone:** M2 — Features & baselines
**Overall:** M0 complete, M1 Stage A complete (dataset v5: 869 snippets, stable split, [dataset card](docs/dataset-card.md)), M2 baselines + data checks done: bar to beat = LR CV macro-F1 0.917, repeated test 0.931 ([results](docs/results.md))

## Milestone overview

| Milestone | Status | Notes |
| --- | --- | --- |
| M0 Foundations | Done | Scaffold, CI, docs, roadmap |
| M1 Data pipeline | Stage A done | v5: 869 snippets, 196 repos, stable split; Stage B items deferred |
| M2 Features & baselines | In progress | Part 1 merged (baselines, data checks, stable split); part 2 (ablations) on `feat/ablations` |
| M3 TM training | Planned | |
| M4 Tuning & compression | Planned | |
| M5 Explainability | Planned | |
| M6 Deployment & benchmarks | Planned | |
| M7 Showcase & release | Planned | |

## Next steps (in order)

- [x] Verify GitHub repo settings: description, topics, replace `OWNER` in README badges, CI green ([docs/github-setup.md](docs/github-setup.md))
- [x] Check TMU installs: `uv sync --extra tm` (works natively on Windows, no WSL/Docker needed)
- [x] Re-run `uv run ruff check .` and `uv run pytest` after the `features.py` lint fix
- [x] M1: define snippet record schema in code (`Snippet` in `src/codelangtm/data.py`)
- [x] M1: dataset loader + tests on tiny fixtures (JSONL in `data.py`)
- [x] M1: window extractor (20-50 contiguous lines), dedup, label sanity check
- [x] M1: group-by-repo split + k-fold
- [x] M1: GitHub collector (permissive licenses only; token from env var)
- [x] M1: smoke-test collector against real GitHub (4 Python snippets, 2 Apache-2.0 repos, 0 drops)
- [x] M1: `codelangtm data build` command: merge sources → split → write train/test/wild
- [x] M1: Stage A collection run + review drop counts per language
- [x] M1: `data build` on Stage A data
- [x] M1: template-heavy SQL/HTML filter
- [x] M1: `data audit` command + manual sample review
- [x] M1: fix windows cutting comments/strings; re-collect (dataset v3)
- [x] M1: re-skim regenerated `audit.md` (no problems found)
- [x] M1: dataset card (`docs/dataset-card.md`) + ledger rows in [docs/data-sources.md](docs/data-sources.md)
- [x] M1: push `feat/data-build`, open PR, merge → M1 Stage A done (PR merged)
- [x] M2: harden Binarizer (sklearn transformer, deterministic vocabulary, save/load, faster transform)
- [x] M2: 5 baselines (NB, DT, LR, linear SVM, RF) with CV / test macro-F1 → `docs/results.md`
- [x] M2: confident-learning check (out-of-fold predictions) + shortcut probe (top features per language)
- [x] M2: embedded-language rule for HTML (drop windows with < 20% markup lines; stricter tag pattern)
- [x] M2: re-collect HTML → dataset v4; rerun build, baselines, diagnose; update dataset card + results
- [x] M2: stable split (hash-based per-language repo assignment) + repeated-split test reporting → dataset v5
- [ ] M3/M4: use repeated splits for the final TM vs baselines comparison
- [x] M2: push `feat/baselines`, open PR, merge (M2 part 1)
- [x] M2 ablations step 1: YAML config system (`configs/baselines.yaml`, strict loader, settings hash in results)
- [x] M2 ablations step 2: ablation runner (M, n-gram sizes, delimiters) → `docs/ablations.md`
- [x] M2 ablations step 3: label-aware vocabulary selection (fixes M6 and M5), word tokens (no gain), larger M (no gain once selection is on)
- [ ] M2 ablations step 4: run, analyse, freeze the feature config for M3
- [ ] M2: faster binarization (0.31 ms/snippet now; target budget < 0.1 ms end-to-end)
- [ ] Stage B (later): The Stack / CodeSearchNet loaders, stretch languages, embedded-language policy
- [ ] Wild set (later, collected by hand): StackOverflow / blogs / docs

## Blockers / open questions
- None.

## Done log

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
