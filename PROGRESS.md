# CodeLangTM — Progress Log

Living tracker. Update after every work session. Planning detail lives in [docs/roadmap.md](docs/roadmap.md); this file records what is done and what is next.

**Last updated:** 2026-09-24
**Current milestone:** M1 — Data pipeline
**Overall:** M0 complete, M1 in progress (schema done)

## Milestone overview

| Milestone | Status | Notes |
| --- | --- | --- |
| M0 Foundations | Done | Scaffold, CI, docs, roadmap |
| M1 Data pipeline | Next | Real-world data only |
| M2 Features & baselines | Planned | |
| M3 TM training | Planned | |
| M4 Tuning & compression | Planned | |
| M5 Explainability | Planned | |
| M6 Deployment & benchmarks | Planned | |
| M7 Showcase & release | Planned | |

## Next steps (in order)

- [ ] Verify GitHub repo settings: description, topics, replace `OWNER` in README badges, CI green ([docs/github-setup.md](docs/github-setup.md))
- [x] Check TMU installs: `uv sync --extra tm` (works natively on Windows, no WSL/Docker needed)
- [x] Re-run `uv run ruff check .` and `uv run pytest` after the `features.py` lint fix
- [x] M1: define snippet record schema in code (`Snippet` in `src/codelangtm/data.py`)
- [x] M1: dataset loader + tests on tiny fixtures (JSONL in `data.py`)
- [x] M1: window extractor (20-50 contiguous lines), dedup, label sanity check
- [x] M1: group-by-repo split + k-fold
- [x] M1: GitHub collector (permissive licenses only; token from env var)
- [x] M1: smoke-test collector against real GitHub (4 Python snippets, 2 Apache-2.0 repos, 0 drops)
- [x] M1: `codelangtm data build` command: merge sources → split → write train/test/wild
- [ ] M1: Stage A collection run (`uv run codelangtm collect github`) + review drop counts per language
- [ ] M1: `uv run codelangtm data build` on Stage A data
- [ ] M1: public-dataset loaders (The Stack, CodeSearchNet), wild set
- [ ] M1: Stage A dataset (1,000 snippets) + dataset card
- [ ] Log every data source in [docs/data-sources.md](docs/data-sources.md)

## Blockers / open questions
- None.

## Done log

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
No experiments yet. Record each run here: date, config, dataset version, macro-F1 (CV / test / wild), latency, model size.

| Date | Experiment | Dataset | Macro-F1 | Notes |
| --- | --- | --- | --- | --- |
| | | | | |

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
