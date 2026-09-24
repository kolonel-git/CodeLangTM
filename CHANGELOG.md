# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
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

### Changed
- `data build` uses a stable hash-based per-language repo split (`stable_split`, `--salt`) instead of StratifiedGroupKFold; re-collecting one language no longer reshuffles other languages' test repos.
- `codelangtm baselines` reports repeated test macro-F1 over 10 extra balanced splits (`--repeats`).
- `Binarizer` is a scikit-learn transformer (refit per CV fold), with alphabetical tie-breaking, JSON save/load, a `use_delimiters` switch and set-based transform. When `n_features` is smaller than the delimiter list, delimiters are now truncated too.
- Dependencies: removed unused `pandas`; declared `scipy` (used directly by `features.py`).

### Fixed
- HTML label check accepted comparisons (`i < x`) as tags; HTML windows that are mostly inline script/style are now dropped (embedded-language rule).
- TMU crash on NumPy 2: `tm` extra pins `numpy<2`, `scipy<1.14`.
- Windows splitting block comments, docstrings and multi-line strings (4.6% of Stage A v2 snippets).
