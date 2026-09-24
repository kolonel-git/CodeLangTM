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
- Dataset card for Stage A v3 (`docs/dataset-card.md`) and source ledger entries.
- Classical baselines with repo-grouped CV and generated `docs/results.md` (`baselines.py`, `codelangtm baselines`).
- Data diagnostics: confident-learning label-issue candidates and shortcut probe (`diagnostics.py`, `codelangtm diagnose`).

### Changed
- `Binarizer` is a scikit-learn transformer (refit per CV fold), with alphabetical tie-breaking, JSON save/load, a `use_delimiters` switch and set-based transform. When `n_features` is smaller than the delimiter list, delimiters are now truncated too.

### Fixed
- TMU crash on NumPy 2: `tm` extra pins `numpy<2`, `scipy<1.14`.
- Windows splitting block comments, docstrings and multi-line strings (4.6% of Stage A v2 snippets).
