# CodeLangTM

Tsetlin Machine (TMU) classifier that identifies a snippet's programming language with human-readable AND-rules.

## Commands
- `uv sync --extra tm --extra collect --extra viz` — install (plain `uv sync` removes TMU, httpx and matplotlib)
- `uv run pytest` — tests
- `uv run ruff check .` — lint
- `uv run codelangtm --version` — CLI
- `uv run codelangtm collect github` / `data build` / `data audit` — dataset pipeline (see docs/data-sources.md)
- `uv run codelangtm baselines --config configs/baselines.yaml` — classical baselines → docs/results.md + results.json (flags override the YAML)
- `uv run codelangtm diagnose` — confident-learning label issues + shortcut probe → data/processed/diagnostics.md
- `uv run codelangtm ablate [--study NAME]` — feature ablations from configs/ablations.yaml → docs/ablations.md + ablations.json (CV only)
- `uv run codelangtm report` — figures from the JSON sidecars → docs/figures/*.png (narrative in docs/report.md is hand-written)

## Layout
`src/codelangtm/` (full table in docs/architecture.md):
- Data: `data.py` (schema), `github.py` + `windows.py` + `syntax.py` (collection), `labels.py` + `dedup.py` (cleaning), `build.py` + `splits.py` (stable repo split), `audit.py`
- Features and evaluation: `features.py` (binarizer + literals), `config.py` (YAML configs), `baselines.py`, `ablations.py`, `diagnostics.py`, `figures.py` (report PNGs from the JSON sidecars; needs `--extra viz`)
- TM: `model.py` (`TMLanguageClassifier` estimator over TMU, TMU-free `TMState`, JSON model files `codelangtm.tm/1`); stubs until B6/M6: `rules.py` (rule extraction), `export_c.py` (C export)
- TMU seeds must be >= 1 (seed 0 hangs TMU); tests with TMU use `pytest.importorskip("tmu")`
- `cli.py`; experiment configs in `configs/`; generated reports `docs/results.md`, `docs/ablations.md`, their `.json` sidecars and `docs/figures/` (never edit by hand); `docs/report.md` is the hand-written narrative (update its numbers when results change)

## Conventions
- Conventional Commits. Ruff line length 100.
- Real-world data only, collected by the maintainer into `data/raw/` (GitHub: `data/raw/github/<language>.jsonl`, gitignored). No synthetic generators.
- TMU (`tm`) and matplotlib (`viz`) are optional extras; core code must import without them.
- `tests/test_docs.py` checks every relative link/image in the markdown; keep docs links valid.
- Reporting (from M3): TM numbers are means over 5 seeds with spread; TM vs baselines uses a paired significance test on the same folds; save raw per-seed/per-fold runs as JSON; choose settings on CV only, never on test.
- `docs/concepts.md` explains ideas in plain language; add an entry when a step introduces a new concept.
- Project direction (portfolio now, research write-up later; CLI + C runtime + web demo): see the Principles in docs/roadmap.md.
