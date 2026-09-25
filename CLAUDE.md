# CodeLangTM

Tsetlin Machine (TMU) classifier that identifies a snippet's programming language with human-readable AND-rules.

## Commands
- `uv sync` / `uv sync --extra tm` — install
- `uv run pytest` — tests
- `uv run ruff check .` — lint
- `uv run codelangtm --version` — CLI
- `uv run codelangtm collect github` / `data build` / `data audit` — dataset pipeline (see docs/data-sources.md)
- `uv run codelangtm baselines --config configs/baselines.yaml` — classical baselines → docs/results.md (flags override the YAML)
- `uv run codelangtm diagnose` — confident-learning label issues + shortcut probe → data/processed/diagnostics.md
- `uv run codelangtm ablate [--study NAME]` — feature ablations from configs/ablations.yaml → docs/ablations.md (CV only)
- Always `uv sync --extra tm --extra collect` (plain `uv sync` removes TMU)

## Layout
`src/codelangtm/` (full table in docs/architecture.md):
- Data: `data.py` (schema), `github.py` + `windows.py` + `syntax.py` (collection), `labels.py` + `dedup.py` (cleaning), `build.py` + `splits.py` (stable repo split), `audit.py`
- Features and evaluation: `features.py` (binarizer + literals), `config.py` (YAML configs), `baselines.py`, `ablations.py`, `diagnostics.py`
- TM (stubs until M3/M5/M6): `model.py` (TMU wrapper), `rules.py` (rule extraction), `export_c.py` (C export)
- `cli.py`; experiment configs in `configs/`; generated reports `docs/results.md`, `docs/ablations.md` (never edit by hand)

## Conventions
- Conventional Commits. Ruff line length 100.
- Real-world data only, collected by the maintainer into `data/raw/<language>/`. No synthetic generators.
- TMU is an optional extra; core code must import without it.
