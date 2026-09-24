# CodeLangTM

Tsetlin Machine (TMU) classifier that identifies a snippet's programming language with human-readable AND-rules.

## Commands
- `uv sync` / `uv sync --extra tm` — install
- `uv run pytest` — tests
- `uv run ruff check .` — lint
- `uv run codelangtm --version` — CLI

## Layout
`src/codelangtm/`: `features.py` (binarizer + literals), `model.py` (TMU wrapper), `rules.py` (rule extraction), `export_c.py` (C export), `cli.py`.

## Conventions
- Conventional Commits. Ruff line length 100.
- Real-world data only, collected by the maintainer into `data/raw/<language>/`. No synthetic generators.
- TMU is an optional extra; core code must import without it.
