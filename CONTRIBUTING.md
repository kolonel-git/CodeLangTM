# Contributing

## Setup
```bash
uv sync            # Python 3.12 venv + dev tools
uv sync --extra tm # add TMU (needs a C compiler)
```

## Workflow
- Branch from `main`: `feat/<topic>`, `fix/<topic>`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Before PR: `uv run ruff check . && uv run pytest`.
- One roadmap item per PR; link the issue (`Closes #N`). Issues belong to a milestone (M1-M7, see [docs/roadmap.md](docs/roadmap.md)).
- Feature branches only; no direct commits to `main`. See [docs/github-setup.md](docs/github-setup.md).
- Experiments: use YAML configs and fixed seeds; commit resulting tables to `docs/results.md`.

## Data
Datasets live in `data/` and are gitignored. Do not commit code with unclear licensing; record each source and its license in `docs/data-sources.md`.
