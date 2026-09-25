# Contributing

## Setup
```bash
uv sync --extra tm --extra collect   # Python 3.12 venv, dev tools, TMU, collector
```
A plain `uv sync` removes the `tm` and `collect` extras again.

## Workflow
- Branch from `main`: `feat/<topic>`, `fix/<topic>`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Before PR: `uv run ruff check . && uv run pytest`.
- One roadmap item per PR; link the issue (`Closes #N`). Issues belong to a milestone (M1-M7, see [docs/roadmap.md](docs/roadmap.md)).
- Feature branches only; no direct commits to `main`. See [docs/github-setup.md](docs/github-setup.md).
- Experiments: define them in `configs/*.yaml` (fixed seeds) and commit the generated reports (`docs/results.md`, `docs/ablations.md`); never edit generated reports by hand.
- Log problems and their fixes in `docs/issues-and-fixes.md`, and session progress in `PROGRESS.md`.

## Data
Datasets live in `data/` and are gitignored. Do not commit code with unclear licensing; record each source and its license in `docs/data-sources.md`.
