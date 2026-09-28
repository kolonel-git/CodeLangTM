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
- `uv run codelangtm tm-curve [--setting NAME]` — TM training curves on CV folds from configs/tm.yaml → docs/tm-curves.md + tm-curves.json (recommends `epochs`)
- `uv run codelangtm tm-results [--setting NAME]` — TM vs baselines: full protocol per seed + corrected t-tests → docs/tm-results.md + .json, final models → models/ (gitignored; ~25 min)
- `uv run codelangtm tm-train [--setting] [--seed]` — one TM on all of train → models/<setting>_seed<k>.json (~8 s); `uv run codelangtm tm-select` — official model models/tm.json = median-CV seed of the tm-results run (tm_400 seed 4)
- `uv run codelangtm resources` — process-level fit/predict resources (fresh process per job, 3 repeats) for the baselines + each TM's median-CV model → docs/resources.md + .json (~1-2 min)
- `uv run codelangtm clauses` — clause inspector on models/tm.json: every clause as a rule + train statistics, signatures, overlap, cross-seed stability (other models/tm_400_seed*.json), formation replay (~1 min, needs TMU; `--no-formation` skips) → docs/clauses.md + clauses.json
- `uv run codelangtm errors` — error analysis on out-of-fold CV predictions (TM seeds + baselines, train only; ~4 min) → docs/errors.md + errors.json; snippet code only in data/processed/errors-review.md (never commit)
- `uv run codelangtm explain --file FILE|-` — one prediction traced to the clauses that fired (NumPy only)
- `uv run codelangtm report` — figures from the JSON sidecars → docs/figures/*.png (narrative in docs/report.md is hand-written)

## Layout
`src/codelangtm/` (full table in docs/architecture.md):
- Data: `data.py` (schema), `github.py` + `windows.py` + `syntax.py` (collection), `labels.py` + `dedup.py` (cleaning), `build.py` + `splits.py` (stable repo split), `audit.py`
- Features and evaluation: `features.py` (binarizer + literals), `config.py` (YAML configs), `baselines.py`, `ablations.py`, `diagnostics.py`, `curves.py` (TM training curves), `tm_results.py` (TM vs baselines, significance), `resources.py` (process-level resources), `errors.py` (out-of-fold error analysis), `figures.py` (report PNGs from the JSON sidecars; needs `--extra viz`)
- TM: `model.py` (`TMLanguageClassifier` estimator over TMU, TMU-free `TMState`, JSON model files `codelangtm.tm/1`), `rules.py` (clause inspector: rules, statistics, explanations, formation replay); stub until M6: `export_c.py` (C export)
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
