# CodeLangTM

[![CI](https://github.com/kolonel-git/CodeLangTM/actions/workflows/ci.yml/badge.svg)](https://github.com/kolonel-git/CodeLangTM/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Interpretable source code language identification via Tsetlin Machines.**

CodeLangTM learns transparent Boolean rules instead of opaque floating-point weights:

```
python = has("def ") AND has(":") AND NOT has(";")
```

- **Fast, cheap inference** — bitwise AND/OR/NOT instead of matrix multiplication.
- **Explainable** — every prediction traces to human-readable clauses.
- **Tiny** — no weight matrices; target < 500 KB.

Planned deliverables: a `codelangtm predict` CLI, a zero-dependency C runtime and a web demo that shows the clauses behind each prediction. The project is a portfolio piece, written so it can later become a research write-up (multi-seed results, significance tests, saved raw runs).

## Pipeline

```
snippet -> binarizer (M label-selected character 2/3-grams) -> literals (x, NOT x) -> Tsetlin Machine (TMU) -> argmax vote -> language
```

See [docs/architecture.md](docs/architecture.md). Data is real-world code only, from permissively licensed GitHub repositories ([dataset card](docs/dataset-card.md)).

## Targets

| Metric | Target |
| --- | --- |
| Macro F1 (8 languages) | >= 96% |
| Latency / snippet | < 0.1 ms |
| Model size | < 500 KB |

Languages: Python, C++, Java, JavaScript, Rust, Go, SQL, HTML.

## Quickstart

```bash
uv sync --extra tm --extra collect --extra viz   # Python 3.12 env, dev tools, TMU, collector, figures (plain `uv sync` drops extras)
uv run codelangtm --version
uv run pytest
```

Reproduce the pipeline (needs a read-only `GITHUB_TOKEN`, see [docs/data-sources.md](docs/data-sources.md)):

```bash
uv run codelangtm collect github                                  # data/raw/
uv run codelangtm data build                                      # data/processed/
uv run codelangtm baselines --config configs/baselines.yaml       # docs/results.md + .json
uv run codelangtm ablate                                          # docs/ablations.md + .json
uv run codelangtm report                                          # docs/figures/*.png
```

## Status

Pre-alpha. **Read the [report](docs/report.md)** for the results so far, with figures; new to Tsetlin Machines or the evaluation terms? Start with the [concepts guide](docs/concepts.md). Full plan in [docs/roadmap.md](docs/roadmap.md).

| Milestone | Status |
| --- | --- |
| M0 Foundations | done |
| M1 Data pipeline | Stage A done: 869 snippets, 8 languages, 196 repos ([dataset card](docs/dataset-card.md)) |
| M2 Features & baselines | done: best baseline CV macro-F1 0.963, repeated test 0.968 ([results](docs/results.md)), thanks to label-aware feature selection ([ablations](docs/ablations.md)) |
| M3 TM training | in progress: TM classifier and training curves done; on CV the TM reaches 0.95 vs Naive Bayes 0.963 ([report](docs/report.md), section 7); full comparison next |
| M4 Tuning & compression | planned |
| M5 Explainability | planned |
| M6 Deployment & benchmarks | planned |
| M7 Showcase & release | planned |

## License

[MIT](LICENSE)
