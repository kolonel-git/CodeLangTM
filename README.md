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

## Pipeline

```
snippet -> n-gram/delimiter binarizer (top-M features) -> literals (x, NOT x) -> Tsetlin Machine (TMU) -> argmax vote -> language
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
uv sync --extra tm --extra collect   # Python 3.12 env, dev tools, TMU, collector (plain `uv sync` drops extras)
uv run codelangtm --version
uv run pytest
```

Reproduce the pipeline (needs a read-only `GITHUB_TOKEN`, see [docs/data-sources.md](docs/data-sources.md)):

```bash
uv run codelangtm collect github                                  # data/raw/
uv run codelangtm data build                                      # data/processed/
uv run codelangtm baselines --config configs/baselines.yaml       # docs/results.md
uv run codelangtm ablate                                          # docs/ablations.md
```

## Status

Pre-alpha. Full plan in [docs/roadmap.md](docs/roadmap.md).

| Milestone | Status |
| --- | --- |
| M0 Foundations | done |
| M1 Data pipeline | Stage A done: 869 snippets, 8 languages, 196 repos ([dataset card](docs/dataset-card.md)) |
| M2 Features & baselines | done (PR pending): best baseline CV macro-F1 0.963, repeated test 0.968 ([results](docs/results.md)), thanks to label-aware feature selection ([ablations](docs/ablations.md)) |
| M3 TM training | planned |
| M4 Tuning & compression | planned |
| M5 Explainability | planned |
| M6 Deployment & benchmarks | planned |
| M7 Showcase & release | planned |

## License

[MIT](LICENSE)
