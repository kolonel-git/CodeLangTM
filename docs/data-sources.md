# Data sources

Datasets are gitignored (`data/`). This file is the provenance ledger: every source used must be listed here with its license.

## Policy
- Only permissively licensed code (MIT, Apache-2.0, BSD-2/3) unless a source states otherwise.
- Keep attribution (repo URL + commit) for every snippet in its record.
- Wild test set sources (StackOverflow, blogs, docs) must have license/attribution terms noted below.

## Collecting from GitHub
1. Create a fine-grained personal access token: public repositories, read-only, no extra permissions, 90-day expiry.
2. Store it in the `GITHUB_TOKEN` user environment variable (never in a file in this repo), then restart the terminal.
3. Install and run:
   ```bash
   uv sync --extra tm --extra collect
   uv run codelangtm collect github --language python --repos 2 --per-repo 2   # smoke test
   uv run codelangtm collect github                                            # Stage A: 8 languages
   ```
Outputs `data/raw/github/<language>.jsonl` (snippets) and `<language>.manifest.json` (repos, commit SHAs, licenses, drop counts). Downloads are cached in `data/cache/github/`, so interrupted runs resume. Add one summary row per run to the ledger below.

Selection rules: MIT / Apache-2.0 / BSD licenses only; no forks or archived repos; repos spread across star bands (50-199, 200-999, 1000-4999, 5000+); vendored, generated, minified, tiny (< 400 B) and huge (> 200 KB) files skipped; one random 20-50 line window per file; every snippet passes the label check and dedup.

## Snippet record schema
| Field | Description |
| --- | --- |
| `text` | Snippet content (20-50 contiguous lines) |
| `language` | Label |
| `source` | `github`, `the-stack`, `codesearchnet`, `wild` |
| `repo` | `owner/name` (grouping key for splits) |
| `commit` | Commit SHA |
| `path` | File path in repo |
| `license` | SPDX id |
| `start_line`, `end_line` | Window position in file |

## Source ledger
| Source | URL | License | Languages | Snippets | Date collected | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| _(add rows as data is collected)_ | | | | | | |
