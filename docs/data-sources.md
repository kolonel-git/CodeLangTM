# Data sources

Datasets are gitignored (`data/`). This file is the provenance ledger: every source used must be listed here with its license.

## Policy
- Only permissively licensed code (MIT, Apache-2.0, BSD-2/3) unless a source states otherwise.
- Keep attribution (repo URL + commit) for every snippet in its record.
- Wild test set sources (StackOverflow, blogs, docs) must have license/attribution terms noted below.

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
