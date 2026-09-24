# Dataset card: CodeLangTM Stage A (v3)

Real-world source code windows labelled with their programming language, built to train and evaluate an interpretable Tsetlin Machine language identifier. Structure follows *Datasheets for Datasets* (Gebru et al.). How problems found along the way were fixed: [issues-and-fixes.md](issues-and-fixes.md).

## At a glance

| | |
| --- | --- |
| Version | Stage A **v3**, collected 2026-09-24 |
| Instances | **861** code windows (20-50 lines each) |
| Classes | 8: Python, C++, Java, JavaScript, Rust, Go, SQL, HTML |
| Source | 196 public GitHub repositories, MIT / Apache-2.0 / BSD licensed |
| Splits | train 689 / test 172 (by repository), 5 CV folds on train, wild set empty |
| Files | `data/processed/{train,test,wild}.jsonl`, `folds.json`, `dataset.json` (not committed) |

## Motivation
- **Purpose:** train and fairly evaluate a language classifier whose decisions are human-readable rules. Evaluation must reflect unseen codebases, so the dataset is built to prevent repository-level leakage.
- **Why not synthetic data:** generated snippets lack the noise of real code (comments, tests, style variation), so scores would not transfer.
- **Created by:** kolonel-git, as part of the CodeLangTM portfolio project.

## Composition

| Language | Train | Test | Total | Repos | Largest repo share | Median chars | Median comment ratio | Test-file share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Python | 90 | 24 | 114 | 25 | 4% | 1,062 | 7% | 23% |
| C++ | 90 | 21 | 111 | 25 | 5% | 1,012 | 8% | 6% |
| Java | 93 | 21 | 114 | 25 | 4% | 998 | 8% | 22% |
| JavaScript | 85 | 21 | 106 | 25 | 5% | 1,021 | 3% | 29% |
| Rust | 98 | 25 | 123 | 25 | 4% | 1,015 | 10% | 13% |
| Go | 99 | 25 | 124 | 25 | 4% | 811 | 5% | 40% |
| SQL | 61 | 16 | 77 | 21 | 6% | 1,022 | 8% | 1% |
| HTML | 73 | 19 | 92 | 25 | 5% | 1,103 | 0% | 9% |
| **Total** | **689** | **172** | **861** | **196** | | | | |

- **Instance:** one contiguous window of 20-50 lines from one file (median 30-34 lines), with provenance. Schema (`Snippet` in `src/codelangtm/data.py`):
  `text, language, source, repo, commit, path, license, start_line, end_line`.
- **Label:** the file's language from its extension (`.h` resolved as C or C++ from content), cross-checked against content (see Cleaning).
- **One window per file**, at most 5 per repo, so no repository or file dominates a language.
- **Non-English content:** negligible (mean non-ASCII share <= 0.84%, highest for HTML).
- **Personal data:** none targeted. Snippets may contain author names or emails in comments, as published in the source repositories.

## Collection process
Command: `codelangtm collect github` (`src/codelangtm/github.py`); per-language parameters and repo lists in `data/raw/github/<language>.manifest.json`.

1. **Repository search:** GitHub search by language, excluding forks and archived repos, in four star bands (50-199, 200-999, 1000-4999, 5000+), taking repos round-robin across bands. Target 25 repos per language.
2. **License allowlist:** repository license must be MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause; anything else is never fetched.
3. **Pinning:** each repo's HEAD commit SHA is recorded; files are downloaded at that commit, so every snippet can be traced to exact lines.
4. **File selection:** matching extension; 400 B-200 KB; excluding `vendor/`, `node_modules/`, `third_party/`, `build/`, `dist/` and similar, plus minified and generated files (`.min.js`, `_pb2.py`, `.pb.go`, `.d.ts`). Up to 15 files sampled per repo (seeded).
5. **Windowing:** one random 20-50 line window per file (seeded); boundaries never fall inside a block comment or multi-line string; blank, license-header and comment-only windows skipped.

**Deviation:** SQL was collected with `--min-stars 10` (all other languages: 50), because too few permissively licensed SQL repositories exist above 50 stars. SQL therefore has 21 repos, mostly in the lowest star band (14 of 21 at 10-199 stars).

**Stars:** median 999, range 11 to 482,661. Bands are near-uniform for all languages except SQL.

## Cleaning and filtering
Every snippet passes these checks at collection and again at build time (`labels.py`, `syntax.py`, `dedup.py`):

| Check | Drops (collection, v3) |
| --- | --- |
| No usable window (file too short, or only low-signal windows) | 130 |
| Template-heavy SQL/HTML (> 30% Jinja/Liquid/ERB lines) | 38 (SQL 13, HTML 25) |
| Not UTF-8 | 18 (SQL 16) |
| Minified or data (line > 500 chars) | 16 |
| Expected markers missing (HTML without tags, SQL without keywords) | 17 |
| Mostly non-ASCII | 2 |
| Extension/content mismatch, other-language content, cut comment/string | 0 |
| Near-duplicate or exact duplicate (MinHash, Jaccard >= 0.8) | 2 |

Repos yielding no usable snippets were skipped (HTML 9, SQL 5, Go 1, Python 1) and replaced by the next candidate.

## Splits
Command: `codelangtm data build` (`src/codelangtm/build.py`, `splits.py`).

- **Train/test:** `StratifiedGroupKFold` grouped by repository, seed 0; test = one fold (172 snippets, 20.0%, 39 repos). **No repository appears in both.** Verified by `check_no_leakage` on every build.
- **CV:** 5 stratified group folds over train; fold id per snippet in `folds.json`; a repo never spans folds.
- **Wild set:** empty in v3. Planned: hand-collected StackOverflow / blog / documentation snippets, used only for final evaluation; build drops any wild snippet that duplicates training data.

## Licensing and redistribution
- Snippets carry their repository license: MIT 505, Apache-2.0 315, BSD-3-Clause 26, BSD-2-Clause 15.
- Licensing is checked at repository level; individual files may carry different notices.
- The dataset is **not redistributed** in this repository (`data/` is gitignored). It is reproducible from the commands below; every record keeps repo, commit and path for attribution.

## Known biases and limitations
- **Popularity bias:** repos come from GitHub search ordering within star bands; very obscure code styles are under-represented.
- **Class imbalance:** SQL (77) and HTML (92) are smaller than Go (124). Report macro-F1, not only accuracy.
- **SQL quality and dialects:** lower-star repos, dialect mix unmeasured (Postgres, MySQL, T-SQL, PL/SQL all present), some lightly templated dbt code (< 30% template lines).
- **Test code share varies:** Go 40%, JavaScript 29%, Python 23%, Java 22%, others <= 13%. Checked in M2: test idioms (`t.Run`, `assert`, `@Test`) are not among the top baseline features, so this is not acting as a shortcut.
- **Window length:** 20-50 lines only. Accuracy on one-line or very short snippets is not measured by this dataset.
- **Embedded languages:** HTML windows may contain inline JS/CSS; Python/Java may contain SQL strings. No explicit policy yet (Stage B).
- **Snapshot:** single collection date; languages evolve (e.g. newer syntax) after it.
- **Label noise:** labels come from file extensions with heuristic content checks, not human annotation. Manual review of 80 random samples (10 per language) after v3 found no mislabelled snippets. Confident learning on the training split (M2) flagged 6 of 689: 3 HTML windows whose content is inline JavaScript (see issues-and-fixes D5), 1 JavaScript window dominated by an HTML template string, and 2 correct but hard examples.

## Intended use
- **In scope:** training and evaluating language identifiers on multi-line snippets of the 8 languages; comparing interpretable and classical models under leak-free evaluation.
- **Out of scope:** code quality or security judgements, authorship attribution, single-line language detection, languages outside the 8 classes.

## Reproduction and versioning

```bash
uv sync --extra tm --extra collect
uv run codelangtm collect github --language python --language cpp --language java --language javascript --language rust --language go --language html
uv run codelangtm collect github --language sql --min-stars 10
uv run codelangtm data build
uv run codelangtm data audit
```

Exact reproduction also needs the same repository HEADs; GitHub search results and HEAD commits change over time, so a re-run produces a similar, not identical, dataset. v3 is identified by these SHA-256 prefixes (full hashes in `dataset.json`):

| File | SHA-256 (prefix) |
| --- | --- |
| `train.jsonl` | `a8402ed86ad3` |
| `test.jsonl` | `7a5bf586beec` |
| `folds.json` | `aa4673b4765d` |
| `wild.jsonl` | `e3b0c44298fc` (empty) |

**History:** v1 829 snippets (SQL 40) → v2 845 (SQL top-up, template filter) → v3 861 (cut-safe windows, re-collected). Details in [issues-and-fixes.md](issues-and-fixes.md).

**Maintenance:** Stage B will add The Stack / CodeSearchNet sources, stretch languages (C, C#, TypeScript, Kotlin, PHP, Ruby) and the wild set, as a new version with its own card entry.
