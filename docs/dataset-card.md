# Dataset card: CodeLangTM Stage A (v5)

Real-world source code windows labelled with their programming language, built to train and evaluate an interpretable Tsetlin Machine language identifier. Structure follows *Datasheets for Datasets* (Gebru et al.). How problems found along the way were fixed: [issues-and-fixes.md](issues-and-fixes.md).

## At a glance

| | |
| --- | --- |
| Version | Stage A **v5** (v4 snippets, stable split), collected 2026-09-24 |
| Instances | **869** code windows (20-50 lines each) |
| Classes | 8: Python, C++, Java, JavaScript, Rust, Go, SQL, HTML |
| Source | 196 public GitHub repositories, MIT / Apache-2.0 / BSD licensed |
| Splits | train 696 / test 173, stable hash-based repo split (5 test repos per language, SQL 4), 5 CV folds on train, wild set empty |
| Files | `data/processed/{train,test,wild}.jsonl`, `folds.json`, `dataset.json` (not committed) |

## Motivation
- **Purpose:** train and fairly evaluate a language classifier whose decisions are human-readable rules. Evaluation must reflect unseen codebases, so the dataset is built to prevent repository-level leakage.
- **Why not synthetic data:** generated snippets lack the noise of real code (comments, tests, style variation), so scores would not transfer.
- **Created by:** kolonel-git, as part of the CodeLangTM portfolio project.

## Composition

| Language | Train | Test | Total | Repos | Largest repo share | Median chars | Median comment ratio | Test-file share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Python | 93 | 21 | 114 | 25 | 4% | 1,062 | 7% | 23% |
| C++ | 90 | 21 | 111 | 25 | 5% | 1,012 | 8% | 6% |
| Java | 94 | 20 | 114 | 25 | 4% | 998 | 8% | 22% |
| JavaScript | 85 | 21 | 106 | 25 | 5% | 1,021 | 3% | 29% |
| Rust | 98 | 25 | 123 | 25 | 4% | 1,015 | 10% | 13% |
| Go | 99 | 25 | 124 | 25 | 4% | 811 | 5% | 40% |
| SQL | 61 | 16 | 77 | 21 | 6% | 1,022 | 8% | 1% |
| HTML | 76 | 24 | 100 | 25 | 5% | 1,319 | 0% | 7% |
| **Total** | **696** | **173** | **869** | **196** | | | | |

- **Instance:** one contiguous window of 20-50 lines from one file (median 30-34 lines), with provenance. Schema (`Snippet` in `src/codelangtm/data.py`):
  `text, language, source, repo, commit, path, license, start_line, end_line`.
- **Label:** the file's language from its extension (`.h` resolved as C or C++ from content), cross-checked against content (see Cleaning).
- **One window per file**, at most 5 per repo, so no repository or file dominates a language.
- **Non-English content:** negligible (mean non-ASCII share <= 0.91%, highest for HTML).
- **Personal data:** none targeted. Snippets may contain author names or emails in comments, as published in the source repositories.

## Collection process
Command: `codelangtm collect github` (`src/codelangtm/github.py`); per-language parameters and repo lists in `data/raw/github/<language>.manifest.json`.

1. **Repository search:** GitHub search by language, excluding forks and archived repos, in four star bands (50-199, 200-999, 1000-4999, 5000+), taking repos round-robin across bands. Target 25 repos per language.
2. **License allowlist:** repository license must be MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause; anything else is never fetched.
3. **Pinning:** each repo's HEAD commit SHA is recorded; files are downloaded at that commit, so every snippet can be traced to exact lines.
4. **File selection:** matching extension; 400 B-200 KB; excluding `vendor/`, `node_modules/`, `third_party/`, `build/`, `dist/` and similar, plus minified and generated files (`.min.js`, `_pb2.py`, `.pb.go`, `.d.ts`). Up to 15 files sampled per repo (seeded).
5. **Windowing:** one random 20-50 line window per file (seeded); boundaries never fall inside a block comment or multi-line string; blank, license-header and comment-only windows skipped.

**Deviation:** SQL was collected with `--min-stars 10` (all other languages: 50), because too few permissively licensed SQL repositories exist above 50 stars. SQL therefore has 21 repos, mostly in the lowest star band (14 of 21 at 10-199 stars).

**Stars:** median ~1,000, range 11 to 482,661. Bands are near-uniform for all languages except SQL.

## Cleaning and filtering
Every snippet passes these checks at collection and again at build time (`labels.py`, `syntax.py`, `dedup.py`):

| Check | Drops (collection, v4) |
| --- | --- |
| No usable window (file too short, or only low-signal windows) | 132 |
| Template-heavy SQL/HTML (> 30% Jinja/Liquid/ERB lines) | 45 (SQL 13, HTML 32) |
| Expected markers missing (HTML without a real tag, SQL without keywords) | 26 (HTML 22, SQL 4) |
| Mostly embedded script/style (HTML with < 20% markup lines) | 18 |
| Not UTF-8 | 18 (SQL 16) |
| Minified or data (line > 500 chars) | 16 |
| Mostly non-ASCII | 2 |
| Extension/content mismatch, other-language content, cut comment/string | 0 |
| Near-duplicate or exact duplicate (MinHash, Jaccard >= 0.8) | 2 |

Repos yielding no usable snippets were skipped (HTML 13, SQL 5, Go 1, Python 1) and replaced by the next candidate.

## Splits
Command: `codelangtm data build` (`src/codelangtm/build.py`, `splits.py`).

- **Train/test (stable hash split, `splits.stable_split`, salt `codelangtm-v1`):** within each language, repos are ordered by a SHA-256 hash of their name and the first round(n × 0.2) go to test: 5 repos per language (SQL 4 of 21), 173 snippets (19.9%), 39 repos. **No repository appears in both.** Verified by `check_no_leakage` on every build.
- **CV:** the remaining repos of each language are ordered by a second hash and cut into 5 equal consecutive folds (repo counts differ by at most 1); fold id per snippet in `folds.json`; a repo never spans folds.
- **Stability:** re-collecting one language never moves another language's repos; adding or removing one repo moves at most one existing repo in or out of test. Test scores of unchanged languages stay comparable across dataset versions.
- **Test-score variance:** with 39 test repos, the test score depends on which repos land in test (v4 experiment: 0.870-0.978 across 10 random splits). Results therefore report CV macro-F1 (model selection) and **repeated test macro-F1**: mean ± std over 10 further balanced repo-level splits of train + test (salts `codelangtm-v1:repeat:0..9`).
- **Wild set:** empty. Planned: hand-collected StackOverflow / blog / documentation snippets, used only for final evaluation; build drops any wild snippet that duplicates training data.

## Licensing and redistribution
- Snippets carry their repository license: MIT 509, Apache-2.0 319, BSD-3-Clause 26, BSD-2-Clause 15.
- Licensing is checked at repository level; individual files may carry different notices.
- The dataset is **not redistributed** in this repository (`data/` is gitignored). It is reproducible from the commands below; every record keeps repo, commit and path for attribution.

## Known biases and limitations
- **Popularity bias:** repos come from GitHub search ordering within star bands; very obscure code styles are under-represented.
- **Class imbalance:** SQL (77) is smaller than Go (124). Report macro-F1, not only accuracy.
- **SQL quality and dialects:** lower-star repos, dialect mix unmeasured (Postgres, MySQL, T-SQL, PL/SQL all present), some lightly templated dbt code (< 30% template lines).
- **SQL learned partly as "data rows":** many SQL windows are `INSERT ... VALUES` rows. With the default frequency-ranked features, baselines assign repetitive list-like code in other languages (long runs of `call("x", 6),` lines) to SQL. Labels are correct; the cause was feature selection: label-aware selection (M2 ablations) keeps `SELECT` fragments and raises SQL CV F1 from 0.87 to 0.99 ([issues-and-fixes](issues-and-fixes.md) M5, M6).
- **Test code share varies:** Go 40%, JavaScript 29%, Python 23%, Java 22%, others <= 13%. Checked in M2: test idioms (`t.Run`, `assert`, `@Test`) are not among the top baseline features, so this is not acting as a shortcut.
- **Embedded languages:** HTML windows that are mostly inline `<script>`/`<style>` are excluded (< 20% markup lines); HTML with some script (>= 20% markup) is kept. JavaScript windows dominated by HTML template strings and Python/Java with embedded SQL strings are not filtered.
- **Window length:** 20-50 lines only. Accuracy on one-line or very short snippets is not measured by this dataset.
- **Snapshot:** single collection date; languages evolve (e.g. newer syntax) after it.
- **Label noise:** labels come from file extensions with heuristic content checks, not human annotation. Manual review of 80 random samples found no mislabels. Confident learning on the training split flagged 6 of 689 in v3 (3 HTML windows of inline script, fixed in v4) and 5 of 696 in v4, all with correct labels (hard or list-like examples).

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

Exact reproduction also needs the same repository HEADs; GitHub search results and HEAD commits change over time, so a re-run produces a similar, not identical, dataset. v5 is identified by these SHA-256 prefixes (full hashes in `dataset.json`):

| File | SHA-256 (prefix) |
| --- | --- |
| `train.jsonl` | `8cdc4c91ebe6` |
| `test.jsonl` | `54649be8e4e0` |
| `folds.json` | `77229e0f7668` |
| `wild.jsonl` | `e3b0c44298fc` (empty) |

**History:**

| Version | Change | Snippets |
| --- | --- | --- |
| v1 | First collection | 829 (SQL 40) |
| v2 | SQL top-up (`--min-stars 10`), template filter | 845 |
| v3 | Cut-safe windows, full re-collection | 861 |
| v4 | HTML embedded-language rule, HTML re-collected | 869 (HTML 100) |
| v5 | Same snippets as v4; stable hash-based split replaces StratifiedGroupKFold | 869 |

Details in [issues-and-fixes.md](issues-and-fixes.md).

**Maintenance:** Stage B will add The Stack / CodeSearchNet sources, stretch languages (C, C#, TypeScript, Kotlin, PHP, Ruby) and the wild set, as a new version with its own card entry.
