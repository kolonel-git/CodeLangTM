# Dataset card: CodeLangTM Stage B (v6)

Real-world source code windows in 14 programming languages, built for the Stage B experiments. Structure follows *Datasheets for Datasets* (Gebru et al.). Every report published so far (results, ablations, TM results, clauses, errors) uses **v5**, described in [dataset-card.md](dataset-card.md); v6 replaces it once the experiments are rerun under the [v6 analysis plan](analysis-plan-v6.md). Problems and fixes along the way: [issues-and-fixes.md](issues-and-fixes.md) (D6-D10, M12).

## At a glance

| | |
| --- | --- |
| Version | Stage B **v6**, collected 2026-09-30 |
| Instances | **8,993** code windows (20-50 lines each), plus 216 evaluation-only hard examples |
| Classes | 14: the 8 core languages (Python, C++, Java, JavaScript, Rust, Go, SQL, HTML) and 6 stretch languages (C, C#, TypeScript, Kotlin, PHP, Ruby) |
| Source | 1,960 public GitHub repositories (140 per language), MIT / Apache-2.0 / BSD licensed |
| Splits | train 7,202 / test 1,791, stable hash-based repo split (1,568 / 392 repos, 28 test repos per language), 5 CV folds on train, wild set empty |
| Files | `data/processed-v6/{train,test,wild,hard}.jsonl`, `folds.json`, `dataset.json` (not committed). SHA-256 prefixes: train `80a71a412bc8`, test `a20567d1250d`, hard `60b3f50662ef`, folds `5c7b8ed4a599` |
| Target | 10,000+ snippets: **missed by 10%** (8,993). Every language reached 140 repos, but repos gave 4.6 snippets on average instead of 5 |

## Composition

| Language | Train | Test | Total | Repos | Largest repo share | Median lines | Median comment ratio | Test-file share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Python | 528 | 126 | 654 | 140 | 1% | 34 | 7% | 24% |
| C++ | 543 | 134 | 677 | 140 | 1% | 32 | 12% | 12% |
| Java | 536 | 122 | 658 | 140 | 1% | 32 | 4% | 19% |
| JavaScript | 488 | 119 | 607 | 140 | 1% | 32 | 2% | 25% |
| Rust | 551 | 139 | 690 | 140 | 1% | 32 | 10% | 11% |
| Go | 547 | 140 | 687 | 140 | 1% | 32 | 4% | 32% |
| SQL | 441 | 111 | 552 | 140 | 1% | 31 | 7% | 11% |
| HTML | 382 | 84 | 466 | 140 | 1% | 32 | 0% | 11% |
| C | 506 | 132 | 638 | 140 | 1% | 33 | 14% | 9% |
| C# | 541 | 138 | 679 | 140 | 1% | 31 | 5% | 14% |
| TypeScript | 537 | 138 | 675 | 140 | 1% | 32 | 0% | 27% |
| Kotlin | 548 | 140 | 688 | 140 | 1% | 31 | 0% | 21% |
| PHP | 522 | 137 | 659 | 140 | 1% | 32 | 13% | 32% |
| Ruby | 532 | 131 | 663 | 140 | 1% | 32 | 3% | **44%** |
| **Total** | **7,202** | **1,791** | **8,993** | **1,960** | | | | |

- **Instance:** one contiguous 20-50 line window from one file, with provenance (`text, language, source, repo, commit, path, license, start_line, end_line`). One window per file, at most 5 per repo: 1,655 repos gave 5, 73 gave 4, 57 gave 3, 80 gave 2, 95 gave 1.
- **No repo appears under two languages.** No repo holds more than 1% of its language (v5: up to 6%).
- **Label:** the file's extension, checked against the content (see Cleaning). `.h` files are labelled C or C++ from their content (C 226, C++ 224); JavaScript is `.js` (566) and `.mjs` (41); `.jsx`/`.tsx` are not sampled.
- **SQL dialects** (by markers in the window, a window can have several): PL/pgSQL 158, T-SQL 120, MySQL 46, PL/SQL 31, none (generic) 211. SQL repos are found under four GitHub language names (`SQL`, `TSQL`, `PLpgSQL`, `PLSQL`; D8).
- **Hard examples (evaluation only, never trained on):** 216 windows (HTML 177, PHP 39) from 83 repos that are more than half embedded code (`<script>`/`<style>` in HTML, HTML markup in PHP; D7). 175 come from repos in train and 41 from repos in test; only the 41 are leak-free for evaluation. Three repos supply 34.

## Collection process
Command: `codelangtm collect github --repos 140 --per-repo 5 --pages 4` (three runs on 2026-09-30, `--out data/full/github --hard-out data/full-hard/github`); manifests with every repo, commit SHA, license and drop count in `data/full/github/<language>.manifest.json`.

1. GitHub repository search per language in four star bands (50-199, 200-999, 1000-4999, 5000+; **the same 50-star floor for every language**, SQL included), up to 4 result pages per band, forks and archived repos excluded, candidates taken round-robin across bands.
2. License allowlist (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause), checked on the repository.
3. Each repo pinned to its HEAD commit; files sampled from that commit (vendored, generated, minified, tiny and huge files skipped; `.d.ts`, `.jsx`, `.tsx`, `.g.cs`, `.designer.cs`, `.pb.*`, `_pb.rb` skipped; `obj/`, `bin/` skipped).
4. Up to 3 random windows tried per file; the first that passes every check is kept, a mostly-embedded window is set aside as a hard example.

Repos: 1,960 used, 93 skipped (no usable window; HTML 46, SQL 29). Stars: median 997, range 52 to 484,505; bands 538 / 498 / 471 / 453 (near uniform). Files fetched: 10,980.

## Cleaning and filtering

Drops during collection (windows tried, not files; all languages):

| Reason | Count |
| --- | --- |
| No usable window (file too short or only blank/license/comment windows) | 1,393 |
| Expected markers missing (HTML without a real tag, mostly script-only windows; SQL without keywords) | 299 |
| Template-heavy (more than 30% Jinja/Liquid/ERB lines; SQL, HTML) | 246 |
| `.h` file that is C, collected as C++ | 170 |
| Minified or data (a line over 500 characters) | 159 |
| Content looks like another language (C++/Objective-C in C, TypeScript in `.js`, ...) | 90 |
| `.h` file that is C++, collected as C | 56 |
| HTML with too little markup | 45 |
| Mostly embedded script/style or markup (counted only when another check also failed; otherwise set aside as hard) | 41 |
| Mostly non-ASCII | 31 |
| Not UTF-8 | 20 |
| Binary (NUL byte) | 2 |

At build time: 0 label drops, 0 duplicates (the collector already deduplicated within each language).

## Splits
- Stable hash-based per-language repo split (salt `codelangtm-v1`): 28 of each language's 140 repos go to test (20%); the rest are cut into 5 CV folds by a second hash. No repo is in both train and test (checked on every build).
- **The v6 test set has been scored once**, by the five baselines on 2026-09-30 (B-S5 check). Under the [analysis plan](analysis-plan-v6.md) no choice may use it.

## Licensing and redistribution
- MIT 5,720, Apache-2.0 2,696, BSD-3-Clause 438, BSD-2-Clause 139 snippets; licenses are checked per repository.
- Not redistributed (`data/` is gitignored). Every record keeps repo, commit and path.

## Known biases and limitations
- **Not reproducible from the collection commands alone.** GitHub's search order changes over time: only 1,000 of the 2,909 slice snippets collected a few hours earlier reappear in v6. Re-running the commands gives a different sample. The fix (refetching exact repo, commit, path and lines from the manifests) is planned before the v6 experiments ([roadmap](roadmap.md), review flags).
- **JavaScript vs TypeScript is partly undecidable from the window.** About a third of TypeScript windows contain no TypeScript-specific syntax (66% do, by an approximate pattern), against 5% of JavaScript windows. The label is the file extension. This caps what any model can reach on this pair.
- **C vs C++:** 10% of C++ windows contain no C++-only construct (C-like code in C++ files); 0% of C windows contain one (enforced by the label check).
- **Test-file share:** Ruby 44% (audit flag, threshold 40%), Go and PHP 32%. The shortcut probe found no test idioms among the top features.
- **Embedded code:** HTML windows that are mostly script/style are hard examples, which makes the HTML class easier than raw HTML files; report HTML with and without them. SQL-like lines fill 25% or more of 24 Python windows (3.7%), and 0-2 windows elsewhere; HTML tags fill over half of 8 C# and 8 JavaScript windows.
- **SQL is a mix of dialects**, and 6% of SQL windows are dominated by INSERT/VALUES rows.
- **Popularity:** every repo has 50+ stars and a permissive license; window length is 20-50 lines only (short snippets not covered).
- **Label noise:** confident learning flags 390 of 7,202 training snippets (5.4%): 223 JavaScript/TypeScript, 45 C/C++. The reviewed cases are ambiguous windows, not wrong labels.

## Intended use
- **In scope:** training and evaluating language identifiers on multi-line windows of the 14 languages; comparing interpretable and classical models under leak-free evaluation; the 8-language subset (`--languages core`) for comparison with v5.
- **Out of scope:** single-line or very short snippets, languages outside the 14, code quality or authorship.

## Reproduction

```bash
uv sync --extra tm --extra collect
uv run codelangtm collect github --repos 140 --per-repo 5 --pages 4 --out data/full/github --hard-out data/full-hard/github --language python --language cpp --language java --language javascript --language rust --language go --language html
uv run codelangtm collect github --repos 140 --per-repo 5 --pages 4 --out data/full/github --hard-out data/full-hard/github --language c --language csharp --language typescript --language php --language kotlin --language ruby
uv run codelangtm collect github --repos 140 --per-repo 5 --pages 4 --out data/full/github --hard-out data/full-hard/github --language sql
uv run codelangtm data build --source data/full --hard-source data/full-hard --out data/processed-v6
uv run codelangtm data audit --data data/processed-v6 --out data/processed-v6/audit.md
```

These commands reproduce the *procedure*, not the exact sample (see the first limitation).
