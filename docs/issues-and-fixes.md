# Issues & fixes

Engineering log of problems hit while building CodeLangTM: symptom, root cause, fix, and measured impact. Newest first within each section. Short status lives in [PROGRESS.md](../PROGRESS.md).

## Dataset evolution (Stage A)

| Version | Change | Loaded | Dropped at build | Final | SQL | HTML |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | First collection (25 repos × 5 snippets, >= 50 stars) | 829 | 0 | 829 | 40 | 97 |
| v2 | SQL top-up (`--min-stars 10`) + template filter | 866 | 21 (template-heavy) | 845 | 71 | 82 |
| v3 | Cut-safe windows, full re-collection | 861 | 0 | 861 | 77 | 92 |

v3 audit: no flags; largest single repo <= 6% of any language; median windows 0-10% comments.

---

## Modelling

### M1. Feature vocabulary could leak across CV folds
- **Risk:** fitting the n-gram vocabulary once on all training data, then cross-validating, lets n-grams that appear only in the validation fold shape the features. A subtle leak that inflates CV scores.
- **Fix:** `Binarizer` became a scikit-learn transformer inside `Pipeline([Binarizer, model])`, cloned per fold. A test spies on `Binarizer.fit` and asserts it never receives test or held-out fold snippets.

### M3. Baseline-driven data checks (confident learning + shortcut probe)
- **Method:** out-of-fold logistic-regression probabilities on train (never test); a snippet is a label-issue candidate when the model's confidence in another language exceeds that language's average self-confidence (Northcutt et al. confident learning). Shortcut probe: top-weighted n-grams per language and how many repos each appears in.
- **Label issues:** 6 of 689 (0.9%): 3 HTML windows that are entirely inside `<script>` blocks (content is JavaScript), 1 JavaScript window that is mostly an HTML template string, 1 Rust FFI struct mirroring C (`#[repr(C)]`, `c_uint`; label correct, hard example), 1 Java interface of bare signatures with Chinese Javadoc (label correct, little signal).
- **Root cause of the HTML cases:** the "HTML must contain a tag" check used `<\s*[a-zA-Z]`, which also matches comparisons such as `i < elements`. Across the dataset, 9 of 92 HTML windows have < 20% markup lines.
- **Shortcuts:** none. Every top feature appears in 13-20 repos and is real syntax (`def`, `func`, gofmt tabs, `::`, `let`, `public`, `--`, `<`). Test idioms (`assert`, `t.Run`, `@Test`) are not among top features, so the uneven test-file share is not acting as a shortcut.
- **Status:** fix proposed (embedded-language rule for HTML + stricter tag regex); see D5.

### M2. Latency is dominated by feature extraction
- **Finding:** end-to-end baseline latency is ~0.31 ms/snippet, and binarization alone is ~0.31 ms. Model inference (even random forest) is negligible.
- **Implication:** the < 0.1 ms target is a feature-extraction problem. Planned: faster Python binarization in M2, and C feature extraction in M6.

---

## Data quality

### D5. HTML windows that are really JavaScript (open)
- **Symptom:** found by confident learning (M3): HTML-labelled windows consisting of `<script>` code; model predicts JavaScript with p > 0.8.
- **Root cause:** window lies inside an inline `<script>` block, and the tag check was satisfied by comparison operators.
- **Measured:** markup-line share < 10%: 6 of 92 HTML windows; < 20%: 9; < 30%: 16.
- **Proposed fix:** tag regex requires a real tag (`<name` / `</name` followed by space, `>` or `/`); drop HTML windows where fewer than 20% of non-blank lines contain markup (embedded-language rule). Pending decision.

### D4. Windows cut through comments and strings
- **Symptom:** during manual review of `audit.md`, some snippets began or ended mid-comment. Prose looked like code, and real code after a closing `*/` or `"""` looked commented out.
- **Root cause:** `extract_windows` cut at arbitrary line numbers with no knowledge of block comments (`/* */`, `<!-- -->`) or multi-line strings (Python `"""`, JS/Go backticks, Java text blocks).
- **Impact before fix:** 40 of 866 snippets (4.6%): Python 14, Java 11, JavaScript 7, Go 6, C++ 1, HTML 1. Worst cases: a "Go" window that was entirely HTML inside a backtick string; a "Python" window that was entirely prompt prose inside a docstring.
- **Fix:**
  1. `syntax.py`: a minimal per-language scanner that tracks whether each line starts inside a block comment or multi-line string (it also tracks line comments and single-line strings, so `"/*"` or `// /*` are not mistaken for openers).
  2. `extract_windows(..., language=...)`: windows only start and end on lines outside comments/strings; the length closest to the random target that satisfies this is chosen.
  3. `check_label` gains `cut comment or string` (unterminated at end, or a stray closing `*/`/`-->` at start) as a safety net for existing and hand-collected wild data.
- **Result:** re-collection produced 861 snippets with 0 build-time drops.
- **Known limits:** Rust nested block comments and raw strings (`r#"..."#`), C++ raw strings, and a Python window starting mid-docstring that contains another docstring later are not fully handled. Rare in practice.

### D3. Template syntax inside SQL and HTML
- **Symptom:** sampled SQL came from dbt projects full of Jinja (`{%- macro -%}`, `{{ ref() }}`); HTML included Jekyll/Liquid (`{{ site.url }}`, `---` front matter) and Django templates.
- **Risk:** the model learns shortcuts ("`{%` means SQL") instead of language structure.
- **Fix:** `check_label` drops SQL/HTML windows where more than 30% of non-blank lines contain `{% %}`, `{{ }}` or `<% %>` (plus `---` front matter for HTML). Other languages are exempt because `{{` is legitimate there (format-string escapes, JSX style props, nested initialisers). `---` is not counted for SQL, where it is a comment.
- **Result:** 21 windows dropped (HTML 15, SQL 6) from 6 repos. 11 windows at 17-29% templating are kept by design; revisit if they show up in error analysis.

### D2. SQL scarcity and class imbalance
- **Symptom:** SQL had 40 snippets from 10 repos vs ~110 for other languages; only 6 SQL test snippets (one error = ~17 F1 points).
- **Root cause:** few repos that GitHub classifies as SQL also have a permissive license and >= 50 stars.
- **Fix:** SQL-only re-collection with `--min-stars 10` (recorded in `sql.manifest.json`). Result: 77 snippets from 21 repos.
- **Remaining:** SQL is still the smallest class. Accepted for Stage A; macro-F1 will expose it. Planned fix: The Stack loader in Stage B, not lowering the star bar further.

### D1. HTML windows with no markup
- **Symptom:** during v1 collection, 21 HTML windows were dropped as "expected markers missing".
- **Root cause:** windows fully inside `<script>` or `<style>` blocks; they are JavaScript/CSS, not HTML.
- **Decision:** dropping them is correct. The embedded-language policy (Stage B) will formalise this for Python-with-SQL and JS-with-HTML too.

---

## Environment

### E4. `uv sync` removed TMU
- **Symptom:** after a plain `uv sync`, TMU disappeared from the venv.
- **Root cause:** `uv sync` makes the venv match exactly the requested extras; unrequested extras are uninstalled.
- **Fix:** always `uv sync --extra tm --extra collect`.

### E3. TMU crashed on NumPy 2
- **Symptom:** `OverflowError: Python integer -1 out of bounds for uint32` in `tmu/clause_bank/clause_bank.py`.
- **Root cause:** TMU 0.8.3 uses `np.uint32(~0)`, which NumPy 2 rejects. Newer SciPy also requires NumPy 2.
- **Fix:** `tm` extra pins `numpy<2` and `scipy<1.14`. Verified with a toy XOR run (93.5% accuracy, CPU backend).

### E2. Stale virtual environment
- **Symptom:** `No Python at '...cpython-3.12.12...\python.exe'`.
- **Root cause:** `.venv` pointed to a uv-managed Python that did not exist on the machine.
- **Fix:** `uv python install 3.12` then `uv sync --reinstall`.

### E1. Python 3.14 vs TMU
- **Decision:** pin Python 3.12 with uv; TMU builds a C extension and had no 3.14 support.

---

## Tooling & workflow

### W4. `±` printed as `�` in the console
- **Root cause:** Windows console uses cp1252, which cannot encode `±`.
- **Fix:** console tables use ASCII `+/-`; generated Markdown files (UTF-8) keep `±`.

### W3. Audit file looked stale
- **Symptom:** after regenerating, `audit.md` appeared unchanged in the editor.
- **Root cause:** the file was regenerated, but VS Code keeps showing an open buffer with unsaved edits (ticked boxes).
- **Fix:** "File: Revert File" or close without saving. The audit header now shows a generation timestamp and snippet count, so staleness is visible.

### W2. `git branch -d` refused to delete a branch
- **Root cause:** the local branch was 1 commit ahead of its remote copy; `-d` checks the upstream, not the branch it was merged into.
- **Fix:** verified the commit was contained in the pushed branch, then `git branch -D`. Rule since: delete branches only after their PR is merged.

### W1. First push rejected (non-fast-forward)
- **Root cause:** the GitHub repo was created with an initial commit the local repo did not have.
- **Fix:** `git pull origin main --allow-unrelated-histories`, keep local files, push.
