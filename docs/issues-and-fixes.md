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

## Data quality

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
