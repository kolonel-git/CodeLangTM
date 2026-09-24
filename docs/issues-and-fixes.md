# Issues & fixes

Engineering log of problems hit while building CodeLangTM: symptom, root cause, fix, and measured impact. Newest first within each section. Short status lives in [PROGRESS.md](../PROGRESS.md).

## Dataset evolution (Stage A)

| Version | Change | Loaded | Dropped at build | Final | SQL | HTML |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | First collection (25 repos × 5 snippets, >= 50 stars) | 829 | 0 | 829 | 40 | 97 |
| v2 | SQL top-up (`--min-stars 10`) + template filter | 866 | 21 (template-heavy) | 845 | 71 | 82 |
| v3 | Cut-safe windows, full re-collection | 861 | 0 | 861 | 77 | 92 |
| v4 | HTML embedded-language rule, HTML re-collected | 869 | 0 | 869 | 77 | 100 |
| v5 | Same snippets, stable hash-based split (M4) | 869 | 0 | 869 | 77 | 100 |

v3/v4 audit: no flags; largest single repo <= 6% of any language; median windows 0-10% comments.

---

## Modelling

### M6. Frequency-ranked vocabulary picks generic n-grams (fixed: label-aware selection)
- **Symptom:** first ablation run (`docs/ablations.md`): bigrams alone beat the 2+3-gram base at the same M=500 (logistic regression CV 0.944 vs 0.917, paired Δ +0.027 ± 0.021; Naive Bayes +0.036 ± 0.021). Adding 4-grams (n=2+3+4) is worse still (-0.011).
- **Root cause:** the binarizer keeps the top M n-grams by document frequency across all languages. With 2+3-grams, 183 of the 500 slots go to 3-grams that are common everywhere, such as `ing`, `ion`, `tio`, `ent`, `con` (English identifiers and comments) and runs of spaces. They push out 183 lower-ranked bigrams that separate languages better. Frequency is not discriminative power.
- **Also seen:** vocabulary size is the largest effect (M=100: -0.131; M=1000: +0.014 and still rising), consistent with useful features sitting below the frequency cut-off.
- **Fix:** `Binarizer(selection=...)` with two label-aware options, fit on each fold's training part only (a test spies on the labels `fit` receives):
  - `chi2`: rank candidates by chi² association with the language labels;
  - `class_balanced`: languages take turns picking their next most distinctive candidate, scored P(g | language) − P(g | other languages), so every language gets an equal share of M.
- **Result (dataset v5, CV, M=500, 2+3-grams):**

  | Selection | Logistic regression | Naive Bayes |
  | --- | --- | --- |
  | frequency (old) | 0.917 | 0.857 |
  | chi2 | 0.958 (Δ +0.041 ± 0.017) | 0.952 (+0.095 ± 0.037) |
  | class_balanced | 0.959 (Δ +0.042 ± 0.014) | 0.960 (+0.103 ± 0.034) |

  Largest per-language gains (LR): SQL 0.87 → 0.99, JavaScript 0.86 → 0.92, C++ 0.88 → 0.92. With label-aware selection the vocabulary size stops mattering (M=2000 adds +0.002 to +0.006, within noise), whereas frequency ranking needs M=2000 to reach the same level. `min_df` 1/5/20 makes no difference.
- **Lesson:** the feature budget, not the model, was the bottleneck. Naive Bayes, which cannot re-weight away uninformative features, gained the most (+0.10).

### M5. Repetitive list-like code predicted as SQL (fixed by M6, not by keywords)
- **Symptom:** v4 confident learning flagged 3 correctly labelled Python, Java and C++ windows as SQL (p 0.77-0.85). All three are long runs of near-identical lines such as `mapDecoration("white_banner", 10),` or `Round::deregisterNode(pluginFn);`.
- **Root cause (first guess):** many SQL windows are `INSERT ... VALUES` rows, so SQL's top features are punctuation (`),`, `␠(`, `(\n`), and 2/3-character n-grams cannot see whole keywords such as `SELECT`.
- **Tested:** whole-word features (`word_tokens`: identifier and keyword tokens such as `SELECT`, `fn`, `impl` join the candidates). No gain: +0.003 ± 0.008 with frequency selection and within noise with label-aware selection (class_balanced 0.959 → 0.963 ± 0.019, chi2 0.958 → 0.955), at ~10% extra binarize time.
- **Actual fix:** label-aware selection (M6). SQL out-of-fold F1 rose 0.87 → 0.99 without word features. The real cause was the frequency cut-off pushing SQL-specific n-grams out of the vocabulary, not the n-gram length: with `class_balanced`, SQL's first picks are `SE`, `EL`, `ELE`, `EC`, `ECT`, `SEL`, fragments of `SELECT` that 2/3-grams can see. Word tokens stay available but off.

### M4. Test score swung 5.5 points after an HTML-only change (fixed: stable split)
- **Symptom:** after re-collecting only HTML (v3 → v4), logistic regression CV macro-F1 stayed flat (0.920 → 0.923) but test macro-F1 fell 0.951 → 0.896.
- **Root cause:** not the model. Changing HTML repositories changed the group list, so `StratifiedGroupKFold` reshuffled which repos of *every* language went to test (e.g. Python 90/24 → 93/21 train/test). With only 39 test repos, the test score is very sensitive to that draw.
- **Evidence:** same v4 data and model, 10 different split seeds: test macro-F1 0.870-0.978, mean 0.929 ± 0.030, matching CV (0.923).
- **Options weighed:** (A) stable split, (B) keep StratifiedGroupKFold and never compare test across versions, (C) freeze v4's test repo list in a lock file. B leaves every future data change (Stage B) reshuffling test and leaves seed choice open; C adds a lock file for little gain. A chosen because the dataset will change again and switching now costs one baseline rerun, versus redoing ablations and TM training later.
- **Design check:** pure hashing (repo in test if hash < 0.2) was simulated on v4 and rejected: Java would get 2 test repos (5%), Python 9 (35%), some CV folds a single repo. Chosen design: within each language, order repos by hash and take the first round(n × 0.2) for test; order the rest by a second hash and cut into 5 equal folds.
- **Guarantees (tested):** exact test repo count per language; fold sizes differ by at most 1 repo; other languages unaffected by changes to one language; adding/removing one repo moves at most one existing repo in or out of test (50 randomised trials).
- **Fix:** `stable_split` (salt `codelangtm-v1`) used by `data build` → dataset v5 (same snippets as v4). `codelangtm baselines` adds repeated test macro-F1 over 10 further balanced splits (salts `…:repeat:k`), reported as mean ± std (min-max), never used for selection.
- **Result (v5):** logistic regression CV 0.917 ± 0.008, single test 0.943, repeated test 0.931 ± 0.007. CV and repeated test now agree within ~0.015.

### M1. Feature vocabulary could leak across CV folds
- **Risk:** fitting the n-gram vocabulary once on all training data, then cross-validating, lets n-grams that appear only in the validation fold shape the features. A subtle leak that inflates CV scores.
- **Fix:** `Binarizer` became a scikit-learn transformer inside `Pipeline([Binarizer, model])`, cloned per fold. A test spies on `Binarizer.fit` and asserts it never receives test or held-out fold snippets.

### M3. Baseline-driven data checks (confident learning + shortcut probe)
- **Method:** out-of-fold logistic-regression probabilities on train (never test); a snippet is a label-issue candidate when the model's confidence in another language exceeds that language's average self-confidence (Northcutt et al. confident learning). Shortcut probe: top-weighted n-grams per language and how many repos each appears in.
- **Label issues:** 6 of 689 (0.9%): 3 HTML windows that are entirely inside `<script>` blocks (content is JavaScript), 1 JavaScript window that is mostly an HTML template string, 1 Rust FFI struct mirroring C (`#[repr(C)]`, `c_uint`; label correct, hard example), 1 Java interface of bare signatures with Chinese Javadoc (label correct, little signal).
- **Root cause of the HTML cases:** the "HTML must contain a tag" check used `<\s*[a-zA-Z]`, which also matches comparisons such as `i < elements`. Across the dataset, 9 of 92 HTML windows have < 20% markup lines.
- **Shortcuts:** none. Every top feature appears in 13-20 repos and is real syntax (`def`, `func`, gofmt tabs, `::`, `let`, `public`, `--`, `<`). Test idioms (`assert`, `t.Run`, `@Test`) are not among top features, so the uneven test-file share is not acting as a shortcut.
- **Status:** fixed with an embedded-language rule for HTML and a stricter tag pattern; see D5.

### M2. Latency is dominated by feature extraction
- **Finding:** end-to-end baseline latency is ~0.31 ms/snippet, and binarization alone is ~0.31 ms. Model inference (even random forest) is negligible. Measured latency varies between runs (0.31 on v4, 0.43-0.48 on v5 for the same pipeline, on the same machine with other load); treat single-run latency as approximate until a dedicated benchmark (M6).
- **Implication:** the < 0.1 ms target is a feature-extraction problem. Planned: faster Python binarization in M2, and C feature extraction in M6.
- **Ablation finding:** delimiters cost ~40% of binarize time (0.315 vs 0.185 ms/snippet) for a gain within fold noise (+0.012 ± 0.014). The delimiter list contains 1- and 4-character items (`;`, `\t`, 4 spaces), so `transform` extracts all 1-grams and 4-grams of every snippet just to test them. Cheap fix for the faster-binarization task: test delimiters by direct substring search instead of adding their lengths to the n-gram pass.

---

## Data quality

### D5. HTML windows that are really JavaScript
- **Symptom:** found by confident learning (M3): HTML-labelled windows consisting of `<script>` code; model predicts JavaScript with p > 0.8.
- **Root cause:** window lies inside an inline `<script>` block, and the tag check `<\s*[a-zA-Z]` was satisfied by comparison operators (`i < elements`).
- **Measured:** markup-line share < 10%: 6 of 92 HTML windows; < 20%: 9; < 30%: 16.
- **Fix:**
  1. `HTML_TAG` requires a real tag: `<name` or `</name` ending in space, `>`, `/` or end of line, not directly after an identifier or `)`/`]` (so `a<b` does not match), or `<!`.
  2. Embedded-language rule: HTML windows with tags on fewer than 20% of non-blank lines are dropped as `mostly embedded script/style`.
- **Result on v3 raw data:** exactly the 9 predicted windows dropped (5 mostly script, 4 with no real tags); no other language affected. A 25%-markup window (VisualDL) is kept as a hard example.

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
