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

### M11. The "Python sink" did not replicate, and a verdict rule was too weak (B6b)
- **Symptom:** on the test set the TM sent other languages' snippets to Python (Java→Python 8 over 5 seeds), so B6b set out to explain a Python sink. On out-of-fold CV predictions (every training snippet, 5 seeds), Python receives only 21 of the TM's 178 errors (12%), *below* the 14% that an even spread over the other 7 languages would give.
- **Second problem:** hypothesis H1 ("the sink is real") was first written as "the TM sends a larger share of its errors to Python than the baselines do". Python passed (TM 12% against 4% and 9%), although it receives fewer errors than chance. The rule compared against the wrong reference.
- **Fix:** H1 now also requires a share above chance. This was changed after seeing the data and is marked as such in [errors.md](errors.md) and the report. With the user's agreement, the probes were then run on the language that really collects most errors out of fold: JavaScript (52 of 178, 29%).
- **Lesson:** the test set has 173 snippets and about 5 TM errors per seed. A pattern in so few errors, most from a handful of repositories, is anecdote, not evidence. The out-of-fold view gives ~36 errors per seed, all without spending the test set.

### M10. Very long clauses that memorise a few training snippets (B6, open: M4)
- **Symptom:** the clause inspector found 173 of the official model's 3,200 clauses with more than 40 literals (up to 268), almost all `NOT has(...)` parts, 157 of them "for" clauses. They fire on few training snippets (median 4; 40 fire on at most one) with median precision 1.0, and carry above-average weight (median |weight| 9, against 6 for all clauses).
- **Reading:** Type II feedback keeps adding "not this n-gram" literals until a clause stops firing on other languages. For odd snippets that no short rule covers, the clause ends up describing those few snippets. This is memorisation: it costs readability, and possibly generalisation.
- **Next:** M4 tries TMU's `max_included_literals` (a literal budget) and reports accuracy with and without these specialists. It stays unfixed here, because the frozen M3 setting is what the comparison with the baselines measured.

### M9. Formation measures that looked fine but meant nothing (B6, fixed before commit)
- **Symptom:** the first formation replay reported that 98.3% of include decisions already matched the final model after epoch 1, and that every signature n-gram appeared in a "for" clause at epoch 1.
- **Root cause:** about 99% of all (clause, literal) slots are "exclude" in any sparse model, so raw agreement is near 100% from the start. And after one epoch clauses hold about 9 literals each, so almost every n-gram is included somewhere by chance.
- **Fix:** compare only the included decisions (Jaccard similarity with the final include set: 0.10 after epoch 1, 0.48 after 20, 0.85 after 80), and follow each signature n-gram's *share* of the language's "for" clauses per epoch. Its *settled epoch* is the one from which that share stays at least half its final value (median: epoch 5).

### M8. The planned TM setting trails the baselines (B1 spike, CV only)
- **Setup:** throwaway spike, not committed. TMU 0.8.3 `TMClassifier` on the frozen features, 5 repo-grouped CV folds of train (the test set was not used), 150 epochs, macro-F1 on the held-out fold after every epoch. Naive Bayes on the same folds: 0.963 ± 0.005.
- **Symptom:** the planned starting point (N_c=100 clauses per class, T=30, s=3.5, weighted clauses) ends at 0.925-0.940 (mean of the last 20 epochs, 3 seeds), about 3 points below Naive Bayes. Training F1 reaches 1.0 by epoch 15-60 while held-out F1 stays lower: the model fits the training set completely.
- **Probe (6 settings, a sanity check rather than tuning; M4 tunes properly):**

  | Setting | Last-20-epoch mean | Best epoch mean (optimistic) | s/epoch |
  | --- | --- | --- | --- |
  | N_c=100, T=30, s=3.5 (seeds 42 / 7 / 123) | 0.925 / 0.940 / 0.934 | 0.951 / 0.949 / 0.946 | 0.028 |
  | same, unweighted clauses | 0.891 (falls after epoch ~30) | 0.944 | 0.046 |
  | N_c=100, T=15, s=3.5 | 0.935 | 0.949 | 0.028 |
  | N_c=100, T=30, s=6 | 0.916 | 0.946 | 0.028 |
  | N_c=200, T=50, s=3.5 | 0.950 | 0.962 | 0.031 |
  | **N_c=400, T=100, s=5** | **0.962** | 0.969 | 0.037 |

- **Reading:** capacity (more clauses, with T scaled up so the votes do not saturate) closes the gap; changing s alone does not. "Best epoch" is the maximum of 150 noisy means, so it flatters every setting; the last-20 mean is the fairer number.
- **Noise:** held-out F1 moves ±0.02 between consecutive epochs and the fold std is 0.02-0.03. A "smallest epoch within one std of the best" rule picks epochs 5-12 here, too early; B4 needs a smoothed curve or a window average.
- **Cost of the larger setting:** 8 × 400 = 3,200 clauses over 1,000 literals, 35,960 included literals (11.2 per clause). Bit-packed include matrix: 400 KB, close to the 500 KB target; stored sparsely (2-byte literal ids + clause offsets + weights) about 80 KB.
- **Decision (2026-09-27):** B3 uses N_c=400, T=100, s=5, weighted clauses as the main TM setting (chosen on CV only), with the planned N_c=100/T=30/s=3.5 kept as a reference row; both are reported over 5 seeds. Full tuning stays in M4, which also looks for the smallest model within ~1 F1 point.
- **Correction after B4 (5 seeds): the 0.962 was a lucky seed.** The B4 training curves (seeds 1-5, same folds, final code) give the 400-clause setting 0.945-0.950 per seed over the last 20 epochs (mean 0.948), not 0.962. Rerunning seed 42 through the final code reproduces 0.962 (per fold 0.915 / 0.970 / 0.969 / 0.966 / 0.991), so the code is consistent and seed 42 is an outlier. The probe above compared six settings on that single seed and kept the best, which is selection bias twice over (one seed, best of six). The ranking still holds (400 clauses beat 100 by about 0.016 over 5 seeds), but the TM trails Naive Bayes (0.963) by about 1.5 points, not zero. The multi-seed rule (roadmap Principles) exists for exactly this.

### M7. First resource numbers were misleading (fixed before commit)
- **Symptom:** the first "Resources" table showed fit CPU 2.7 s against 0.45 s wall time, and an identical 71.7 MB peak memory for all five models.
- **Root cause:** both came from one memory-traced fit. `tracemalloc` slows code down about 6×, which inflated CPU time; and the peak was the binarizer's candidate table (same for every model), which hid the differences between classifiers.
- **Fix:** CPU time is taken from a plain fit; memory is measured in two separate traced runs, binarizer (fit + transform) and classifier (on the binarized matrix), and reported in separate columns. Result: CPU ≈ wall time (random forest 0.94 s vs 0.76 s wall: threads); binarizer 71.7 MB for every model; classifier 1.4 MB (decision tree) to 4.3 MB (Naive Bayes).
- **Known limit:** `tracemalloc` sees only Python/NumPy allocations, not memory allocated inside C extensions (liblinear, TMU). The TM comparison in M3 therefore also measures process-level peak memory in a subprocess, for baselines and TM alike (roadmap M3).
- **Finding (baselines):** the pipeline is dominated by the binarizer, not the classifier: 72 MB and ~0.18 ms/snippet of feature extraction against ≤ 4 MB and ~0.01 ms for the classifier (except random forest at 7 MB pickled, 0.2 ms).

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
- **Ablation finding:** delimiters cost ~40% of binarize time (0.315 vs 0.185 ms/snippet) for a gain within fold noise (+0.012 ± 0.014). The frozen feature config therefore drops them (delimiters add nothing once selection is label-aware). After the faster binarizer (below) their share grew: rerun ablations show 0.114 ms/snippet with delimiters vs 0.029 without (~75% of binarize time).
- **Profile (frozen config, 1,174 chars/snippet):** 92% of `transform` was building the set of all 2- and 3-character substrings of each snippet (0.20 of 0.22 ms); the 500 membership tests were the other 8%.
- **Fix:** `Binarizer.transform` no longer builds substring sets. Each snippet becomes an array of code points; characters map to small ids, and 1-3 character terms are found by indexing a direct-address table (`table[id_a * base + id_b]`) with NumPy. Terms longer than 3 characters and word features keep the set method; if the alphabet were so large that a table would exceed 4M entries, the code falls back to a sorted-key binary search (5-6× slower than the table). Lookup tables are derived from the vocabulary, built once, and not pickled.
- **Correctness:** output is identical to testing `term in snippet` for every term. Fuzz tests compare against that definition on random ASCII, control, BMP and non-BMP strings (empty, 1-character and unseen-character inputs included), with n-gram lengths 1-4, word tokens, the delimiter list and the forced binary-search fallback. A mutation check (breaking the unknown-character id) fails 5 tests. Baseline scores are bit-for-bit unchanged (0.963 / 0.959 / 0.968).
- **Result:** interleaved old-vs-new runs in one process (so both see the same machine load), 5 runs: **5.4-6.0× faster** (old 0.47-0.54, new 0.078-0.099 ms/snippet on a machine running ~2× slower than earlier that day). The set method took 0.22 ms on the quieter machine, so the same ratio suggests ≈ 0.04 ms there; that figure is an inference, not a measurement. Absolute timings on this machine vary ±50% with background load, so whether the < 0.1 ms end-to-end target is met is still to be confirmed by the M6 benchmark on a quiet machine.
- **Next lever:** the remaining cost is per-snippet NumPy call overhead; the C export (M6) removes it.

---

## Data quality

### D8. Findings from the first Stage B slice (B-S3)
- **Slice:** 45 repos per language, at most 5 snippets per repo, 2,807 snippets from 14 languages (details: [PROGRESS.md](../PROGRESS.md)). Built into `data/processed-slice/` and audited by eye plus measurements on every snippet.
- **C labelled from C++ and Objective-C headers:** `.h` files are the weak spot. PowerToys' `InclusiveCrosshairs.h` (`constexpr`, `enum struct`, `winrt::Windows::...`) was labelled C because the first C++ markers were `class`/`namespace`/`template<`/`std::` only; yabai's `autorelease.h` is Objective-C. Fix: a larger set of C++-only constructs (shared by the `.h` decision and the C check; comment lines are skipped for `::`; `extern "C"` guards are not markers) plus Objective-C markers for C. On the slice it removes 3 of 415 C/C++ snippets (2 from C, 1 plain-C-looking header from a C++ repo). Checked with real cases in `tests/test_labels.py`, including C headers that must stay C (`extern "C"` guards, `Foo::bar` in comments, `class_id`, `int new`).
- **Detector recall is limited by design:** 75% of the `.cpp` windows contain a C++-only marker; the rest are C-like code and stay C++ because of their extension.
- **HTTP 429 without hints:** one Python repo was skipped (`HTTP 429`). GitHub's secondary rate limit does not always send `Retry-After`; a bare 429 now backs off 60 s x attempt and retries.
- **SQL shortfall (78 snippets, 21 repos), first diagnosis wrong:** I first blamed the collector for reading only page 1 of each search band and added `--pages N`. Rerunning SQL with `--pages 3` gave exactly the same 78 snippets from 21 repos. Asking GitHub directly showed why: the search class `language:"SQL"` holds only 71 repositories with 10+ stars in total (47 with 10-49 stars, 14 with 50-199, 8 with 200-999, 2 with 1,000-4,999), so there was nothing on page 2. GitHub files most repositories that are mainly `.sql` under a dialect name: `TSQL` (686 repos with 10-49 stars), `PLpgSQL` (645), and `PLSQL`.
- **Fix:** SQL is searched under all four names (`SQL`, `TSQL`, `PLpgSQL`, `PLSQL`), merged round-robin over names and star bands, with the same license, fork and star rules as every other language. At the standard floor of 50 stars this gives at least 135 candidate repositories (the cap I read; 87 MIT, 37 Apache-2.0, 11 BSD), so SQL no longer needs the lower `--min-stars 10` floor. Every file still has to be `.sql` and pass the SQL keyword check. The manifest records the names searched (`search_languages`). Consequence for the dataset card (v6): SQL now mixes dialects (generic, T-SQL, PL/pgSQL, PL/SQL) and the dialect mix is not measured per snippet.
- **`--pages N` stays** (default 1, tested) but nothing needed it: for every language the first page already gives enough candidates.
- **Ruby star floor:** Ruby was collected with `--min-stars 10` although it needs no such floor. Rerun at the standard floor of 50 it gave the same 224 snippets from 45 repos and the same drop counts (the top-ranked repositories of both searches coincide), so the earlier run is equivalent and the star-floor deviation applies to SQL only, and no longer to SQL after this fix.
- **Measured, no policy change:**
  - SQL-like content inside other languages: 9 of 211 Python windows have 25% or more SQL-like lines (4%); at most 1 window in every other language.
  - HTML tags on more than half the lines outside HTML: 3 C#, 3 JavaScript, 1 Ruby, 1 Rust window.
  - Heredocs the scanner does not track: about 2% of Ruby and PHP windows begin or end inside one. No C# verbatim-string cuts.
  - Hard HTML examples: 96 windows from 24 repos (median embedded share 0.81); 13 come from a single repository.

### D7. Embedded-language policy: hard examples instead of dropping (Stage B, B-S2)
- **Problem:** the 20%-markup rule of D5 drops only windows that are almost all script. B6b found that 21 of 76 HTML training windows are still more than half `<script>`/`<style>`, and 4 of them cause every out-of-fold HTML → JavaScript error. Dropping them would hide a real weakness; keeping them in training teaches "HTML looks like JavaScript".
- **Decision (user, 2026-09-29/30):** windows with more than 50% embedded code are set aside as **hard examples**, per window (not per repository, to keep as much good data as possible): reported on, never trained on. Applies to HTML (lines inside `<script>`/`<style>`, measured by `labels.embedded_share`, moved from `errors.py`) and to PHP (more than 50% of lines carry an HTML tag: templates). A window with almost no tags that is not embedded code is still dropped ("too little markup").
- **Implementation:**
  1. `check_label(...).hard` is true only when the sole reason is an embedded one; any other failure (template-heavy, extension mismatch, ...) is a plain drop.
  2. The collector tries up to 3 random windows per file, so a mixed file still gives its good part to the dataset; at most one hard window per file is kept, from repos that gave usable data, deduplicated against the dataset. Hard windows never count toward the per-repo quota.
  3. Hard files live in `data/hard/`, not `data/raw/`, so `data build` cannot pick them up as training data by accident. `data build` writes `hard.jsonl`, records counts and where each hard window's repo sits (train, test, neither) in `dataset.json`, and never lets hard windows change train, test or folds (tested: identical hashes with and without).
- **Measured on v5 raw (real data):** threshold 0.5 moves 25 of 100 HTML windows to the hard file (0.3: 34, 0.4: 30, 0.6: 20, 0.7: 14); HTML would have 75 windows (55 train, 20 test); the other 7 languages are unchanged. Of the 25 hard windows, 17 come from repos in train, 4 from repos in test, 4 from repos with no normal window at all.
- **Consequences:** v5 as published (100 HTML windows) stays reproducible from the old rule only via the earlier code; all v5 numbers in the report still include these windows. Evaluating on hard windows from *train* repos is not leak-free (the model saw other windows of the same repo): the manifest's `repo_in` split says which are fair to use (test repos and "neither").
- **Not covered:** SQL inside other languages (Python/Java strings) and JavaScript dominated by HTML templates; to be measured on the slice audit (B-S3) before deciding.

### D6. Adding six languages to the code (Stage B, B-S1)
- **What changed:** C, C#, TypeScript, Kotlin, PHP and Ruby are known to the collector, the label check, the comment/string scanner, `data build` and `data audit`. Evaluation code still defaults to the 8 core languages (`LANGUAGES`) until the v6 dataset exists; `ALL_LANGUAGES` has all 14.
- **Traps found and handled:**
  1. `.ts` is also the extension of Qt translation files (XML) and video streams: a TypeScript red flag rejects `<?xml` and `<TS` lines.
  2. A `.c` file can hold C++ (`namespace`, `class X`, `template<`, `std::`): red flag for C. Line-anchored, so a comment like ` * class of problems` passes (tested).
  3. `.h` is C or C++: the existing content rule decides; the collector now samples `.h` for both languages, so a C header in a C++ repo is dropped as an extension mismatch and vice versa.
  4. C# `using System` in a `.java` file, `import java.` in a `.cs` file: red flags in both directions.
  5. `.jsx`/`.tsx` mix markup into code (the embedded-language problem of D5). Decision: keep them valid labels (v5 has 3 `.jsx` and 8 `.mjs` JavaScript snippets, so rebuilding v5 stays byte-identical) but do not sample them in new collections; `.mjs` is plain JavaScript and stays.
  6. Generated and build-output files: `.g.cs`, `.designer.cs`, `.pb.cc`, `.pb.h`, `_pb.rb`, and `obj/` and `bin/` directories are skipped.
- **Known limits (checked on the real slice in B-S3, not assumed away):** the scanner does not track C# verbatim strings (`@"..."`) or Ruby/PHP heredocs that span lines, so a window can still begin or end inside one; PHP templates that are mostly HTML are handled by the embedded-language policy (B-S2).
- **Check:** rebuilding v5 from `data/raw/` with the new code gives byte-identical `train.jsonl`, `test.jsonl`, `folds.json` (same SHA-256 as `dataset.json`).

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

### E6. TMU training hangs forever with `seed=0`
- **Symptom:** while writing B2's tests, a tiny TM (4 clauses, 3 classes) never finished its first epoch. Any clause count or T hung the same way, yet the same model trained in 0.05 s in other tests.
- **Cause found by elimination:** the difference was the seed. `seed=0` hangs; 1 and 42 do not. TMU passes the seed to `lib.pcg32_seed` and `lib.xorshift128p_seed` in its C code. A xorshift128+ generator whose state is all zeros returns 0 forever, and a TMU loop that draws random numbers until a condition holds then never ends. That mechanism is inferred from the source and the behaviour (the compiled C was not stepped through); the trigger, seed 0, is confirmed.
- **Fix:** `TMLanguageClassifier` rejects seeds below 1 (and non-integers) with a clear message; the default seed is 1. Seeds for multi-seed runs start at 1. Test: seeds 0, -1, 1.5 and `True` are rejected before TMU is touched.
- **Lesson:** the B1 spike used seed 42 and never met this; a test suite with small, fast models surfaced it.

### E5. TMU prints a pycuda traceback on import
- **Symptom:** importing `tmu.models.classification` logs a WARNING and an ERROR with a full `No module named 'pycuda'` traceback.
- **Root cause:** TMU tries to load its CUDA backend at import and logs the failure with `_LOGGER.exception`. Harmless: the CPU backend works.
- **Fix (B2):** raise the level of the `tmu.clause_bank.clause_bank_cuda` and `tmu.util.cuda_profiler` loggers before importing TMU, so real errors still show.

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

### W7. CI failed after the M3 merge: "peak" memory below "current" on Linux
- **Symptom:** `test_memory_is_measured_and_grows` failed on the Ubuntu CI runner after PR #7 was merged (it passes on Windows): current 318.68 MB but peak 318.44 MB.
- **Root cause:** on Linux, `resources.memory_mb` read current memory from `/proc/self/status` (`VmRSS`) but the peak from `getrusage` (`ru_maxrss`). The kernel updates `ru_maxrss` lazily, so right after new pages are touched it can lag behind the current value. The test had just touched 80 MB.
- **Fix:** read both from the same `/proc/self/status` snapshot (`VmRSS` and `VmHWM`; the kernel reports `VmHWM` as at least the current value), keep `getrusage` only as a fallback, and never report a peak below the current value. A new test simulates the lagging `getrusage` on any OS.
- **Impact on results:** none. The B5 resource numbers were measured on Windows, which reads both values from one `GetProcessMemoryInfo` call.
- **Lesson:** CI runs on a different OS than development; code with OS-specific branches needs a test that exercises each branch everywhere (here by faking the Linux inputs).

### W6. Pickling a model silently stripped it (fixed before commit)
- **Symptom:** B2 smoke test: after `pickle.dumps(pipeline)`, calling `predict_tmu` on the *original* model failed with "not fitted".
- **Root cause:** `__getstate__` removed the TMU object from the state dict. Since Python 3.11, `object.__getstate__()` (which scikit-learn's `BaseEstimator.__getstate__` calls) can return the object's **live** `__dict__`, not a copy, so the removal hit the model itself. `Binarizer.__getstate__` had the same pattern (harmless there: it only drops a cache that is rebuilt on demand).
- **Fix:** both copy the dict first (`dict(super().__getstate__())`). Regression test: after pickling, the original still has its TMU object and keeps training; the unpickled copy predicts identically and refuses to train with a clear message.

### W5. Keeping figure files stable in git
- **Goal:** redrawing a figure from the same sidecar must give the same bytes, on this machine and in CI, so regenerated PNGs only show up in git when the numbers change.
- **Risks found while building `figures.py`:** default PNGs carry a `Software: matplotlib <version>` stamp (a different matplotlib version would change every file); system fonts differ between Windows and Linux; tick labels are created at save time, so a style applied only while building the figure would not reach them.
- **Fix:** `figures.save_png` passes `metadata={"Software": None}` and saves inside the same style context used to build the figure; the font is DejaVu Sans (bundled with matplotlib); size and dpi are fixed. Tests: two renders are byte-identical, and PNGs contain no `matplotlib`/`Software` bytes (checked: a default save does contain them, so the test would catch a regression).

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
