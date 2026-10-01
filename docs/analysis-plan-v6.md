# Analysis plan for the v6 experiments (DRAFT)

**Status: draft, 2026-10-01. Not frozen.** Points marked **[decide]** are open; each has a recommended option. Once every point is decided, this file is committed as *frozen* before any TM run on v6. After that, any change is added to the amendment log at the bottom with the date, the reason, and whether it was made before or after seeing a result. Plain-language background: [concepts.md](concepts.md) (analysis plan, equivalence test, ambiguity ceiling).

## Why a plan
So far many choices were made after looking at results (epoch rule, a verdict rule, thresholds, collection settings; all logged in [issues-and-fixes.md](issues-and-fixes.md) and [PROGRESS.md](../PROGRESS.md)). Each was reasonable, but together they make the results hard to trust as research: with enough small choices, a good-looking number can appear by chance. Writing the choices down before the v6 runs, and sticking to them, removes that doubt.

## 1. Data
- Dataset v6 as described in [dataset-card-v6.md](dataset-card-v6.md): `data/processed-v6/`, SHA-256 prefixes train `80a71a412bc8`, test `a20567d1250d`, hard `60b3f50662ef`, folds `5c7b8ed4a599`. Any rebuild that changes a hash is a new dataset version and invalidates the plan.
- **Reproducibility first:** a `collect from-manifest` command (refetch exact repo, commit, path and line range) is added and checked against these hashes before the experiments start. **[decide]** recommended: yes, before freezing; alternative: after M4.
- **Test set lock:** the v6 test set was scored once on 2026-09-30 (baselines, B-S5 check). From now on it is used once more, for the final head-to-head in section 6, and never for a choice. Every choice (features, TM settings, epochs, smallest model) uses CV on train only. The repeated-split score pools train and test, so it is never used for choices either.

## 2. Languages and views
- **Primary:** all 14 languages (`--languages all`).
- **Secondary:** the original 8 (`--languages core`, same folds), for comparison with v5.
- **JavaScript/TypeScript framing [decide]** (about a third of TypeScript windows contain no TypeScript syntax; the label is the extension):
  - (a) **recommended:** keep 14 classes as the primary result, and add two secondary views computed from the same predictions: a *pair-merged* score (JavaScript+TypeScript as one class, C+C++ as one class) and the per-pair confusion. Also state an approximate *ambiguity ceiling* for the pair (the share of windows without any distinguishing syntax).
  - (b) merge JavaScript and TypeScript into one class for training too (13 classes): simpler, but loses a real distinction in the 66% of TypeScript windows that show it.
  - (c) report 14 classes only and explain the pair in the text.

## 3. Targets [decide]
The README targets (macro-F1 >= 96%, < 0.1 ms per snippet, < 500 KB) were set for 8 languages.
- Recommended:
  - **accuracy:** 96% macro-F1 stays the target for the original 8. For 14 languages the target is relative: the TM within the equivalence margin (section 6) of the best baseline, because the JavaScript/TypeScript ceiling makes an absolute 96% unreachable for any model.
  - **size:** < 500 KB for the 14-language model.
  - **speed:** < 0.1 ms per snippet, measured in M6 (C runtime) as planned.
- Alternative: keep 96% for 14 languages and report it as missed.

## 4. Features (CV only)
- Rerun the ablation config (`configs/ablations.yaml`, all studies) on v6 train with `--languages all`, models logistic regression and Naive Bayes.
- **Selection rule, fixed now:** for each setting take the mean CV macro-F1 of the two models. Among settings whose paired difference to the best setting is within one fold standard deviation of that difference, choose the one with the smallest M, then the fewest n-gram sizes, then delimiters off. That setting is frozen for every v6 model (baselines and TM).
- Slice result for reference (not used for the choice): class-balanced M=1000 was +0.008 +/- 0.009 over M=500, M=2000 no better.

## 5. TM settings and training (CV only)
- **Settings [decide]:** recommended: 400 and 800 clauses per class (T=100, s=5, weighted clauses), the v5 main setting and a doubled one for 14 classes; 5 seeds (1-5). Alternative: 400 only (cheaper, about 1 h per full protocol on v6).
- Epochs from `tm-curve` with the existing rule (`epoch_rule: max`, 11-epoch smoothed held-out curve, max 150 epochs), on v6 CV folds.
- The main TM setting is the one with the higher mean CV macro-F1 over seeds; the official model is the median-CV seed (`tm-select`), as in v5.

## 6. Primary comparison
- **Question:** is the main TM setting equivalent to the best baseline on v6, 14 languages?
- **Best baseline:** the baseline with the highest CV macro-F1 on v6 with the frozen features (expected: logistic regression; it is chosen by the rule, not by name). Naive Bayes is reported as a second comparison.
- **Measure:** macro-F1 on the 10 repeated repo-level splits (paired with the baseline on the same splits), with the CV folds as a second view.
- **Test:** equivalence test (two one-sided tests, TOST) on the per-split differences with the Nadeau-Bengio corrected variance, as already used. **Margin [decide]:** recommended +/- 0.010 macro-F1 (one point). Verdict: *equivalent* if the 90% interval lies inside +/- margin; *worse* if it lies entirely below -margin; *better* if entirely above +margin; otherwise *inconclusive*. The same test on the original 8.
- **Uncertainty on the single test set:** bootstrap over test repositories (1,000 resamples, whole repos resampled within each language) for macro-F1 and each language's F1.
- **Multiple comparisons:** the primary family is {TM vs best baseline on 14 languages, TM vs best baseline on the original 8}; Holm correction across it. Everything else is reported as descriptive.

## 7. Secondary analyses (descriptive)
- Per-language F1 and confusion matrices; pair-merged view (section 2).
- Hard examples: scored only on the 41 from test repos; HTML reported with and without the hard-example policy.
- Error analysis out of fold (`codelangtm errors`) on v6 train.
- **[decide]** Short snippets: recommended: cut 1, 3, 5 and 10-line windows from the v6 *test* repos (no new collection) and report F1 by length for every model. Alternative: keep in Future.
- **[decide]** External baseline: recommended: run Guesslang (open source) on the v6 test set with a fixed language mapping, reported next to ours. Alternative: none.

## 8. Interpretability measures [decide]
Recommended, defined now and measured for the TM and for logistic regression on the same test windows:
- **Explanation size:** the number of clauses (TM) or features (logistic regression) needed to account for 90% of the winning language's vote margin over the runner-up.
- **Rule length:** literals per clause that fired, and per explanation.
- **Faithfulness check:** removing the explanation's features flips or keeps the prediction (share of flips).
- Alternative: postpone to M5.

## 9. Not allowed after freezing
Changing features, settings, epoch rule, margin, primary measure or the language framing because of a v6 TM result. If something has to change (a bug, a broken assumption), it goes into the amendment log, and the report shows both the planned and the amended result.

## Amendment log
| Date | Change | Reason | Before or after seeing a result |
| --- | --- | --- | --- |
| | | | |
