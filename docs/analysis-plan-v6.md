# Analysis plan for the v6 experiments (FROZEN 2026-10-20)

**Status: frozen on 2026-10-20, before any TM run on v6.** All decisions below were taken by the maintainer on 2026-10-20 (the open points of the 2026-10-01 draft). From now on, any change goes into the amendment log at the bottom with the date, the reason, and whether it was made before or after seeing a result. Plain-language background: [concepts.md](concepts.md) (analysis plan, equivalence test, ambiguity ceiling, learning curve). What "finished" means for the project: [roadmap.md](roadmap.md), *Definition of done for v1.0*.

Two items are completed after the freeze but **before any TM run on v6**: the exact-refetch command (section 1) and the class-merge option (section 2). Both are code, neither can change a result by itself, and both are checked against the hashes below.

## Why a plan
So far many choices were made after looking at results (epoch rule, a verdict rule, thresholds, collection settings; all logged in [issues-and-fixes.md](issues-and-fixes.md) and [PROGRESS.md](../PROGRESS.md)). Each was reasonable, but together they make the results hard to trust as research: with enough small choices, a good-looking number can appear by chance. Writing the choices down before the v6 runs, and sticking to them, removes that doubt.

## 1. Data
- Dataset v6 as described in [dataset-card-v6.md](dataset-card-v6.md): `data/processed-v6/`, SHA-256 prefixes train `80a71a412bc8`, test `a20567d1250d`, hard `60b3f50662ef`, folds `5c7b8ed4a599`. Any rebuild that changes a hash is a new dataset version and invalidates the plan.
- **Reproducibility (decided):** a `collect from-manifest` command (refetch exact repo, commit, path and line range) is built and checked against these hashes **before** the experiments start.
- **Test set lock:** the v6 test set was scored once on 2026-09-30 (baselines, B-S5 check). From now on it is used once more, for the final head-to-head in section 6, and never for a choice. Every choice (features, TM settings, epochs, smallest model) uses CV on train only. The repeated-split score pools train and test, so it is never used for choices either.

## 2. Classes and views
- **Primary: 13 classes.** JavaScript and TypeScript are merged into one class `javascript_typescript` for training and scoring (decided 2026-10-20). C and C++ stay separate classes.
  - **Reason, stated openly:** the label is the file extension, and about a third of TypeScript windows contain no TypeScript syntax, so no model can separate those from JavaScript (the ambiguity ceiling). The decision was taken **after** seeing the v6 baseline results on this pair (JavaScript 0.70, TypeScript 0.75 F1); it is a change of the task definition, not a tuning step, and it is recorded as such.
  - Implementation: a class-merge option (`merge:` in the configs, `--merge`), recorded in every report; folds and test repos are unchanged.
- **Secondary, always reported beside the primary:**
  - the original 8 languages (`--languages core`, same folds), for comparison with v5; this view is not affected by the merge;
  - the 14-class score (JavaScript and TypeScript separate), so the cost of the merge stays visible;
  - the per-pair confusion and an approximate ambiguity ceiling for JavaScript/TypeScript and C/C++.

## 3. Targets
- **Accuracy:** macro-F1 >= 96% for the original 8 languages. For the 13 classes the target is relative: the TM is *equivalent* (section 6) to the best baseline. An absolute 96% is not claimed for 13 classes; the reported number is whatever is measured.
- **Size:** < 500 KB for the 13-class model. **Speed:** < 0.1 ms per snippet, measured on the C runtime (v1.0, milestone M6).

## 4. Features (CV only)
- Rerun the ablation config (`configs/ablations.yaml`, all studies) on v6 train with the 13 classes, models logistic regression and Naive Bayes.
- **Selection rule, fixed now:** for each setting take the mean CV macro-F1 of the two models. Among settings whose paired difference to the best setting is within one fold standard deviation of that difference, choose the one with the smallest M, then the fewest n-gram sizes, then delimiters off. That setting is frozen for every v6 model (baselines and TM).
- Slice result for reference (not used for the choice): class-balanced M=1000 was +0.008 +/- 0.009 over M=500, M=2000 no better.

## 5. TM settings and training (CV only)
- **Settings (decided):** 400 and 800 clauses per class (T=100, s=5, weighted clauses), 5 seeds (1-5).
- Epochs from `tm-curve` with the existing rule (`epoch_rule: max`, 11-epoch smoothed held-out curve, max 150 epochs), on v6 CV folds.
- The main TM setting is the one with the higher mean CV macro-F1 over seeds; the official model is the median-CV seed (`tm-select`), as in v5.
- **Clause sweep (small M4, CV only):** 50, 100, 200, 400 and 800 clauses per class, same protocol. Reported: the most accurate setting and the **smallest setting within 0.010 macro-F1 of it** (the margin of section 6), with the curve between them. No Optuna, no grid over s and T in v1.0.

## 6. Primary comparison
- **Question:** is the main TM setting equivalent to the best baseline on v6?
- **Best baseline:** the baseline with the highest CV macro-F1 on v6 with the frozen features (expected: logistic regression; chosen by the rule, not by name). Naive Bayes is reported as a second comparison.
- **Measure:** macro-F1 on the 10 repeated repo-level splits (paired with the baseline on the same splits), with the CV folds as a second view.
- **Test:** equivalence test (two one-sided tests, TOST) on the per-split differences with the Nadeau-Bengio corrected variance, as already used. **Margin (decided): +/- 0.010 macro-F1** (one point). Verdict: *equivalent* if the 90% interval lies inside +/- margin; *worse* if it lies entirely below -margin; *better* if entirely above +margin; otherwise *inconclusive*.
- **Uncertainty on the single test set:** bootstrap over test repositories (1,000 resamples, whole repos resampled within each language) for macro-F1 and each language's F1.
- **Multiple comparisons:** the primary family is {TM vs best baseline on the 13 classes, TM vs best baseline on the original 8}; Holm correction across it. Everything else is descriptive.
- **The same margin of one point is used everywhere** in the project: this test, the smallest model (section 5) and the "enough data" point of the data-efficiency study (section 10).

## 7. Secondary analyses (descriptive)
- Per-language F1 and confusion matrices; the 14-class and pair views (section 2).
- Hard examples: scored only on the 41 from test repos; HTML reported with and without the hard-example policy.
- Error analysis out of fold (`codelangtm errors`) on v6 train.
- **Short snippets (in v1.0, decided):** cut 1-, 3-, 5- and 10-line windows from the v6 *test* repos (no new collection) and report F1 by length for every model. The cutting rule (seeded, from the same files as the test windows, no overlap with training repos) is fixed when the code is written, before any result.
- **External baseline (Guesslang): after v1.0**, not in this plan.

## 8. Interpretability measures (in v1.0, decided)
Defined now and measured for the TM and for logistic regression on the same test windows:
- **Explanation size:** the number of clauses (TM) or features (logistic regression) needed to account for 90% of the winning language's vote margin over the runner-up.
- **Rule length:** literals per clause that fired, and per explanation.
- **Faithfulness check:** removing the explanation's features flips or keeps the prediction (share of flips).

## 9. Not allowed after freezing
Changing features, settings, epoch rule, margin, primary measure or the class framing because of a v6 TM result. If something has to change (a bug, a broken assumption), it goes into the amendment log, and the report shows both the planned and the amended result.

## 10. Data-efficiency study (after v1.0, milestone M8; pre-registered here)
Recorded now so the later study is not tuned to its results. Details may be refined before it starts, with an amendment-log entry.
- **Question:** how little training data does each model need, how fast does it train, and when do the TM's clauses and signatures settle?
- **Design:** learning curve. Train on 1, 2, 4, 8, 16, 32, 64 and all (112) training repos per language, sampling whole repos; the vocabulary is selected again from each subset; every size is scored on the same held-out CV folds; several random draws x 5 folds x 3 seeds per size. TM settings fixed at the main v1.0 setting. Models: TM, Naive Bayes, logistic regression. Classes: the 13 and the original 8.
- **Measures per size:** macro-F1 (overall and per language), fit time and epochs until the TM levels off, and for the TM clause statistics (precision, empty clauses, rule length, model size) and signature agreement with the full-data models.
- **"Enough data" point (n\*):** the smallest size whose CV macro-F1 is within **0.010** of the same model's full-data score and stays within it at every larger size.
- **Signatures formed:** the smallest size at which the subset model's signature n-grams agree with the full-data models as much as two full-data seeds agree with each other.
- **Data efficiency:** the TM's n\* against each baseline's, and F1 at small budgets (10 and 50 snippets per language).
- The test set is scored once at the end; every choice is on CV.

## Amendment log
| Date | Change | Reason | Before or after seeing a result |
| --- | --- | --- | --- |
| 2026-10-20 | JavaScript and TypeScript merged into one class for the primary 13-class analysis (section 2) | Label is the file extension and about a third of TypeScript windows carry no TypeScript syntax; no model can separate them | After seeing v6 baseline results on this pair (not after any TM result); 14-class score still reported |
