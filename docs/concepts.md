# Concepts guide

Plain-language explanations of the ideas used in this project, for readers who know some programming and basic statistics but not this area. The [report](report.md) uses these terms, and the numbers it quotes come from [results.md](results.md) and [ablations.md](ablations.md). This guide grows as the project introduces new ideas.

**Contents:** [The task](#the-task) · [Turning code into features](#turning-code-into-features) · [The Tsetlin Machine](#the-tsetlin-machine) · [Measuring quality](#measuring-quality) · [Fair evaluation](#fair-evaluation) · [Experiments and noise](#experiments-and-noise) · [Speed and size](#speed-and-size)

## The task

- **Language identification:** given a piece of source code, name its programming language. There are 8 classes here: Python, C++, Java, JavaScript, Rust, Go, SQL, HTML.
- **Snippet (window):** one contiguous block of 20-50 lines cut from a real file on GitHub. Every snippet keeps its repository, commit, path and license.
- **Classifier:** a program that learns from labelled examples (snippet plus its correct language) and then predicts labels for new snippets.
- **Interpretable model:** a model whose decisions a person can read and check. The project's claim is that a Tsetlin Machine can match standard classifiers while staying readable.

## Turning code into features

A classifier cannot read text directly; it needs numbers. Here every snippet becomes a row of 0s and 1s.

- **Character n-gram:** a string of n consecutive characters. The 3-grams of `def f` are `def`, `ef `, `f f`. Short n-grams capture syntax: `::` (C++, Rust), `:=` (Go), `</` (HTML).
- **Binary feature:** "does this snippet contain this n-gram?", answered 1 (yes) or 0 (no). Counts are ignored.
- **Vocabulary (M):** the list of n-grams used as features. There are hundreds of thousands of candidates, so only M are kept. This project uses M = 500.
- **Feature selection:** how the M are chosen.
  - *Frequency:* keep the most common n-grams overall. Simple, but it wastes slots on strings common in every language, like `ing` in English comments.
  - *chi2:* keep the n-grams most statistically associated with the language labels.
  - *class_balanced:* languages take turns picking the n-gram that is most typical of them and least typical of the others.

  The last two are **label-aware**: they use the correct answers to pick features. That is allowed only on training data (see [leakage](#fair-evaluation)).
- **Binarizer:** the component that learns the vocabulary from training snippets, then turns any snippet into its 0/1 row.
- **Literal:** a feature or its negation. With M features there are 2M literals: `has("::")` and `NOT has("::")`, and so on.

## The Tsetlin Machine

- **Clause:** an AND of a few literals, for example `has("def ") AND has(":") AND NOT has(";")`. A clause *fires* (outputs 1) when all its literals are true for a snippet.
- **Empty clause:** a clause that has not included any literal. It never fires at prediction time.
- **Votes:** each language owns a set of clauses. Clauses with positive weight vote *for* the language when they fire; clauses with negative weight vote *against* it.
- **Weight:** how many votes a clause casts. Training adjusts the weights, so reliable clauses count more.
- **Class sum:** the total votes for a language (the sum of the weights of its clauses that fired). The language with the highest class sum is the prediction. Because the rules are plain logic, a prediction can be traced to exactly the clauses that fired.
- **Tsetlin automaton:** a tiny learning unit, one per literal per clause, that decides whether to *include* that literal in the clause. It moves one step towards "include" or "exclude" at a time, based on feedback.
- **Feedback:** after each training example, clauses get nudged:
  - *Type I* feedback strengthens clauses that fire on the right language and prunes literals that don't help;
  - *Type II* feedback adds literals to clauses that fire on the wrong language, so they stop firing there.
- **N_c (clauses per class):** how many clauses each language gets. More clauses let the model capture more patterns, at the cost of size and readability.
- **T (threshold):** roughly the number of votes after which a language counts as "confident enough". Once its class sum reaches T, feedback for that example fades. More clauses need a larger T.
- **s (specificity):** controls how easily literals are dropped. A larger s gives longer, more specific clauses.
- **Epoch:** one pass over all training snippets. The TM trains for many epochs.
- **Model file:** the trained TM saved as one JSON file: the feature vocabulary, every clause (its included literals and weight) and metadata. Anything that can read JSON can use it to classify code, with no machine-learning library.
- **Rule:** a clause written out as text, for example `rust = has("fn ") AND NOT has("\t}")`. Quotes show spaces; `\n` is a line break and `\t` a tab.
- **Coverage, false fires, precision (of a clause):** *coverage* is the share of its own language's training snippets the clause fires on; *false fires* is the share of other languages' snippets it fires on; *precision* is, of all snippets it fires on, the share in its own language. A good "for" clause has high coverage and high precision.
- **Signature n-gram:** an n-gram that a language's "for" clauses use much more often than other languages' "for" clauses do (C++ `::`, Go `:=`). Measured by *lift*: how many times more often it is used here than elsewhere.
- **Jaccard similarity:** overlap between two sets, measured as the size of their intersection divided by the size of their union. It is 1 when the sets are identical and 0 when they share nothing. Used here to compare the n-grams two languages rely on, and a model's clauses during training with its final clauses.
- **Specialist (narrow) clause:** a very long clause, mostly `NOT has(...)` parts, that fires on only a handful of training snippets. It is a way of memorising unusual examples, and costs readability.
- **Explanation of a prediction:** the list of clauses that fired on a snippet, with their weights. Each language's votes are exactly the sum of those weights, so the explanation is complete, not an approximation.
- **Formation replay:** retraining a model with the same seed, one epoch at a time, and saving a snapshot after each epoch to watch how its rules appeared. With the same seed, training is deterministic, so the replay ends at exactly the saved model.
- **Seed:** the starting value of the random number generator. Training uses randomness, so the same seed reproduces the same model, and different seeds show how much results depend on luck. (TMU needs a seed of at least 1: with 0 its random number generator gets stuck.)

## Measuring quality

- **Precision (per language):** of the snippets predicted as Python, the share that really are Python.
- **Recall (per language):** of the real Python snippets, the share predicted as Python.
- **F1:** the harmonic mean of precision and recall. It is 1.0 only when both are perfect.
- **Macro-F1:** F1 computed per language, then averaged. Every language counts equally, so the small SQL class matters as much as the others. This is the project's main metric (target ≥ 0.96).
- **Accuracy:** the share of all snippets predicted correctly. Easier to read, but dominated by the large classes.
- **Confusion matrix:** a table of true language (rows) by predicted language (columns). Off-diagonal cells are mistakes and show which pairs get confused, such as Rust and C++.

## Fair evaluation

- **Training set / test set:** the model learns from the training set; the test set is kept aside and used only to report the final score.
- **Leakage:** any way information from the evaluation data sneaks into training, making scores look better than they are. Examples: the same repository in train and test (shared names and style), or choosing features using held-out data.
- **Group (repository) split:** all snippets from one repository go to the same side, so the model is always tested on projects it has never seen.
- **Cross-validation (CV):** the training set is cut into 5 folds (again by repository). Each fold is held out once while the model trains on the other four. The CV score is the mean ± standard deviation over the 5 folds. All design choices (features, settings, number of epochs) are made with CV, never with the test set.
- **Held-out fold:** the fold not used for training in one CV round.
- **Repeated test splits:** the project also re-splits the data 10 more ways and reports the mean ± std of the test score. With only about 39 test repositories, one test score depends a lot on which repositories landed in test.
- **Overfitting:** the model fits the training data (training F1 near 1.0) better than it generalises to new data. The gap between training and held-out scores measures it.

## Experiments and noise

- **Estimator / pipeline:** scikit-learn's common interface: every model has `fit` (learn) and `predict`. A *pipeline* chains steps, here binarizer → classifier, so the whole chain is trained and evaluated as one model. Because the TM follows the same interface, the same evaluation code runs for every model.
- **Baseline:** a standard model used as a yardstick. Here: Naive Bayes, logistic regression, linear SVM, decision tree and random forest, all on exactly the TM's features.
- **Ablation:** change one design option, keep everything else fixed, and measure the effect.
- **Paired difference (Δ):** compare two settings on the *same* folds and look at the per-fold differences. Some folds are harder than others; pairing cancels that out.
- **Noise band:** differences smaller than their own spread (|mean Δ| ≤ std) are treated as noise, not as real effects.
- **Training curve:** a score measured after every epoch, plotted against the epoch number. The held-out curve shows when the model stops improving; the gap to the training curve shows overfitting.
- **Moving average (smoothing):** replace each point by the average of its neighbours (here 11 epochs centred on it), so single lucky or unlucky epochs stop dominating.
- **Plateau rule:** after smoothing, choose the first epoch whose score is within a small tolerance (0.005) of the best smoothed score: the point where the curve has levelled off. Training longer costs time and gains almost nothing. The alternative, the *peak* of the smoothed curve, gets the last bit of accuracy at the cost of more training; this project uses the peak (decided after seeing the CV curves, before any test-set look) and reports the plateau beside it.
- **Selection bias ("best epoch"):** picking the best of many noisy numbers gives an optimistic result, because the maximum is partly luck. That is why the project reports averages (for example over the last 20 epochs) and chooses settings on CV only.
- **Significance test:** a statistical check of whether a difference (for example TM vs Naive Bayes) is larger than chance variation would produce. Used for the headline comparisons, over several seeds and the same folds.
- **p-value:** if there were truly no difference, how often would chance alone produce a difference at least this large? Below 0.05 is the usual line for "unlikely to be chance". A large p does not prove the models are equal; it means the data cannot tell them apart.
- **95% interval (confidence interval):** the range of true differences that fits the data. If it contains 0, "no difference" is plausible; a narrow interval around 0 is evidence that any difference is small.
- **Correlation (Pearson):** how closely two lists of numbers rise and fall together, from -1 (opposite) through 0 (unrelated) to 1 (in step). Used to compare the signature scores of every n-gram between models trained with different seeds, which is fairer than comparing two top-10 lists whose edges are decided by near-ties.
- **Out-of-fold prediction:** predict each training snippet with the model trained on the *other* CV folds, which never saw it. Every snippet gets an honest prediction without touching the test set, so there are many more errors to study than the test set alone gives.
- **Hard vs unlucky error:** a snippet that every seed gets wrong is *hard* (something about the snippet misleads the model); one that only some seeds get wrong is *unlucky* (the random start decided it).
- **Error sink:** a language that collects a large share of a model's mistakes, the class the model falls back on when unsure.
- **McNemar test:** compares two models on the *same* snippets. It ignores snippets both get right or both get wrong, and asks whether "only model A wrong" happens significantly more often than "only model B wrong".
- **Mann-Whitney U test:** checks whether values in one group tend to be larger than in another (for example, snippet length for errors vs correct predictions), without assuming the values follow a bell curve.
- **Spearman correlation:** like Pearson correlation, but computed on ranks, so it asks "does more of one go with more of the other?" without assuming a straight-line relation.
- **Verdict rule:** the pass/fail criterion for a hypothesis, written down before looking at the result, so the data cannot quietly move the goalposts. Any rule changed afterwards is reported as changed.
- **Embedded code:** one language inside another's file, like JavaScript inside an HTML `<script>` block. Whether such a snippet counts as HTML or JavaScript is a labelling decision, not something a model can get right by itself.
- **Corrected resampled t-test (Nadeau & Bengio):** the test used here. A normal paired t-test assumes the repeated splits are independent, but they share most of their training data, which makes it too confident. The correction widens the uncertainty to account for that overlap.

## Speed and size

- **Latency:** time to classify one snippet, end to end: building its features plus prediction. Target < 0.1 ms.
- **Throughput:** snippets per second (1000 / latency in ms).
- **Model size:** bytes needed to store the trained model. For the TM, each clause's included literals can be stored as bits (*dense*, one bit per literal per clause) or as a list of the included literals only (*sparse*, much smaller when clauses are short). Target < 500 KB.
- **Peak memory:** the most memory used at once while training or predicting. *Process-level* memory (the operating system's view: working set on Windows, RSS on Linux) counts everything the process holds, including memory allocated by C libraries; Python's `tracemalloc` only sees Python and NumPy objects.
- **Median and p95 latency:** the median is the typical time per snippet; p95 is the time that 95% of snippets beat, which shows how slow the slow cases get.
- **Fresh process:** each measurement runs in a new Python interpreter that does one job, so imports, caches and earlier work cannot distort it.
- **Deterministic output:** the same input always gives the same result, byte for byte. This matters for reproducible numbers and for figures that only change in git when the data changes.
