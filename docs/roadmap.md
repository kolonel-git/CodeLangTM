# Roadmap

## Phase 1 — Dataset & pipeline
- [ ] Collect ~1,000 real-world snippets (20-50 lines), 8 languages: Python, C++, Java, JS, Rust, Go, SQL, HTML
- [ ] Record sources/licenses in `docs/data-sources.md`
- [ ] Binarizer (M=500) — initial version in `features.py`
- [ ] 80/20 stratified split

## Phase 2 — Core training
- [ ] TMClassifier baseline (N_c=100, T=30, s=3.5)
- [ ] Naive Bayes + Decision Tree baselines
- [ ] Accuracy + throughput benchmarks

## Phase 3 — Optimization & rules
- [ ] Grid search s, T
- [ ] Drop clause + literal budgeting
- [ ] Render AND-rules

## Phase 4 — Deployment
- [ ] Pure C export
- [ ] CLI tool; IDE plugin

## Future
Relational TM (AST), Convolutional TM (2D layout), FPGA/ASIC (MATADOR), FedTMOS, Sparse TM for vulnerability detection.
