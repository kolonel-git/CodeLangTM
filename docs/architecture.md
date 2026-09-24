# Architecture

```
Code snippet
  -> Feature extraction & binarization   (char 2/3-grams + structural delimiters)
  -> Literal expansion                   (x_k and NOT x_k -> 2M literals)
  -> Multi-class TM engine (TMU)         (bitwise clause evaluation)
  -> Vote summation                      (positive clauses - negative clauses)
  -> Predicted language (argmax)
```

## Features
- Top-M character 2-/3-grams by document frequency (default M=500).
- Structural delimiters (`;`, `{`, `->`, `#`, indentation, comment tokens) always included.
- Literals: `L = [x_1..x_M, NOT x_1..NOT x_M]`.

## Model
`tmu.models.classification.vanilla_classifier.TMClassifier`, N_c clauses per class. Net score
`V_m(X) = sum C+_{m,j}(X) - sum C-_{m,j}(X)`; highest wins.
Feedback: Type I (pattern discovery, erasure with prob 1/s), Type II (false-alarm correction), gated by threshold T.

## Install notes
TMU builds a C extension. Python 3.12 is pinned. If the Windows build fails, use WSL2 or Docker.

## Targets
Macro-F1 >= 96% over 8 languages; < 0.1 ms/snippet; < 500 KB model.
