import pytest

from codelangtm.data import Snippet
from codelangtm.dedup import dedup, jaccard, normalize, shingles


def snip(text, repo="o/r", path="a.py"):
    n = text.count("\n") + 1
    return Snippet(text, "python", "github", repo, "c", path, "MIT", 1, n)


def body(prefix, n=40):
    return "\n".join(f"{prefix}_{i} = call_{prefix}({i}, '{prefix}{i}')" for i in range(n)) + "\n"


def test_normalize_ignores_whitespace():
    assert normalize("a  b\n\n c") == normalize("a b c")


def test_jaccard_basics():
    a = shingles(body("alpha"))
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles(body("beta"))) < 0.2


def test_exact_duplicates_removed_first_kept():
    a = snip(body("alpha"), path="first.py")
    b = snip(body("alpha").replace(" = ", "  =  "), path="second.py")  # whitespace only
    kept, stats = dedup([a, b])
    assert kept == [a]
    assert (stats.exact_removed, stats.near_removed, stats.kept) == (1, 0, 1)


def test_near_duplicate_removed():
    base = body("alpha")
    edited = base.replace("alpha_7 = call_alpha(7, 'alpha7')", "alpha_7 = changed()")
    kept, stats = dedup([snip(base), snip(edited)])
    assert len(kept) == 1
    assert stats.near_removed == 1


def test_distinct_kept():
    items = [snip(body(p)) for p in ("alpha", "beta", "gamma", "delta")]
    kept, stats = dedup(items)
    assert kept == items
    assert stats.kept == 4


def test_deterministic():
    items = [snip(body("alpha")), snip(body("alpha").replace("7", "8")), snip(body("beta"))]
    assert dedup(items)[0] == dedup(items)[0]


def test_bad_threshold():
    with pytest.raises(ValueError):
        dedup([], threshold=0)
