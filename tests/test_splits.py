import pytest

from codelangtm import LANGUAGES
from codelangtm.data import Snippet
from codelangtm.splits import (
    check_no_leakage,
    group_kfold,
    group_split,
    language_counts,
    separate_wild,
)


def make_dataset(repos_per_lang=10, per_repo=5):
    out = []
    for lang in LANGUAGES:
        for r in range(repos_per_lang):
            for i in range(per_repo):
                out.append(
                    Snippet(f"{lang} {r} {i}\n", lang, "github", f"{lang}-owner/repo{r}",
                            "c", f"f{i}.txt", "MIT", 1, 1)
                )  # fmt: skip
    return out


DATA = make_dataset()


def test_no_repo_in_both_sets():
    train, test = group_split(DATA)
    check_no_leakage(train, test)
    assert len(train) + len(test) == len(DATA)


def test_test_share_and_stratification():
    train, test = group_split(DATA, test_size=0.2)
    assert 0.15 <= len(test) / len(DATA) <= 0.25
    tc, trc = language_counts(test), language_counts(train)
    assert set(tc) == set(trc) == set(LANGUAGES)  # every language on both sides


def test_deterministic_and_seed_changes_split():
    a = group_split(DATA, seed=1)
    assert a == group_split(DATA, seed=1)
    assert a[1] != group_split(DATA, seed=2)[1]


def test_kfold_tests_each_snippet_once_without_leakage():
    tested = []
    for train, test in group_kfold(DATA, n_splits=5):
        check_no_leakage(train, test)
        tested += test
    assert len(tested) == len(DATA)
    assert len(set(tested)) == len(DATA)


def test_leakage_detected():
    with pytest.raises(ValueError, match="leakage"):
        check_no_leakage(DATA[:5], DATA[:5])


def test_wild_separated_and_repo_overlap_caught():
    wild = Snippet("q\n", "python", "wild", "so/answer1", "c", "a.txt", "CC-BY-SA-4.0", 1, 1)
    main, w = separate_wild([*DATA, wild])
    assert w == [wild] and len(main) == len(DATA)
    same_repo = Snippet("q\n", "python", "wild", DATA[0].repo, "c", "a.txt", "MIT", 1, 1)
    with pytest.raises(ValueError, match="leakage"):
        check_no_leakage(DATA, [same_repo])


def test_too_few_repos():
    few = make_dataset(repos_per_lang=1)[:5]
    with pytest.raises(ValueError, match="distinct repos"):
        group_split(few)


@pytest.mark.parametrize("bad", [0, 0.5, 0.9])
def test_bad_test_size(bad):
    with pytest.raises(ValueError):
        group_split(DATA, test_size=bad)
