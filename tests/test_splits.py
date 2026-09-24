import random
from collections import Counter

import pytest

from codelangtm import LANGUAGES
from codelangtm.data import Snippet
from codelangtm.splits import (
    check_no_leakage,
    group_kfold,
    group_split,
    language_counts,
    repo_languages,
    separate_wild,
    stable_split,
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


# --- stable_split -------------------------------------------------------------------------


def repos_dataset(counts):
    """counts: {language: n_repos}; 2 snippets per repo."""
    return [
        Snippet(f"{lang} {r} {i}\n", lang, "github", f"{lang}-owner/repo{r}", "c",
                f"f{i}.txt", "MIT", 1, 1)
        for lang, n in counts.items() for r in range(n) for i in range(2)
    ]  # fmt: skip


def test_stable_split_balanced_per_language():
    data = repos_dataset({"python": 25, "go": 21, "rust": 7})
    split = stable_split(data)
    train, test = split.split(data)
    check_no_leakage(train, test)
    for lang, n in {"python": 25, "go": 21, "rust": 7}.items():
        n_test = len({s.repo for s in test if s.language == lang})
        assert n_test == round(n * 0.2)
        train_repos = {s.repo for s in train if s.language == lang}
        fold_sizes = Counter(split.fold_of[r] for r in train_repos)
        assert set(fold_sizes) == set(range(5))
        assert max(fold_sizes.values()) - min(fold_sizes.values()) <= 1


def test_stable_split_deterministic_and_salted():
    data = repos_dataset({"python": 25, "go": 25})
    assert stable_split(data) == stable_split(data)
    assert stable_split(data, salt="other").test_repos != stable_split(data).test_repos


def test_other_languages_unaffected_by_changes():
    base = repos_dataset({"python": 25, "go": 25})
    changed = repos_dataset({"python": 25, "go": 40})  # re-collected Go with more repos
    a, b = stable_split(base), stable_split(changed)
    py = {s.repo for s in base if s.language == "python"}
    assert a.test_repos & py == b.test_repos & py
    assert {r: a.fold_of[r] for r in py if r in a.fold_of} == {
        r: b.fold_of[r] for r in py if r in b.fold_of
    }


def test_adding_or_removing_one_repo_moves_at_most_one_test_repo():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(5, 40)
        data = repos_dataset({"python": n})
        before = stable_split(data)
        if rng.random() < 0.5:
            after_data = repos_dataset({"python": n + 1})  # add repo n
        else:
            drop = f"python-owner/repo{rng.randrange(n)}"
            after_data = [s for s in data if s.repo != drop]
        after = stable_split(after_data)
        common = {s.repo for s in data} & {s.repo for s in after_data}
        moved = {r for r in common if (r in before.test_repos) != (r in after.test_repos)}
        assert len(moved) <= 1


def test_single_repo_language_goes_to_train():
    data = repos_dataset({"python": 10, "sql": 1})
    split = stable_split(data)
    assert "sql-owner/repo0" not in split.test_repos
    assert "sql-owner/repo0" in split.fold_of


def test_repo_language_is_majority():
    s = [
        Snippet("a\n", "python", "github", "o/mixed", "c", "a.py", "MIT", 1, 1),
        Snippet("b\n", "python", "github", "o/mixed", "c", "b.py", "MIT", 1, 1),
        Snippet("c\n", "sql", "github", "o/mixed", "c", "c.sql", "MIT", 1, 1),
    ]
    assert repo_languages(s) == {"o/mixed": "python"}


@pytest.mark.parametrize(("test_size", "n_folds"), [(0, 5), (0.5, 5), (0.2, 1)])
def test_stable_split_bad_args(test_size, n_folds):
    with pytest.raises(ValueError):
        stable_split(repos_dataset({"python": 5}), test_size, n_folds)
