"""Dataset audit: per-language stats, quality flags, and Markdown samples for manual review."""

from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from . import LANGUAGES
from .data import Snippet, load_snippets
from .windows import _is_comment

SPLITS = ("train", "test", "wild")
ROLES = ("source", "test", "example", "migration", "docs", "wild")

MAX_REPO_SHARE = 0.10  # one repo's style should not define a language
MAX_TEST_SHARE = 0.40  # otherwise test idioms (asserts, @Test) become the signal
MAX_COMMENT_RATIO = 0.40  # windows should be mostly code, not prose
MIN_BALANCE = 0.60  # smallest language vs largest

_TEST_DIR = re.compile(r"(^|/)(tests?|__tests__|specs?|testing|testdata)/", re.I)
_TEST_NAME = re.compile(r"^(test_.*|.*_test\.\w+|.*Tests?\.\w+|.*\.(test|spec)\.\w+)$")
_ROLE_DIRS = {
    "example": re.compile(r"(^|/)(examples?|demos?|samples?|tutorials?|playground)/", re.I),
    "migration": re.compile(r"(^|/)(migrations?|migrate)/", re.I),
    "docs": re.compile(r"(^|/)(docs?|documentation)/", re.I),
}


def file_role(path: str, source: str = "github") -> str:
    if source == "wild":
        return "wild"
    if _TEST_DIR.search(path) or _TEST_NAME.match(PurePosixPath(path).name):
        return "test"
    for role, pattern in _ROLE_DIRS.items():
        if pattern.search(path):
            return role
    return "source"


def comment_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return sum(_is_comment(ln) for ln in lines) / len(lines) if lines else 0.0


def non_ascii_share(text: str) -> float:
    return sum(ord(c) > 127 for c in text) / len(text) if text else 0.0


@dataclass
class LanguageStats:
    language: str
    snippets: int = 0
    splits: Counter[str] = field(default_factory=Counter)
    repos: int = 0
    top_repo: str = ""
    top_repo_share: float = 0.0
    median_lines: float = 0.0
    median_chars: float = 0.0
    median_comment_ratio: float = 0.0
    mean_non_ascii: float = 0.0
    roles: Counter[str] = field(default_factory=Counter)
    licenses: Counter[str] = field(default_factory=Counter)


def load_dataset(data_dir: str | Path) -> dict[str, list[Snippet]]:
    data_dir = Path(data_dir)
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"{data_dir / 'train.jsonl'} not found; run `codelangtm data build`"
        )
    return {
        split: list(load_snippets(data_dir / f"{split}.jsonl"))
        for split in SPLITS
        if (data_dir / f"{split}.jsonl").exists()
    }


def language_stats(
    dataset: dict[str, list[Snippet]], languages: Sequence[str] = LANGUAGES
) -> dict[str, LanguageStats]:
    out: dict[str, LanguageStats] = {}
    for lang in languages:
        items = [(split, s) for split, part in dataset.items() for s in part if s.language == lang]
        if not items:
            out[lang] = LanguageStats(lang)
            continue
        snippets = [s for _, s in items]
        repos = Counter(s.repo for s in snippets)
        top_repo, top_n = repos.most_common(1)[0]
        out[lang] = LanguageStats(
            language=lang,
            snippets=len(snippets),
            splits=Counter(split for split, _ in items),
            repos=len(repos),
            top_repo=top_repo,
            top_repo_share=top_n / len(snippets),
            median_lines=statistics.median(s.n_lines for s in snippets),
            median_chars=statistics.median(len(s.text) for s in snippets),
            median_comment_ratio=statistics.median(comment_ratio(s.text) for s in snippets),
            mean_non_ascii=statistics.fmean(non_ascii_share(s.text) for s in snippets),
            roles=Counter(file_role(s.path, s.source) for s in snippets),
            licenses=Counter(s.license for s in snippets),
        )
    return out


def audit_flags(stats: dict[str, LanguageStats]) -> list[str]:
    flags: list[str] = []
    largest = max((st.snippets for st in stats.values()), default=0)
    for lang, st in stats.items():
        if st.snippets == 0:
            flags.append(f"{lang}: no data")
            continue
        if st.top_repo_share > MAX_REPO_SHARE:
            flags.append(
                f"{lang}: {st.top_repo} supplies {st.top_repo_share:.0%} of snippets "
                f"(limit {MAX_REPO_SHARE:.0%})"
            )
        test_share = st.roles["test"] / st.snippets
        if test_share > MAX_TEST_SHARE:
            flags.append(f"{lang}: {test_share:.0%} of snippets come from test files")
        if st.median_comment_ratio > MAX_COMMENT_RATIO:
            flags.append(f"{lang}: median window is {st.median_comment_ratio:.0%} comments")
        if st.snippets < MIN_BALANCE * largest:
            flags.append(
                f"{lang}: {st.snippets} snippets vs {largest} for the largest language "
                "(class imbalance)"
            )
    return flags


def pick_samples(
    dataset: dict[str, list[Snippet]],
    n: int,
    seed: int = 0,
    languages: Sequence[str] = LANGUAGES,
) -> dict[str, list[tuple[str, Snippet]]]:
    rng = random.Random(seed)
    out = {}
    for lang in languages:
        items = [(split, s) for split, part in dataset.items() for s in part if s.language == lang]
        out[lang] = rng.sample(items, min(n, len(items)))
    return out


def permalink(s: Snippet) -> str | None:
    if s.source != "github":
        return None
    return (
        f"https://github.com/{s.repo}/blob/{s.commit}/{quote(s.path)}"
        f"#L{s.start_line}-L{s.end_line}"
    )


def _fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def summary_table(stats: dict[str, LanguageStats]) -> str:
    rows = [
        f"{'language':<12}{'snips':>6}{'repos':>6}{'top':>6}{'lines':>6}"
        f"{'cmt':>6}{'test':>6}"
    ]
    for st in stats.values():
        test = st.roles["test"] / st.snippets if st.snippets else 0.0
        rows.append(
            f"{st.language:<12}{st.snippets:>6}{st.repos:>6}{st.top_repo_share:>6.0%}"
            f"{st.median_lines:>6.0f}{st.median_comment_ratio:>6.0%}{test:>6.0%}"
        )
    return "\n".join(rows)


def render_markdown(
    source: str | Path,
    stats: dict[str, LanguageStats],
    flags: list[str],
    samples: dict[str, list[tuple[str, Snippet]]],
    seed: int,
    generated: str | None = None,
) -> str:
    total = sum(st.snippets for st in stats.values())
    md = [
        "# Dataset audit",
        "",
        f"Generated {generated or 'n/a'} from `{source}` ({total} snippets), sample seed {seed}. "
        "Regenerating overwrites this file, so note findings elsewhere.",
        "",
        "Tick each sample that is correctly labelled, real hand-written code, and not mostly "
        "template/embedded/other-language content.",
        "",
        "## Summary",
        "",
        "| language | snippets | train | test | wild | repos | top repo share | median lines "
        "| median comment ratio | non-ASCII |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for st in stats.values():
        md.append(
            f"| {st.language} | {st.snippets} | {st.splits['train']} | {st.splits['test']} "
            f"| {st.splits['wild']} | {st.repos} | {st.top_repo_share:.0%} | "
            f"{st.median_lines:.0f} | {st.median_comment_ratio:.0%} | {st.mean_non_ascii:.1%} |"
        )
    md += ["", "## Flags", ""]
    md += [f"- {f}" for f in flags] or ["- none"]

    md += ["", "## File roles", "", "| language | " + " | ".join(ROLES) + " |"]
    md.append("| --- |" + " --- |" * len(ROLES))
    for st in stats.values():
        md.append(f"| {st.language} | " + " | ".join(str(st.roles[r]) for r in ROLES) + " |")

    md += ["", "## Licenses", ""]
    for st in stats.values():
        mix = ", ".join(f"{lic} {n}" for lic, n in st.licenses.most_common()) or "-"
        md.append(f"- **{st.language}**: {mix}")

    md += ["", "## Samples"]
    for lang, items in samples.items():
        md += ["", f"### {lang}"]
        for i, (split, s) in enumerate(items, 1):
            link = permalink(s)
            where = f"[`{s.repo}`]({link})" if link else f"`{s.repo}`"
            fence = _fence(s.text)
            md += [
                "",
                f"- [ ] OK — **{i}.** {where} `{s.path}` L{s.start_line}-{s.end_line} "
                f"({split}, {file_role(s.path, s.source)})",
                "",
                f"{fence}{lang}",
                s.text.rstrip("\n"),
                fence,
            ]
    return "\n".join(md) + "\n"


def run_audit(
    data_dir: str | Path,
    out: str | Path,
    n_samples: int = 10,
    seed: int = 0,
    languages: Sequence[str] = LANGUAGES,
) -> tuple[dict[str, LanguageStats], list[str]]:
    dataset = load_dataset(data_dir)
    stats = language_stats(dataset, languages)
    flags = audit_flags(stats)
    samples = pick_samples(dataset, n_samples, seed, languages)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().isoformat(sep=" ", timespec="seconds")
    out.write_text(
        render_markdown(data_dir, stats, flags, samples, seed, generated), encoding="utf-8"
    )
    return stats, flags
