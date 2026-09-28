"""Documentation checks: every relative link and image in the project's markdown resolves."""

import re
from pathlib import Path

import pytest

from codelangtm import figures as fg

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(
    [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md"), *(ROOT / ".github").rglob("*.md")]
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.S)


def slug(heading: str) -> str:
    """GitHub-style anchor: lowercase, drop punctuation except '-', spaces to '-'."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return {slug(m) for m in re.findall(r"^#+\s+(.+)$", text, re.M)}


def links(path: Path) -> list[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return [t for t in LINK.findall(text) if not re.match(r"[a-z]+:", t)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_resolve(doc):
    broken = []
    for target in links(doc):
        file_part, _, anchor = target.partition("#")
        dest = (doc.parent / file_part).resolve() if file_part else doc
        if not dest.exists():
            broken.append(target)
        elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            broken.append(target)
    assert not broken, f"{doc.name}: broken links {broken}"


def test_report_embeds_every_figure():
    embedded = {Path(t).stem for t in links(ROOT / "docs" / "report.md") if t.endswith(".png")}
    assert embedded == (set(fg.RESULTS_FIGURES) | set(fg.ABLATION_FIGURES)
                        | set(fg.CURVE_FIGURES) | set(fg.TM_RESULT_FIGURES)
                        | set(fg.RESOURCE_FIGURES) | set(fg.CLAUSE_FIGURES))
    for name in embedded:
        assert (ROOT / "docs" / "figures" / f"{name}.png").exists()


def test_slug():
    assert slug("4. Baselines: the bar for the TM") == "4-baselines-the-bar-for-the-tm"
    assert slug("M6. Frequency-ranked vocabulary") == "m6-frequency-ranked-vocabulary"
