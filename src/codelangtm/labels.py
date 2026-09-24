"""Label sanity checks: does a snippet's language label agree with its file and content?

Labels come from file extensions, which lie: `.h` is C or C++, `.js` can be minified or
generated, `.html` can be a template, `.txt` from the wild has no extension at all. A wrong
label teaches the model the wrong rule, so suspicious snippets are dropped and counted.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

from .data import Snippet
from .syntax import is_cut

EXTENSION_LANGUAGE = {
    ".py": "python",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".java": "java",
    ".js": "javascript", ".mjs": "javascript", ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".sql": "sql",
    ".html": "html", ".htm": "html",
    # stretch set
    ".c": "c",
    ".cs": "csharp",
    ".ts": "typescript", ".tsx": "typescript",
    ".kt": "kotlin",
    ".php": "php",
    ".rb": "ruby",
}  # fmt: skip

_CPP_MARKERS = re.compile(r"\bclass\s+\w+|\bnamespace\b|\btemplate\s*<|std::|#include\s*<iostream>")

# Strong evidence the text is a *different* language than labelled.
_RED_FLAGS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(#include\b|package\s+[\w.]+;|public\s+class\b|<\?php|<html)", re.M),
    "javascript": re.compile(
        r"^\s*(#include\b|package\s+[\w.]+;|<\?php)|:\s*(string|number|boolean)\b\s*[,;)=]", re.M
    ),
    "cpp": re.compile(r"^\s*(def\s+\w+.*:\s*$|import\s+java\.|package\s+[\w.]+;|fn\s+\w+\()", re.M),
    "java": re.compile(r"^\s*(#include\b|def\s+\w+.*:\s*$|fn\s+\w+\(|func\s+\w+\()", re.M),
    "rust": re.compile(r"^\s*(#include\b|public\s+class\b|def\s+\w+.*:\s*$)", re.M),
    "go": re.compile(r"^\s*(#include\b|public\s+class\b|def\s+\w+.*:\s*$)", re.M),
}

# A real HTML tag (`<div`, `</p`, `<my-el`) or `<!` (comment/doctype). The name must end in
# space, `>`, `/` or end of line (multi-line tags), and must not follow an identifier or `)`/`]`,
# so comparisons like `i < elements` or `a<b` do not count as tags.
HTML_TAG = re.compile(r"(?<![\w)\]])</?[a-zA-Z][\w-]*(?:[\s>/]|$)|<!", re.M)

# HTML windows that are mostly inline <script>/<style> code are really JavaScript/CSS.
MIN_MARKUP_SHARE = 0.20

# Content that must appear for the label to be believable.
_REQUIRED: dict[str, re.Pattern[str]] = {
    "html": HTML_TAG,
    "sql": re.compile(
        r"\b(select|insert|update|delete|create|alter|drop|with|from|where)\b", re.I
    ),
}

MAX_LINE_LEN = 500  # longer lines mean minified or data blobs
MAX_NON_ASCII = 0.30

# Template engines (Jinja/dbt/Liquid/Vue, ERB/EJS) wrapped around SQL or HTML. Mostly-template
# windows teach "{% means SQL" shortcuts. Other languages use `{{` legitimately (format-string
# escapes, JSX style props, nested initialisers), so they are not checked.
_TEMPLATE_MARK = re.compile(r"\{%|%\}|\{\{|\}\}|<%|%>")
TEMPLATE_LANGUAGES = frozenset({"sql", "html"})
MAX_TEMPLATE_FRAC = 0.30


def template_fraction(text: str, front_matter: bool = False) -> float:
    """Share of non-blank lines with template markers (and `---` front matter if enabled)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(bool(_TEMPLATE_MARK.search(ln)) or (front_matter and ln == "---") for ln in lines)
    return hits / len(lines)


def markup_share(text: str) -> float:
    """Share of non-blank lines containing an HTML tag."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return sum(bool(HTML_TAG.search(ln)) for ln in lines) / len(lines) if lines else 0.0


@dataclass(frozen=True)
class LabelCheck:
    ok: bool
    reasons: tuple[str, ...] = ()


def language_from_path(path: str, text: str = "") -> str | None:
    """Language implied by the file extension; `.h` is decided from content."""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".h":
        return "cpp" if _CPP_MARKERS.search(text) else "c"
    return EXTENSION_LANGUAGE.get(suffix)


def check_label(s: Snippet) -> LabelCheck:
    reasons: list[str] = []
    text = s.text

    if "\x00" in text:
        reasons.append("binary: contains NUL")
    if max((len(ln) for ln in text.splitlines()), default=0) > MAX_LINE_LEN:
        reasons.append("minified or data: very long line")
    if text and sum(ord(c) > 127 for c in text) / len(text) > MAX_NON_ASCII:
        reasons.append("mostly non-ASCII")

    if s.source != "wild":  # wild snippets have no meaningful file extension
        implied = language_from_path(s.path, text)
        if implied != s.language:
            reasons.append(f"extension implies {implied!r}, labelled {s.language!r}")

    flag = _RED_FLAGS.get(s.language)
    if flag and flag.search(text):
        reasons.append("content looks like another language")
    required = _REQUIRED.get(s.language)
    if required and not required.search(text):
        reasons.append("expected markers missing")
    elif s.language == "html" and markup_share(text) < MIN_MARKUP_SHARE:
        reasons.append("mostly embedded script/style")
    if is_cut(text, s.language):
        reasons.append("cut comment or string")
    if s.language in TEMPLATE_LANGUAGES:
        if template_fraction(text, front_matter=s.language == "html") > MAX_TEMPLATE_FRAC:
            reasons.append("template-heavy")

    return LabelCheck(not reasons, tuple(reasons))


def filter_labels(snippets: Iterable[Snippet]) -> tuple[list[Snippet], Counter[str]]:
    """Keep snippets whose labels pass; count why the rest were dropped."""
    kept: list[Snippet] = []
    dropped: Counter[str] = Counter()
    for s in snippets:
        result = check_label(s)
        if result.ok:
            kept.append(s)
        else:
            dropped.update(result.reasons)
    return kept, dropped
