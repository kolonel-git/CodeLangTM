"""Minimal lexer: is a line inside a block comment or multi-line string?

Used to keep windows from starting or ending mid-comment, which turns prose into "code" and
code into "comment". Deliberately small: it tracks comments and strings, nothing else.
Unknown languages return None (no constraint).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Syntax:
    line_comments: tuple[str, ...] = ()
    block_comments: tuple[tuple[str, str], ...] = ()
    strings: tuple[str, ...] = ()  # single-line quotes, backslash escapes
    multiline_strings: tuple[str, ...] = ()  # may span lines; checked before `strings`


_C_BLOCK = (("/*", "*/"),)

SYNTAX: dict[str, Syntax] = {
    "python": Syntax(("#",), (), ('"', "'"), ('"""', "'''")),
    "cpp": Syntax(("//",), _C_BLOCK, ('"', "'")),
    "c": Syntax(("//",), _C_BLOCK, ('"', "'")),
    "java": Syntax(("//",), _C_BLOCK, ('"', "'"), ('"""',)),
    "javascript": Syntax(("//",), _C_BLOCK, ('"', "'"), ("`",)),
    "typescript": Syntax(("//",), _C_BLOCK, ('"', "'"), ("`",)),
    "go": Syntax(("//",), _C_BLOCK, ('"', "'"), ("`",)),
    "rust": Syntax(("//",), _C_BLOCK, ('"', "'")),
    "sql": Syntax(("--",), _C_BLOCK, ("'", '"')),
    "html": Syntax((), (("<!--", "-->"),)),  # quotes in HTML text are prose, not strings
}


@dataclass(frozen=True)
class ScanResult:
    outside: list[bool]  # outside[i]: line i starts outside comments/strings; len = lines + 1
    stray_closes: int  # comment-closing lines seen outside a comment (window began mid-comment)


def scan(lines: list[str], syntax: Syntax) -> ScanResult:
    outside: list[bool] = []
    stray = 0
    state: tuple | None = None  # None | ("block", close) | ("str", quote, multiline)
    for line in lines:
        outside.append(state is None)
        stripped = line.rstrip()
        i, n = 0, len(line)
        while i < n:
            if state is None:
                opened = False
                for q in syntax.multiline_strings:
                    if line.startswith(q, i):
                        state, i, opened = ("str", q, True), i + len(q), True
                        break
                if opened:
                    continue
                for open_, close in syntax.block_comments:
                    if line.startswith(open_, i):
                        state, i, opened = ("block", close), i + len(open_), True
                        break
                    if line.startswith(close, i):
                        # a line ending in `*/` with no opener: the window started mid-comment.
                        # (Requiring end-of-line avoids regex literals like /.*/ in JS.)
                        if stripped.endswith(close) and open_ not in line:
                            stray += 1
                        i += len(close)
                        opened = True
                        break
                if opened:
                    continue
                if any(line.startswith(c, i) for c in syntax.line_comments):
                    break
                if line[i] in syntax.strings:
                    state = ("str", line[i], False)
                i += 1
            elif state[0] == "block":
                j = line.find(state[1], i)
                if j < 0:
                    break
                i, state = j + len(state[1]), None
            else:
                quote = state[1]
                if line[i] == "\\":
                    i += 2
                elif line.startswith(quote, i):
                    i, state = i + len(quote), None
                else:
                    i += 1
        if state is not None and state[0] == "str" and not state[2]:
            state = None  # single-line strings end at end of line
    outside.append(state is None)
    return ScanResult(outside, stray)


def safe_line_starts(lines: list[str], language: str | None) -> list[bool] | None:
    """Per line (plus one past the end): may a window boundary sit here? None = no constraint."""
    syntax = SYNTAX.get(language or "")
    return scan(lines, syntax).outside if syntax else None


def is_cut(text: str, language: str) -> bool:
    """True if the text ends inside a comment/string, or starts in the middle of a comment."""
    syntax = SYNTAX.get(language)
    if syntax is None:
        return False
    result = scan(text.splitlines(), syntax)
    return result.stray_closes > 0 or not result.outside[-1]
