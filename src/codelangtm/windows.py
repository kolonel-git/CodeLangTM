"""Cut contiguous line windows from real source files, skipping low-signal ones."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

COMMENT_PREFIXES = ("//", "#", "/*", "*", "*/", "--", "<!--", "-->", '"""', "'''", ";")
LICENSE_RE = re.compile(r"copyright|spdx-license|licensed under|all rights reserved", re.I)


@dataclass(frozen=True)
class Window:
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str


def _is_comment(line: str) -> bool:
    return line.strip().startswith(COMMENT_PREFIXES)


def is_low_signal(
    lines: list[str], min_nonblank: int = 10, max_comment_frac: float = 0.8
) -> bool:
    """True for windows that are mostly blank, license text, or pure comments."""
    nonblank = [ln for ln in lines if ln.strip()]
    if len(nonblank) < min_nonblank:
        return True
    if any(LICENSE_RE.search(ln) for ln in nonblank):
        return True
    comments = sum(_is_comment(ln) for ln in nonblank)
    return comments / len(nonblank) > max_comment_frac


def extract_windows(
    text: str,
    min_lines: int = 20,
    max_lines: int = 50,
    max_windows: int | None = None,
    seed: int = 0,
    min_nonblank: int = 10,
) -> list[Window]:
    """Non-overlapping windows of min_lines..max_lines lines, in file order.

    Window length is drawn per window from a seeded RNG, so output is deterministic.
    Files shorter than min_lines yield nothing.
    """
    if min_lines < 1 or max_lines < min_lines:
        raise ValueError("need 1 <= min_lines <= max_lines")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rng = random.Random(seed)
    out: list[Window] = []
    i = 0
    while i + min_lines <= len(lines):
        size = min(rng.randint(min_lines, max_lines), len(lines) - i)
        chunk = lines[i : i + size]
        if not is_low_signal(chunk, min_nonblank=min_nonblank):
            out.append(Window(i + 1, i + size, "\n".join(chunk) + "\n"))
            if max_windows is not None and len(out) >= max_windows:
                break
        i += size
    return out
