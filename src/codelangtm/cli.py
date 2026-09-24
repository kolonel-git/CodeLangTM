"""Command-line entrypoint."""

from __future__ import annotations

import argparse

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codelangtm", description="Identify the programming language of a code snippet."
    )
    parser.add_argument("--version", action="version", version=f"codelangtm {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
