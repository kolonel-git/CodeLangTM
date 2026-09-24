"""Extract human-readable AND-rules from trained clauses. Phase 3."""

from __future__ import annotations


def format_rule(language: str, included: list[str], negated: list[str]) -> str:
    """Render a clause, e.g. `python = has("def ") AND has(":") AND NOT has(";")`."""
    parts = [f'has("{t}")' for t in included] + [f'NOT has("{t}")' for t in negated]
    return f"{language} = " + " AND ".join(parts)


def extract_rules(model, vocabulary: list[str]) -> list[str]:
    raise NotImplementedError("Phase 3: clause extraction")
