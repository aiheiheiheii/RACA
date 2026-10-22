"""Current paper name and display aliases for saved experimental records.

Aliases only change method labels. Callers retain saved source/version identities
and use the original method label when accessing historical NPZ arrays.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

RACA_HIM = "RACA-HIM"


def canonical_method_name(name: str) -> str:
    return RACA_HIM if name == "Proposed" else name


def normalize_method_rows(rows: Iterable[Mapping]) -> list[dict]:
    """Copy records with current method labels; never mutate saved input rows."""
    return [dict(row, method=canonical_method_name(row["method"])) for row in rows]
