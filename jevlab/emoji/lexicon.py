"""Catalog word index and the closed-class glyph map.

A name token points at the atoms whose English name contains it. Closed-class
words (yes, no, never) use a fixed glyph list instead of the nearest name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .rank import Atom

_NAME = re.compile(r"[a-z0-9]+")
_PHRASE = re.compile(r"[a-z0-9']+")

STOPS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "for",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "be",
        "it",
        "this",
        "that",
        "with",
        "you",
        "your",
        "we",
        "they",
    }
)

# First glyph that is in the catalog wins.
CLOSED: dict[str, tuple[str, ...]] = {
    "yes": ("✅", "👍", "🆗"),
    "yeah": ("✅", "👍", "🆗"),
    "ok": ("✅", "👍", "🆗"),
    "okay": ("✅", "👍", "🆗"),
    "no": ("🚫", "❌", "👎", "🙅"),
    "not": ("🚫", "❌", "👎", "🙅"),
    "never": ("🚫", "❌", "👎", "🙅"),
    "dont": ("🚫", "❌", "👎", "🙅"),
    "don't": ("🚫", "❌", "👎", "🙅"),
    "stop": ("🚫", "❌", "👎", "🙅"),
    "love": ("❤️",),
    "good": ("👍", "💯"),
    "great": ("👍", "💯"),
    "best": ("👍", "💯"),
    "bad": ("👎",),
    "worse": ("👎",),
    "worst": ("👎",),
}


def tokens(text: str) -> list[str]:
    """Casefolded phrase pieces, apostrophes kept so `don't` stays one token."""
    return _PHRASE.findall(text.casefold())


def name_tokens(name: str) -> list[str]:
    """Name pieces long enough to be a word. `ok` stays out; closed-class covers it."""
    return [token for token in _NAME.findall(name.casefold()) if len(token) >= 3]


@dataclass(frozen=True)
class Index:
    """`by_token` lists atoms with that name token, fewest name tokens first."""

    atom_by_text: dict[str, Atom]
    by_token: dict[str, list[Atom]]

    @classmethod
    def build(cls, catalog: list[Atom]) -> Index:
        atom_by_text = {atom.text: atom for atom in catalog}
        buckets: dict[str, list[Atom]] = {}
        for atom in catalog:
            for token in set(name_tokens(atom.name)):
                buckets.setdefault(token, []).append(atom)
        by_token = {
            token: sorted(atoms, key=lambda atom: (len(name_tokens(atom.name)), len(atom.name), atom.text))
            for token, atoms in buckets.items()
        }
        return cls(atom_by_text, by_token)
