"""Emoji words, checked the way the site checks them before a turn is scored.

One append is one grapheme: a flag, a keycap, or a pictograph run joined by
zero-width joiners. A phrase is those words separated by spaces. A single
code point that is not emoji-presentation gains U+FE0F, matching the site.
"""

from __future__ import annotations

import unicodedata

import regex

MAX_CODE_UNITS = 512
ZWJ = "\u200d"

_WORD = regex.compile(
    r"^(?:"
    r"\p{Regional_Indicator}{2}"
    r"|[0-9#*]\uFE0F?\u20E3"
    r"|\p{Extended_Pictographic}[\uFE0F\p{Emoji_Modifier}\U000E0020-\U000E007F]*"
    r"(?:\u200D\p{Extended_Pictographic}[\uFE0F\p{Emoji_Modifier}]*)*"
    r")$"
)
_TEXT_DEFAULT = regex.compile(r"^\P{Emoji_Presentation}$")


class RuleError(ValueError):
    pass


def code_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def graphemes(text: str) -> list[str]:
    return [piece for piece in regex.findall(r"\X", text) if piece.strip()]


def _canonicalize(grapheme: str) -> str:
    if _TEXT_DEFAULT.fullmatch(grapheme):
        return grapheme + "\ufe0f"
    return grapheme


def check_word(word: str) -> str:
    """One emoji grapheme in the form the site stores, or RuleError."""
    text = unicodedata.normalize("NFC", word).strip()
    if not text:
        raise RuleError("empty")
    parts = graphemes(text)
    if not parts or any(_WORD.fullmatch(part) is None for part in parts):
        raise RuleError("not_emoji")
    if len(parts) > 1:
        raise RuleError("too_many_words")
    canonical = _canonicalize(parts[0])
    if code_units(canonical) > MAX_CODE_UNITS:
        raise RuleError("too_long")
    return canonical


def check_phrase(phrase: str) -> str:
    """Space-separated emoji words. Each word stays one grapheme, so a ZWJ chain is one word."""
    text = unicodedata.normalize("NFC", phrase).strip()
    if not text:
        raise RuleError("empty")
    return " ".join(check_word(word) for word in text.split())


def chain(atoms: list[str]) -> str:
    """Join pictographs into one word. The caller checks the result."""
    return ZWJ.join(atoms)
