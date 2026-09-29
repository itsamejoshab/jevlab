"""Strict chain rules, checked locally before anything reaches the oracle.

Latin letters and digits only, no dashes; each word all lowercase, ALL CAPS,
or Capitalized. The site also runs `strictWordCheck`, where Jev rejects
anything it reads as more than one word; long glued tokens are the usual
casualty, so MAX_WORD_LENGTH stays conservative, and shorter words it has
refused are remembered in rules/rejected.py.
"""

from __future__ import annotations

import re
import unicodedata

from . import rejected

MAX_WORD_LENGTH = 16
MAX_WORDS = 60
_LATIN = re.compile(r"^[A-Za-z0-9]+$")


class RuleError(ValueError):
    pass


def check_word(word: str) -> str:
    text = word.strip()
    if not text:
        raise RuleError("empty word")
    if not _LATIN.match(text):
        raise RuleError(f"{word!r}: Latin letters and digits only")
    if len(text) > MAX_WORD_LENGTH:
        raise RuleError(f"{word!r}: longer than {MAX_WORD_LENGTH} (reads as a compound)")
    if rejected.is_rejected(text):
        raise RuleError(f"{word!r}: Jev rejected it as more than one word")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return text
    alpha = "".join(letters)
    if alpha.islower() or alpha.isupper() or (text[0].isupper() and text[1:].lower() == text[1:]):
        return text
    raise RuleError(f"{word!r}: must be lowercase, UPPERCASE, or Capitalized")


def normalize(phrase: str) -> str:
    """Turn free LLM text into a strict chain candidate: drop punctuation, split hyphens."""
    text = unicodedata.normalize("NFKD", phrase)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    words = []
    for word in text.split():
        if not (word.islower() or word.isupper() or (word[0].isupper() and word[1:].islower())):
            word = word.lower()
        words.append(word)
    return " ".join(words)


def option_names(choices: list | None) -> list[str]:
    """Answer option names; the site sends choice options as {"option": ..., "description": ...}."""
    return [str(c.get("option") or "") if isinstance(c, dict) else str(c) for c in choices or []]


OPTION_STOPWORDS = frozenset({"a", "an", "and", "at", "for", "in", "of", "on", "or", "the", "to", "with"})


def _fold(text: str) -> str:
    """The site's word key: NFD, keep only letters and digits, lowercase."""
    stripped = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c)[0] in "LN")
    return stripped.lower()


def option_parts(option: str) -> list[str]:
    """Parts of an answer name a word may not contain; stopwords only count when they are the whole name."""
    parts = [p for p in (_fold(w) for w in option.split()) if p]
    kept = [p for p in parts if p not in OPTION_STOPWORDS] if len(parts) > 1 else parts
    return kept or parts


def option_clash(phrase: str, choices: list | None) -> tuple[str, str] | None:
    """(word, option) for the first word containing part of an answer name, as the site's client checks it."""
    words = [(w, _fold(w)) for w in phrase.split()]
    for option in option_names(choices):
        for part in option_parts(option):
            for word, folded in words:
                if folded and part in folded:
                    return word, option
    return None


def drop_clashing(rows: list[dict], choices: list | None, me: str) -> list[dict]:
    """Board rows without our own rows that break the answer-name rule; the server keeps them, we don't count them."""
    if not choices or not me:
        return rows
    return [r for r in rows if not (r.get("userId") == me and r.get("phrase")
                                    and option_clash(r["phrase"], choices))]


def check_phrase(phrase: str, choices: list | None = None, max_words: int = MAX_WORDS) -> str:
    """Validate a whole chain; returns the canonical space-joined form."""
    words = [check_word(w) for w in phrase.split()]
    if not words:
        raise RuleError("empty phrase")
    if len(words) > max_words:
        raise RuleError(f"{len(words)} words is over the {max_words} cap")
    clash = option_clash(" ".join(words), choices)
    if clash:
        raise RuleError(f"{clash[0]!r} contains part of the answer option {clash[1]!r}")
    return " ".join(words)


def is_valid(phrase: str, choices: list[str] | None = None) -> bool:
    try:
        check_phrase(phrase, choices)
        return True
    except RuleError:
        return False


def canonical(phrase: str) -> str:
    return " ".join(phrase.casefold().split())


class CopyGuard:
    """Never submit an exact phrase we have seen from another player; ours are fine."""

    def __init__(self, foreign: set[str] | None = None):
        self.foreign = {canonical(p) for p in foreign or set()}

    def allowed(self, phrase: str) -> bool:
        return canonical(phrase) not in self.foreign
