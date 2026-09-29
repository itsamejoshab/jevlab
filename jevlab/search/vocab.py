"""Big candidate vocabulary for exhaustive sweeps and beam building."""

from __future__ import annotations

import threading
from collections import Counter

from ..db import DB
from ..rules import RuleError, check_word

_lock = threading.Lock()
_global: list[str] | None = None


def _valid(word: str) -> str | None:
    try:
        return check_word(word)
    except RuleError:
        return None


def global_words(db: DB, english: int = 20000) -> list[str]:
    """Words from every phrase we ever scored (most used first), then common English. Cached per process."""
    global _global
    with _lock:
        if _global is not None:
            return _global
        counts: Counter[str] = Counter()
        for row in db.all("SELECT DISTINCT state FROM oracle_samples"):
            for word in (row["state"] or "").split():
                counts[word.lower()] += 1
        ordered: dict[str, None] = {}
        for word, _ in counts.most_common():
            if (w := _valid(word)) is not None:
                ordered.setdefault(w, None)
        try:
            from wordfreq import top_n_list

            for word in top_n_list("en", english):
                if (w := _valid(word)) is not None:
                    ordered.setdefault(w, None)
        except ImportError:
            pass
        _global = list(ordered)
        return _global


def build(ctx, limit: int = 20000) -> list[str]:
    """Question-specific words first (pins, helpful impacts, our best lines, synonyms), then the global list."""
    ordered: dict[str, None] = {}
    for word in ctx.word_pool(2000):
        ordered.setdefault(word, None)
    for word in global_words(ctx.engine.db):
        if len(ordered) >= limit:
            break
        if word.casefold() not in ctx.banned:
            ordered.setdefault(word, None)
    return list(ordered)[:limit]


def casings(word: str) -> list[str]:
    """The strict-legal casings of a word: lowercase, Capitalized, UPPER."""
    if not any(c.isalpha() for c in word):
        return [word]
    return list(dict.fromkeys([word.lower(), word.capitalize(), word.upper()]))
