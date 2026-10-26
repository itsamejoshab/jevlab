"""Phrases the site refused with banned_phrase.

The error quotes the span, for example '"this logic game" is banned'. Only that
span is remembered. Its words stay legal on their own, and an error that does
not name a span is not stored.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

from ..config import JEV_DATA

PATH = JEV_DATA / "banned_phrases.json"
_RECHECK = 2.0
_QUOTED = re.compile(r'"([^"]+)"\s+is banned', re.IGNORECASE)

_phrases: dict[str, dict] = {}
_needles: list[tuple[str, tuple[str, ...]]] = []
# Spans grouped by their first token, in the same order as `_needles`.
_by_first: dict[str, list[tuple[int, str, tuple[str, ...]]]] = {}
_mtime: float | None = None
_checked = 0.0


class BannedPhrase(Exception):
    """A submission the site banned. `named` is the quoted span, or empty when the server did not say."""

    def __init__(self, submitted: str, named: str = ""):
        self.submitted = submitted
        self.named = named
        detail = f"{named!r} is banned" if named else "banned_phrase, span not named"
        super().__init__(detail)


def quoted_ban(message: str) -> str | None:
    """The span the server named, or None when the message does not name one."""
    text = (message or "").replace('\\"', '"')
    match = _QUOTED.search(text)
    if not match:
        return None
    span = " ".join(match.group(1).split())
    return span or None


def tokens(phrase: str) -> list[str]:
    """Words of a chain. Hyphens count as spaces, because casual posts a phrase that way."""
    return [word for word in re.split(r"[\s\-]+", (phrase or "").casefold()) if word]


def _matches_at(hay: list[str], start: int, needle: tuple[str, ...]) -> bool:
    width = len(needle)
    if not width or width > len(hay) - start:
        return False
    if hay[start] != needle[0]:
        return False
    return width == 1 or all(hay[start + offset] == needle[offset] for offset in range(1, width))


def _contains_tokens(hay: list[str], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    if not width or width > len(hay):
        return False
    last = len(hay) - width + 1
    for start in range(last):
        if _matches_at(hay, start, needle):
            return True
    return False


def _remember(phrases: dict[str, dict], mtime: float | None) -> dict[str, dict]:
    global _phrases, _needles, _by_first, _mtime
    _phrases = phrases
    _mtime = mtime
    needles = []
    by_first: dict[str, list[tuple[int, str, tuple[str, ...]]]] = {}
    for key, row in phrases.items():
        span = row.get("phrase") or key
        needle = tuple(tokens(span))
        if not needle:
            continue
        ordinal = len(needles)
        needles.append((span, needle))
        by_first.setdefault(needle[0], []).append((ordinal, span, needle))
    _needles = needles
    _by_first = by_first
    return _phrases


def _refresh(force: bool = False) -> dict[str, dict]:
    global _checked
    now = time.monotonic()
    if not force and now - _checked < _RECHECK:
        return _phrases
    _checked = now
    try:
        mtime = PATH.stat().st_mtime
    except FileNotFoundError:
        return _remember({}, None)
    if mtime != _mtime:
        try:
            loaded = json.loads(PATH.read_text()).get("phrases") or {}
        except (OSError, json.JSONDecodeError):
            return _phrases
        return _remember(loaded, mtime)
    return _phrases


def phrases() -> dict[str, dict]:
    return dict(_refresh(force=True))


def contains(phrase: str) -> bool:
    return hit(phrase) != ""


def hit(phrase: str, span: str = "") -> str:
    """The stored span inside `phrase`, or empty when none of them occur.

    `span` tests that one phrase and skips the rest. A sweep after a new ban has only that span to apply.
    """
    hay = tokens(phrase)
    if not hay:
        return ""
    if span:
        needle = tuple(tokens(span))
        return span if _contains_tokens(hay, needle) else ""
    _refresh()
    # One pass over the phrase. Each word is compared only with spans that start with it.
    # The lowest stored ordinal wins, which is the span the old full scan would return.
    found: int | None = None
    named = ""
    for index, word in enumerate(hay):
        bucket = _by_first.get(word)
        if not bucket:
            continue
        for ordinal, candidate, needle in bucket:
            if found is not None and ordinal >= found:
                break
            if not _matches_at(hay, index, needle):
                continue
            found = ordinal
            named = candidate
            break
        if found == 0:
            return named
    return named


def add(span: str, detail: str = "") -> None:
    span = " ".join((span or "").split())
    if not span:
        return
    current = dict(_refresh(force=True))
    key = " ".join(tokens(span))
    if not key or key in current:
        return
    current[key] = {
        "phrase": span,
        "detail": detail,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"phrases": dict(sorted(current.items()))}, indent=1))
    tmp.replace(PATH)
    _refresh(force=True)


def note(code: str, message: str) -> str | None:
    """Store a server-named ban. None when this is not a ban, or the server did not name the span."""
    text = f"{code} {message}".casefold()
    if "banned_phrase" not in text and "is banned" not in text:
        return None
    named = quoted_ban(message)
    if not named:
        return None
    add(named, message)
    return named
