"""Words the live site refused as "more than one word", learned from publish attempts.

Jev's strictWordCheck is an LLM call we cannot run offline, so the only reliable
signal is a rejected turn. Every rejected word lands here and `check_word` refuses
it from then on, which keeps it out of the search, the vault picker, and the queue.
The file is shared between processes (lab, TUI, publisher), so reads follow its mtime.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from ..config import JEV_DATA

# The word check is the site's, shared by every edition.
PATH = JEV_DATA / "rejected_words.json"
_RECHECK = 2.0

_words: dict[str, dict] = {}
_mtime: float | None = None
_checked = 0.0


def is_word_rejection(message: str) -> bool:
    return "more than one word" in message.casefold()


def _refresh(force: bool = False) -> dict[str, dict]:
    global _words, _mtime, _checked
    now = time.monotonic()
    if not force and now - _checked < _RECHECK:
        return _words
    _checked = now
    try:
        mtime = PATH.stat().st_mtime
    except FileNotFoundError:
        _words, _mtime = {}, None
        return _words
    if mtime != _mtime:
        try:
            _words = json.loads(PATH.read_text()).get("words") or {}
        except (OSError, json.JSONDecodeError):
            return _words
        _mtime = mtime
    return _words


def words() -> dict[str, dict]:
    return dict(_refresh(force=True))


def is_rejected(word: str) -> bool:
    return word.casefold() in _refresh()


def add(word: str, slug: str = "", detail: str = "") -> None:
    current = dict(_refresh(force=True))
    key = word.casefold()
    if key in current:
        return
    current[key] = {"word": word, "slug": slug, "detail": detail,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"words": dict(sorted(current.items()))}, indent=1))
    tmp.replace(PATH)
    _refresh(force=True)
