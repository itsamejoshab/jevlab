"""Winners found offline, waiting to be published.

One JSON file per question, mode, and board: data/vault/<slug>/<mode>.json for Highest (the original
layout) and data/vault/<slug>/<mode>.shortestYes.json for Shortest yes. Lines aimed at one answer of a choice
question live beside them as <mode>@<answer>.json and <mode>@<answer>.shortestYes.json.
Status lifecycle: candidate -> queued -> published | failed | rejected (or dropped).
`failed` may be retried (the server score fell short); `rejected` means Jev refused a word as more than one
word, which no retry fixes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import VAULT
from .modes import BOARDS, HIGH_SCORES
from .objective import Leader, Objective

STATUSES = ("candidate", "queued", "published", "failed", "rejected", "dropped")


def target_key(target: str) -> str:
    """File-name-safe form of a choice answer."""
    return re.sub(r"[^a-z0-9]+", "-", target.casefold()).strip("-") or "answer"


def _path(slug: str, mode: str, board: str = HIGH_SCORES, target: str = "") -> Path:
    stem = f"{mode}@{target_key(target)}" if target else mode
    if board == HIGH_SCORES:
        return VAULT / slug / f"{stem}.json"
    return VAULT / slug / f"{stem}.{board}.json"


def _split(path: Path) -> tuple[str, str]:
    """(mode, board) from a vault file name."""
    stem = path.name[: -len(".json")]
    mode, _, board = stem.partition(".")
    return mode.partition("@")[0], board or HIGH_SCORES


def load(slug: str, mode: str, board: str = HIGH_SCORES, target: str = "") -> dict:
    path = _path(slug, mode, board, target)
    if path.exists():
        data = json.loads(path.read_text())
        data.setdefault("board", board)
        return data
    return {"slug": slug, "mode": mode, "board": board, "target": target, "entries": []}


def _write(data: dict) -> None:
    path = _path(data["slug"], data["mode"], data.get("board") or HIGH_SCORES, data.get("target") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def eligible(objective: Objective, p_lcb: float, units: int, leader: Leader | None) -> bool:
    """Only save lines whose lower confidence bound still wins on the board."""
    return objective.beats(p_lcb, units, leader)


def save(slug: str, mode: str, phrase: str, *, p_mean: float, p_lcb: float, spread: float, n: int,
         units: int, leader: Leader | None, beats: bool, title: str = "", origin: str = "",
         status: str = "candidate", note: str = "", board: str = HIGH_SCORES, extra: dict | None = None,
         target: str = "", estimated_from: str = "", jev_p: float | None = None) -> dict:
    """`extra` carries board-specific fields, e.g. Shortest yes gambles: gamble, p_reach (best roll), hits.
    `target` files the line under one answer of a choice question (its p values are P(target)).
    `estimated_from` names the edition whose scores stand in for this one's (a Kev line copied from Jev, n=0).
    An estimate never overwrites a measured line; a measured save turns an estimate into a measured line."""
    data = load(slug, mode, board, target)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = next((e for e in data["entries"] if e["phrase"] == phrase), None)
    if estimated_from and entry is not None and not entry.get("estimated_from"):
        if jev_p is not None and entry.get("jev_p") != round(jev_p, 4):
            entry["jev_p"] = round(jev_p, 4)
            _write(data)
        return entry
    fields = {
        "phrase": phrase, "p_mean": round(p_mean, 4), "p_lcb": round(p_lcb, 4), "spread": round(spread, 4),
        "n": n, "units": units, "beats": beats, "title": title, "origin": origin,
        "leader": {"p": leader.probability, "units": leader.units, "name": leader.name} if leader else None,
        "updated_at": now,
    }
    if board != HIGH_SCORES:
        fields.update({"gamble": False, "p_reach": None, "hits": None} | (extra or {}))
    if jev_p is not None:
        fields["jev_p"] = round(jev_p, 4)
    if estimated_from:
        fields["estimated_from"] = estimated_from
    elif entry is not None and entry.get("estimated_from"):
        entry.pop("estimated_from")
        entry.setdefault("history", []).append({"status": "measured", "at": now, "n": n})
    if entry is None:
        entry = fields | {"status": status, "saved_at": now, "note": note, "history": []}
        data["entries"].append(entry)
    else:
        if entry.get("status") in ("published",):
            fields.pop("beats")
        entry.update(fields)
        if status == "queued" and entry.get("status") == "candidate":
            entry["status"] = "queued"
    if board == HIGH_SCORES:
        data["entries"].sort(key=lambda e: (-round(e["p_mean"], 2), e["units"], -e["p_mean"]))
    else:
        data["entries"].sort(key=lambda e: (e["units"], -round(e["p_mean"], 2), -e["p_mean"]))
    _write(data)
    return entry


def set_status(slug: str, mode: str, phrase: str, status: str, board: str = HIGH_SCORES, target: str = "",
               **extra) -> dict | None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    data = load(slug, mode, board, target)
    for entry in data["entries"]:
        if entry["phrase"] == phrase:
            entry["status"] = status
            entry.setdefault("history", []).append(
                {"status": status, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")} | extra
            )
            entry.update({k: v for k, v in extra.items() if k in ("server_p", "detail")})
            _write(data)
            return entry
    return None


def all_entries(board: str | None = None, root: Path | None = None) -> list[dict]:
    """Every entry, tagged with slug, mode, and board; only one board's when `board` is given.
    `root` reads another edition's vault (config.JEV_VAULT from Kev)."""
    out = []
    root = root or VAULT
    if not root.exists():
        return out
    for path in sorted(root.glob("*/*.json")):
        mode, file_board = _split(path)
        if file_board not in BOARDS or (board is not None and file_board != board):
            continue
        data = json.loads(path.read_text())
        for entry in data["entries"]:
            out.append(entry | {"slug": data["slug"], "mode": data.get("mode") or mode, "board": file_board,
                                "target": data.get("target") or ""})
    return out


def queued(board: str | None = None) -> list[dict]:
    return [e for e in all_entries(board) if e.get("status") == "queued"]
