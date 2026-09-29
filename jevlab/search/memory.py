"""What earlier runs on a question learned, so a re-run does not start from zero.

One memory per question and board: a Shortest yes run never resumes from a Highest run's level,
exhausted frames, or scheduler state. Runs aimed at one answer of a choice question keep their own memory."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from ..db import DB
from ..modes import HIGH_SCORES


@dataclass
class RunMemory:
    runs: int = 0
    wins: int = 0
    best_p: float = 0.0
    best_units: int = 0
    styles: dict[str, dict] = field(default_factory=dict)  # archetype -> {"best": p, "tries": n}
    arms: dict[str, dict] = field(default_factory=dict)  # scheduler arm state
    compressed: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    exhausted: list[list] = field(default_factory=list)  # [phrase, p, units] frames that plateaued short
    history: list[dict] = field(default_factory=list)  # one summary per run
    level: int = 0  # highest escalation level the last run reached without winning
    grow: list[str] = field(default_factory=list)  # plateau mode: the word-by-word beam, resumed next run
    grow_gains: dict = field(default_factory=dict)  # plateau mode: word -> [total score change, appends]
    drift: str = ""  # plateau mode: where the walk along the top level stands
    probe: dict = field(default_factory=dict)  # plateau mode: probe climb's claim and line

    @classmethod
    def from_json(cls, text: str | None) -> "RunMemory":
        if not text:
            return cls()
        data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load(db: DB, qkey: str, board: str = HIGH_SCORES) -> RunMemory:
    row = db.one("SELECT data FROM lab_memory WHERE qkey = ? AND board = ?", (qkey, board))
    return RunMemory.from_json(row["data"] if row else None)


def save(db: DB, qkey: str, slug: str, memory: RunMemory, board: str = HIGH_SCORES) -> None:
    db.execute("INSERT OR REPLACE INTO lab_memory (qkey, board, slug, data, updated) VALUES (?, ?, ?, ?, ?)",
               (qkey, board, slug, json.dumps(asdict(memory)), time.time()))


def by_target(db: DB, board: str = HIGH_SCORES) -> dict[tuple[str, str], RunMemory]:
    """Latest memory per (question, answer) for answer-aimed runs, stored under board '<board>@<answer>'."""
    out: dict[tuple[str, str], RunMemory] = {}
    for row in db.all("SELECT slug, board, data FROM lab_memory WHERE board LIKE ? ORDER BY updated",
                      (f"{board}@%",)):
        out[(row["slug"], row["board"].partition("@")[2])] = RunMemory.from_json(row["data"])
    return out


def by_slug(db: DB, board: str = HIGH_SCORES) -> dict[str, RunMemory]:
    """Latest memory per question on one board (a question revision gets a new qkey)."""
    out: dict[str, RunMemory] = {}
    for row in db.all("SELECT slug, data FROM lab_memory WHERE board = ? ORDER BY updated", (board,)):
        out[row["slug"]] = RunMemory.from_json(row["data"])
    return out
