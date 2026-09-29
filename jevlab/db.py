"""SQLite store for snapshots, oracle samples, and publish results."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    edition TEXT NOT NULL,
    me TEXT,
    raw_dir TEXT
);
CREATE TABLE IF NOT EXISTS questions (
    slug TEXT PRIMARY KEY,
    snapshot_id INTEGER,
    revision_id TEXT,
    title TEXT,
    instructions TEXT,
    kind TEXT,
    goal TEXT,
    baseline REAL,
    yes_threshold REAL,
    model_version TEXT,
    jev_request TEXT,
    raw TEXT
);
CREATE TABLE IF NOT EXISTS boards (
    slug TEXT,
    mode TEXT,
    board TEXT,
    snapshot_id INTEGER,
    rows TEXT,
    PRIMARY KEY (slug, mode, board)
);
CREATE TABLE IF NOT EXISTS word_impacts (
    slug TEXT,
    mode TEXT,
    snapshot_id INTEGER,
    rows TEXT,
    PRIMARY KEY (slug, mode)
);
CREATE TABLE IF NOT EXISTS attempts (
    slug TEXT,
    mode TEXT,
    snapshot_id INTEGER,
    raw TEXT,
    PRIMARY KEY (slug, mode)
);
CREATE TABLE IF NOT EXISTS site_scores (
    slug TEXT,
    mode TEXT,
    state TEXT,
    probability REAL,
    source TEXT,
    seen_at TEXT,
    PRIMARY KEY (slug, mode, state, source)
);
CREATE TABLE IF NOT EXISTS oracle_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_hash TEXT NOT NULL,
    model TEXT,
    state TEXT,
    noul REAL,
    answer TEXT,
    latency_ms INTEGER,
    cost REAL,
    at REAL
);
CREATE INDEX IF NOT EXISTS oracle_hash ON oracle_samples(request_hash);
CREATE TABLE IF NOT EXISTS lab_candidates (
    qkey TEXT,
    state TEXT,
    origin TEXT,
    parent TEXT,
    created REAL,
    PRIMARY KEY (qkey, state)
);
CREATE TABLE IF NOT EXISTS lab_memory (
    qkey TEXT,
    board TEXT NOT NULL DEFAULT 'highScores',
    slug TEXT,
    data TEXT,
    updated REAL,
    PRIMARY KEY (qkey, board)
);
CREATE TABLE IF NOT EXISTS publishes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT,
    mode TEXT,
    phrase TEXT,
    estimate REAL,
    server REAL,
    status TEXT,
    detail TEXT,
    at TEXT,
    board TEXT NOT NULL DEFAULT 'highScores'
);
"""

# Before game modes, lab_memory was keyed by qkey alone and every row was a Highest run.
MIGRATE_LAB_MEMORY = """
BEGIN;
ALTER TABLE lab_memory RENAME TO lab_memory_v1;
CREATE TABLE lab_memory (
    qkey TEXT,
    board TEXT NOT NULL DEFAULT 'highScores',
    slug TEXT,
    data TEXT,
    updated REAL,
    PRIMARY KEY (qkey, board)
);
INSERT INTO lab_memory (qkey, board, slug, data, updated)
    SELECT qkey, 'highScores', slug, data, updated FROM lab_memory_v1;
DROP TABLE lab_memory_v1;
COMMIT;
"""


class DB:
    """Thread-safe enough for our use: one connection, one lock."""

    def __init__(self, path: Path = DB_PATH, readonly: bool = False):
        """`readonly` opens another edition's database (Kev reading Jev's) without schema work or writes."""
        self.path = path
        if readonly:
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False,
                                        isolation_level=None)
        else:
            self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        if not readonly:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.lock = threading.Lock()
        if readonly:
            return
        with self.lock:
            self.conn.executescript(SCHEMA)
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(oracle_samples)")}
            if "qkey" not in columns:
                self.conn.execute("ALTER TABLE oracle_samples ADD COLUMN qkey TEXT")
            self.conn.execute("CREATE INDEX IF NOT EXISTS oracle_qkey ON oracle_samples(qkey)")
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(lab_candidates)")}
            if "archetype" not in columns:
                self.conn.execute("ALTER TABLE lab_candidates ADD COLUMN archetype TEXT")
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(lab_memory)")}
            if "board" not in columns:
                try:
                    self.conn.executescript(MIGRATE_LAB_MEMORY)
                except sqlite3.Error:
                    if self.conn.in_transaction:
                        self.conn.execute("ROLLBACK")
                    raise
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(publishes)")}
            if "board" not in columns:
                self.conn.execute("ALTER TABLE publishes ADD COLUMN board TEXT NOT NULL DEFAULT 'highScores'")

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: list) -> None:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(sql, rows)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return list(self.conn.execute(sql, params))

    def one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    # Convenience readers used by the lab and publisher.

    def question(self, slug: str) -> dict | None:
        row = self.one("SELECT * FROM questions WHERE slug = ?", (slug,))
        if not row:
            return None
        out = dict(row)
        out["raw"] = json.loads(out["raw"]) if out["raw"] else {}
        out["jev_request"] = json.loads(out["jev_request"]) if out["jev_request"] else None
        if out["jev_request"] is None and out["raw"].get("jevRequest"):
            from .site.client import decode_jev_request

            out["jev_request"] = decode_jev_request(out["raw"]["jevRequest"])
        return out

    def board(self, slug: str, mode: str, board: str) -> list[dict]:
        row = self.one(
            "SELECT rows FROM boards WHERE slug = ? AND mode = ? AND board = ?",
            (slug, mode, board),
        )
        if not row:
            return []
        rows = json.loads(row["rows"])
        choices = self.one("SELECT json_extract(raw, '$.choices') AS choices FROM questions "
                           "WHERE slug = ? AND kind = 'choice'", (slug,))
        if choices and choices["choices"]:
            from .rules import drop_clashing

            rows = drop_clashing(rows, json.loads(choices["choices"]), self.me())
        return rows

    def word_impacts(self, slug: str, mode: str) -> list[dict]:
        row = self.one("SELECT rows FROM word_impacts WHERE slug = ? AND mode = ?", (slug, mode))
        return json.loads(row["rows"]) if row else []

    def me(self) -> str:
        row = self.one("SELECT me FROM snapshots ORDER BY id DESC LIMIT 1")
        return (row["me"] if row else "") or ""

    def latest_snapshot(self) -> dict | None:
        row = self.one("SELECT * FROM snapshots ORDER BY id DESC LIMIT 1")
        return dict(row) if row else None
