"""SQLite store for snapshots, oracle samples, and publish results."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

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
CREATE TABLE IF NOT EXISTS oracle_refusals (
    qkey TEXT NOT NULL,
    state TEXT NOT NULL,
    detail TEXT,
    at REAL,
    PRIMARY KEY (qkey, state)
);
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
CREATE TABLE IF NOT EXISTS phrase_stats (
    qkey TEXT NOT NULL,
    state TEXT NOT NULL,
    n INTEGER NOT NULL,
    total REAL NOT NULL,
    sumsq REAL NOT NULL,
    PRIMARY KEY (qkey, state)
);
CREATE TABLE IF NOT EXISTS phrase_seen (
    state TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS lab_meta (
    key TEXT PRIMARY KEY,
    n INTEGER NOT NULL
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

    def __init__(self, path: Path | None = None, readonly: bool = False):
        """`readonly` opens another edition's database (a mirror reading Jev's) without schema work or writes."""
        if path is None:
            from .config import DB_PATH

            path = DB_PATH
        self.path = path
        if readonly:
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, isolation_level=None)
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
            self._ensure_phrase_stats()
            self._split_embeddings()

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

    def _question(self, row) -> dict:
        out = dict(row)
        out["raw"] = json.loads(out["raw"]) if out["raw"] else {}
        out["jev_request"] = json.loads(out["jev_request"]) if out["jev_request"] else None
        if out["jev_request"] is None and out["raw"].get("jevRequest"):
            from .site.client import decode_jev_request

            out["jev_request"] = decode_jev_request(out["raw"]["jevRequest"])
        return out

    def question(self, slug: str) -> dict | None:
        row = self.one("SELECT * FROM questions WHERE slug = ?", (slug,))
        return self._question(row) if row else None

    def questions(self) -> dict[str, dict]:
        return {row["slug"]: self._question(row) for row in self.all("SELECT * FROM questions")}

    def _choice_lists(self) -> dict[str, list]:
        lists = {}
        for row in self.all(
            "SELECT slug, json_extract(raw, '$.choices') AS choices FROM questions WHERE kind = 'choice'"
        ):
            if row["choices"]:
                lists[row["slug"]] = json.loads(row["choices"])
        return lists

    def _without_clashes(self, slug: str, rows, choices: dict[str, list], me: str) -> list[dict]:
        # `winner` is one object, not a row list. Clash filtering only applies to row lists.
        if not isinstance(rows, list):
            return []
        if slug not in choices:
            return rows
        from .rules import drop_clashing

        return drop_clashing(rows, choices[slug], me)

    def board(self, slug: str, mode: str, board: str) -> list[dict]:
        row = self.one(
            "SELECT rows FROM boards WHERE slug = ? AND mode = ? AND board = ?",
            (slug, mode, board),
        )
        if not row:
            return []
        choices = self.one(
            "SELECT json_extract(raw, '$.choices') AS choices FROM questions WHERE slug = ? AND kind = 'choice'",
            (slug,),
        )
        parsed = {slug: json.loads(choices["choices"])} if choices and choices["choices"] else {}
        return self._without_clashes(slug, json.loads(row["rows"]), parsed, self.me())

    def boards_for(self, mode: str) -> dict[tuple[str, str], list[dict]]:
        """Every board of one play mode, keyed by (slug, board), with the same clash filter as board()."""
        me = self.me()
        choices = self._choice_lists()
        out = {}
        for row in self.all("SELECT slug, board, rows FROM boards WHERE mode = ?", (mode,)):
            out[(row["slug"], row["board"])] = self._without_clashes(
                row["slug"], json.loads(row["rows"]) if row["rows"] else [], choices, me
            )
        return out

    def word_impacts(self, slug: str, mode: str) -> list[dict]:
        row = self.one("SELECT rows FROM word_impacts WHERE slug = ? AND mode = ?", (slug, mode))
        return json.loads(row["rows"]) if row else []

    def me(self) -> str:
        row = self.one("SELECT me FROM snapshots ORDER BY id DESC LIMIT 1")
        return (row["me"] if row else "") or ""

    def latest_snapshot(self) -> dict | None:
        row = self.one("SELECT * FROM snapshots ORDER BY id DESC LIMIT 1")
        return dict(row) if row else None

    def phrase_count(self) -> int:
        row = self.one("SELECT n FROM lab_meta WHERE key = 'phrases'")
        return int(row["n"]) if row else 0

    def _ensure_phrase_stats(self) -> None:
        """One pass over oracle samples. Later opens only read the marker. Caller holds self.lock."""
        if self.conn.execute("SELECT n FROM lab_meta WHERE key = 'phrase_stats'").fetchone() is not None:
            return
        if self.conn.execute("SELECT 1 FROM oracle_samples LIMIT 1").fetchone():
            print("building phrase summary from oracle samples...", flush=True)
            self.conn.execute("BEGIN")
            try:
                self.conn.execute(
                    """INSERT INTO phrase_stats (qkey, state, n, total, sumsq)
                       SELECT qkey, state, COUNT(*), SUM(noul), SUM(noul * noul)
                       FROM oracle_samples
                       WHERE qkey IS NOT NULL AND noul IS NOT NULL
                       GROUP BY qkey, state"""
                )
                self.conn.execute(
                    "INSERT INTO phrase_seen (state) SELECT DISTINCT state FROM oracle_samples WHERE state IS NOT NULL"
                )
                n = self.conn.execute("SELECT COUNT(*) FROM phrase_seen").fetchone()[0]
                self.conn.execute("INSERT INTO lab_meta (key, n) VALUES ('phrases', ?)", (n,))
                self.conn.execute("INSERT INTO lab_meta (key, n) VALUES ('phrase_stats', 1)")
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return
        self.conn.execute("INSERT INTO lab_meta (key, n) VALUES ('phrases', 0)")
        self.conn.execute("INSERT INTO lab_meta (key, n) VALUES ('phrase_stats', 1)")

    def _split_embeddings(self) -> None:
        """Copy the embedding cache out of the hot database, then shrink the file. Caller holds self.lock."""
        from .config import embeddings_path

        tables = {row[0] for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        side = embeddings_path()
        ready = False
        if side.exists():
            side_conn = sqlite3.connect(f"file:{side}?mode=ro", uri=True)
            try:
                ready = (
                    side_conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embeddings_ready'"
                    ).fetchone()
                    is not None
                )
            finally:
                side_conn.close()
        if "embeddings" in tables and not ready:
            print(f"moving embeddings to {side}...", flush=True)
            side.parent.mkdir(parents=True, exist_ok=True)
            if side.exists():
                side.unlink()
            self.conn.execute("ATTACH DATABASE ? AS emb", (str(side),))
            self.conn.execute(
                "CREATE TABLE emb.embeddings (model TEXT, text TEXT, vec BLOB, PRIMARY KEY (model, text))"
            )
            self.conn.execute("INSERT INTO emb.embeddings SELECT model, text, vec FROM embeddings")
            count = self.conn.execute("SELECT COUNT(*) FROM emb.embeddings").fetchone()[0]
            self.conn.execute("CREATE TABLE emb.embeddings_ready (n INTEGER)")
            self.conn.execute("INSERT INTO emb.embeddings_ready VALUES (?)", (count,))
            self.conn.execute("DETACH DATABASE emb")
            ready = True
        if "embeddings" in tables and ready:
            self.conn.execute("DROP TABLE embeddings")
            self.conn.execute(
                "INSERT INTO lab_meta (key, n) VALUES ('embeddings_split', 1) ON CONFLICT(key) DO UPDATE SET n = 1"
            )
        flag = self.conn.execute("SELECT n FROM lab_meta WHERE key = 'embeddings_split'").fetchone()
        freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
        if flag and flag["n"] == 1 and freelist > 1000:
            print("reclaiming space in the main database...", flush=True)
            self.conn.execute("VACUUM")
            self.conn.execute("UPDATE lab_meta SET n = 2 WHERE key = 'embeddings_split'")

    def add_oracle_samples(self, rows: list[tuple]) -> None:
        """Insert oracle rows and keep the phrase summary in the same transaction.

        Each row is (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey).
        """
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(
                    "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                for row in rows:
                    state, noul, qkey = row[2], row[3], row[8]
                    if qkey is None or noul is None or not state:
                        continue
                    value = float(noul)
                    self.conn.execute(
                        """INSERT INTO phrase_stats (qkey, state, n, total, sumsq) VALUES (?, ?, 1, ?, ?)
                           ON CONFLICT(qkey, state) DO UPDATE SET
                             n = n + 1,
                             total = total + excluded.total,
                             sumsq = sumsq + excluded.sumsq""",
                        (qkey, state, value, value * value),
                    )
                    inserted = self.conn.execute("INSERT OR IGNORE INTO phrase_seen (state) VALUES (?)", (state,))
                    if inserted.rowcount:
                        self.conn.execute("UPDATE lab_meta SET n = n + 1 WHERE key = 'phrases'")
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
