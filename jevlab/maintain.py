"""Shrink the data directories without dropping oracle scores.

`jevlab maintain` and the home-screen Vacuum button run this. Old snapshot folders
go away. The distinct-phrase table is only a counter, so it is dropped and the
counter moves onto phrase summaries. Embedding vectors are rewritten at half precision.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .config import EDITIONS, JEV_DATA
from .search.vectors import LEGACY_F32_BYTES

KEEP_SNAPSHOTS = 1


def data_roots() -> list[Path]:
    roots = [JEV_DATA]
    for name in EDITIONS:
        if name == "jev":
            continue
        path = JEV_DATA / name
        if (path / "jev.db").exists() or (path / "snapshots").exists() or (path / "embeddings.db").exists():
            roots.append(path)
    return roots


def prune_snapshots(root: Path, keep: int = KEEP_SNAPSHOTS) -> int:
    """Delete older raw snapshot folders. Search reads the SQLite copy of the boards."""
    snap = root / "snapshots"
    if not snap.is_dir():
        return 0
    dirs = sorted((p for p in snap.iterdir() if p.is_dir()), key=lambda p: p.name)
    stale = dirs[:-keep] if keep else dirs
    for path in stale:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    return len(stale)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def drop_phrase_seen(db_path: Path) -> bool:
    """Drop the distinct-phrase table. The counter becomes the number of phrase summaries."""
    conn = _connect(db_path)
    try:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'phrase_seen'"
        ).fetchone()
        pending = conn.execute("SELECT n FROM lab_meta WHERE key = 'phrase_seen_vacuum'").fetchone()
        if present is None:
            if pending is not None and pending[0] == 1 and conn.execute("PRAGMA freelist_count").fetchone()[0] > 1000:
                conn.execute("VACUUM")
                conn.execute("UPDATE lab_meta SET n = 0 WHERE key = 'phrase_seen_vacuum'")
                return True
            return False
        count = conn.execute("SELECT count(*) FROM phrase_stats").fetchone()[0]
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO lab_meta (key, n) VALUES ('phrase_seen_vacuum', 1) "
                "ON CONFLICT(key) DO UPDATE SET n = 1"
            )
            conn.execute("DROP TABLE phrase_seen")
            conn.execute(
                "INSERT INTO lab_meta (key, n) VALUES ('phrases', ?) "
                "ON CONFLICT(key) DO UPDATE SET n = excluded.n",
                (count,),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("VACUUM")
        conn.execute("UPDATE lab_meta SET n = 0 WHERE key = 'phrase_seen_vacuum'")
        return True
    finally:
        conn.close()


def compact_embeddings(path: Path, log=print) -> bool:
    """Rewrite a vector cache as float16, packed onto large pages. A 4 KB page holds one
    half-size vector and wastes the rest, so the rewrite uses 32 KB pages."""
    if not path.exists():
        return False
    conn = _connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS embedding_codec (key TEXT PRIMARY KEY, n INTEGER NOT NULL)")
        done = conn.execute("SELECT n FROM embedding_codec WHERE key = 'done'").fetchone()
        page = conn.execute("PRAGMA page_size").fetchone()[0]
        if done is not None and done[0] == 1 and page >= 32768:
            return False
    finally:
        conn.close()

    dest = path.with_name(path.name + ".half")
    if dest.exists():
        dest.unlink()
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = sqlite3.connect(dest, isolation_level=None)
    try:
        out.execute("PRAGMA page_size=32768")
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute(
            "CREATE TABLE embeddings (model TEXT, text TEXT, vec BLOB, PRIMARY KEY (model, text))"
        )
        out.execute("CREATE TABLE embedding_codec (key TEXT PRIMARY KEY, n INTEGER NOT NULL)")
        pending: list[tuple] = []
        written = 0

        def flush() -> None:
            nonlocal pending
            if not pending:
                return
            out.executemany("INSERT INTO embeddings (model, text, vec) VALUES (?, ?, ?)", pending)
            pending = []

        for model, text, vec in src.execute("SELECT model, text, vec FROM embeddings"):
            blob = bytes(vec)
            if len(blob) == LEGACY_F32_BYTES:
                blob = np.frombuffer(blob, dtype=np.float32).astype(np.float16).tobytes()
            pending.append((model, text, blob))
            written += 1
            if len(pending) >= 2000:
                flush()
                if written % 100000 < 2000:
                    log(f"{path.parent.name}/embeddings: {written} vectors rewritten")
        flush()
        out.execute("INSERT INTO embedding_codec (key, n) VALUES ('done', 1)")
        log(f"{path.parent.name}/embeddings: {written} vectors at half precision")
    finally:
        src.close()
        out.close()
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)
    dest.replace(path)
    return True


def run(log=print) -> None:
    for root in data_roots():
        removed = prune_snapshots(root)
        if removed:
            log(f"{root.name}: removed {removed} old snapshot folders")
        db_path = root / "jev.db"
        if db_path.exists() and drop_phrase_seen(db_path):
            log(f"{root.name}: dropped the distinct-phrase table and reclaimed its space")
        embeddings = root / "embeddings.db"
        if embeddings.exists() and compact_embeddings(embeddings, log):
            log(f"{root.name}: embedding cache is half precision")
