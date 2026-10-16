"""Maintenance shrinks caches and leaves oracle scores alone."""

import sqlite3

import numpy as np

from jevlab.maintain import compact_embeddings, drop_phrase_seen, prune_snapshots
from jevlab.search.vectors import LEGACY_F32_BYTES, pack_half, unpack_vec


def test_half_precision_round_trip_and_legacy_full_precision():
    original = np.array([0.25, -1.5, 0.0, 4.0], dtype=np.float32)
    restored = unpack_vec(pack_half(original))
    assert restored.dtype == np.float32
    assert np.allclose(restored, original, atol=1e-3)

    legacy = np.arange(LEGACY_F32_BYTES // 4, dtype=np.float32)
    blob = legacy.tobytes()
    assert len(blob) == LEGACY_F32_BYTES
    assert np.array_equal(unpack_vec(blob), legacy)


def test_prune_snapshots_keeps_the_newest_folder(tmp_path):
    snap = tmp_path / "snapshots"
    for name in ("20261001T000000Z", "20261015T000000Z", "20261016T000000Z"):
        folder = snap / name
        folder.mkdir(parents=True)
        (folder / "menu.json").write_text("{}")
    assert prune_snapshots(tmp_path) == 2
    left = list(snap.iterdir())
    assert [p.name for p in left] == ["20261016T000000Z"]
    assert (left[0] / "menu.json").read_text() == "{}"


def test_drop_phrase_seen_keeps_phrase_summaries(tmp_path):
    path = tmp_path / "jev.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE phrase_stats (qkey TEXT, state TEXT, n INTEGER, total REAL, sumsq REAL)")
    conn.execute("INSERT INTO phrase_stats VALUES ('q', 'alpha', 2, 1.5, 1.25)")
    conn.execute("CREATE TABLE phrase_seen (state TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO phrase_seen VALUES ('alpha')")
    conn.execute("CREATE TABLE lab_meta (key TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    conn.execute("INSERT INTO lab_meta VALUES ('phrases', 1)")
    conn.commit()
    conn.close()

    assert drop_phrase_seen(path) is True
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'phrase_seen'").fetchone() is None
    assert conn.execute("SELECT n, total FROM phrase_stats").fetchone() == (2, 1.5)
    assert conn.execute("SELECT n FROM lab_meta WHERE key = 'phrases'").fetchone()[0] == 1
    conn.close()
    assert drop_phrase_seen(path) is False


def test_compact_embeddings_rewrites_legacy_vectors(tmp_path):
    path = tmp_path / "embeddings.db"
    vec = np.linspace(-1, 1, LEGACY_F32_BYTES // 4, dtype=np.float32)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE embeddings (model TEXT, text TEXT, vec BLOB, PRIMARY KEY (model, text))")
    conn.execute("INSERT INTO embeddings VALUES ('baai/bge-m3', 'alpha', ?)", (vec.tobytes(),))
    conn.commit()
    conn.close()

    assert compact_embeddings(path, log=lambda _message: None) is True
    conn = sqlite3.connect(path)
    blob = conn.execute("SELECT vec FROM embeddings WHERE text = 'alpha'").fetchone()[0]
    done = conn.execute("SELECT n FROM embedding_codec WHERE key = 'done'").fetchone()[0]
    assert conn.execute("PRAGMA page_size").fetchone()[0] == 32768
    conn.close()
    assert len(blob) == LEGACY_F32_BYTES // 2
    assert done == 1
    assert np.allclose(unpack_vec(blob), vec, atol=1e-3)
    assert compact_embeddings(path, log=lambda _message: None) is False


def test_launch_does_not_vacuum(monkeypatch):
    called = []
    monkeypatch.setattr("jevlab.maintain.run", lambda log=None: called.append(log))
    monkeypatch.setattr("jevlab.cli.cmd_tui", lambda _args: 0)
    from jevlab.cli import main

    assert main([]) == 0
    assert called == []


def test_maintain_command_runs_the_job(monkeypatch):
    called = []
    monkeypatch.setattr("jevlab.maintain.run", lambda log=None: called.append(True))
    from jevlab.cli import main

    assert main(["maintain"]) == 0
    assert called == [True]
