"""Phrase summaries stand in for scanning every oracle sample."""

import json
import sqlite3

from jevlab.boards import long_shot
from jevlab.db import DB
from jevlab.objective import Leader, objective_for
from jevlab.oracle import question_key
from jevlab.search.global_model import scored_phrases


def test_new_samples_update_the_summary_and_the_distinct_count(tmp_path):
    db = DB(tmp_path / "t.db")
    assert scored_phrases(db) == 0
    db.add_oracle_samples(
        [
            ("h1", "m", "alpha beta", 0.8, "{}", 1, 0, 1.0, "q1"),
            ("h1", "m", "alpha beta", 0.6, "{}", 1, 0, 1.0, "q1"),
            ("h2", "m", "gamma", 0.4, "{}", 1, 0, 1.0, "q1"),
        ]
    )
    assert scored_phrases(db) == 2
    row = db.one("SELECT n, total, sumsq FROM phrase_stats WHERE qkey = 'q1' AND state = 'alpha beta'")
    assert row["n"] == 2
    assert row["total"] == 1.4
    db.add_oracle_samples([("h1", "m", "alpha beta", 0.9, "{}", 1, 0, 1.0, "q2")])
    assert scored_phrases(db) == 2
    other = db.one("SELECT n, total FROM phrase_stats WHERE qkey = 'q2' AND state = 'alpha beta'")
    assert other["n"] == 1
    assert other["total"] == 0.9


def test_long_shot_reads_the_summary(tmp_path):
    req = {"questions": {"q": {"type": "noul"}}}
    qkey = question_key(req)
    db = DB(tmp_path / "t.db")
    db.execute("INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES ('t', 'jev', 'me', '')")
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Edge', '', 'noul', 'yes', 0.2, 0.5, 'm', ?, '{}')",
        ("edge", json.dumps(req)),
    )
    phrase = "alpha beta gamma"
    db.add_oracle_samples(
        [
            ("h", "m", phrase, 0.80, "{}", 1, 0, 1.0, qkey),
            ("h", "m", phrase, 0.70, "{}", 1, 0, 2.0, qkey),
        ]
    )
    shot = long_shot(
        db, "edge", "strict_chain", objective_for({"kind": "noul", "goal": "yes"}), Leader(0.72, 40, "Ada"), None, set()
    )
    assert shot is not None
    assert shot["phrase"] == phrase
    assert shot["n"] == 2
    assert shot["p_mean"] == 0.75


def test_embeddings_leave_the_main_database(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.config.DATA", tmp_path)
    path = tmp_path / "jev.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE embeddings (model TEXT, text TEXT, vec BLOB, PRIMARY KEY (model, text))")
    conn.execute("INSERT INTO embeddings VALUES ('m', 'hi', ?)", (b"abcd",))
    conn.commit()
    conn.close()

    db = DB(path)
    assert db.one("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'embeddings'") is None
    side = sqlite3.connect(tmp_path / "embeddings.db")
    row = side.execute("SELECT vec FROM embeddings WHERE model = 'm' AND text = 'hi'").fetchone()
    assert row[0] == b"abcd"
    side.close()

    again = DB(path)
    assert again.one("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'embeddings'") is None
