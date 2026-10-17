"""Vault rescore keeps current-checkpoint samples and rewrites the stored score from those alone."""

import importlib.util
from pathlib import Path

from jevlab.db import DB
from jevlab.modes import HIGH_SCORES, SHORTEST_YES
from jevlab.objective import Leader, Objective
from jevlab.oracle import Score

PATH = Path(__file__).resolve().parents[1] / "scripts" / "rescore_vault.py"
spec = importlib.util.spec_from_file_location("rescore_vault", PATH)
rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore)


def _sample(db: DB, model: str, state: str, noul: float, qkey: str = "qk", request_hash: str = "h") -> None:
    db.execute(
        "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey)"
        " VALUES (?, ?, ?, ?, '{}', 1, 0, 1, ?)",
        (request_hash, model, state, noul, qkey),
    )


def test_keep_current_drops_older_checkpoint_and_rebuilds_the_summary(tmp_path):
    db = DB(tmp_path / "t.db")
    _sample(db, "jev-1.13.0", "hello", 0.9)
    _sample(db, None, "hello", 0.2, qkey=None, request_hash="h")
    _sample(db, "typesafe/jev-1.14-20261014", "hello", 0.4)
    _sample(db, "jev-1.13.0", "other", 0.7, qkey="other", request_hash="other")
    db.execute("INSERT INTO phrase_stats (qkey, state, n, total, sumsq) VALUES ('qk', 'hello', 3, 1.5, 1)")

    rescore.keep_current(db, "qk", "h", "hello")

    left = db.all("SELECT model, state FROM oracle_samples ORDER BY state")
    assert [(row["model"], row["state"]) for row in left] == [
        ("typesafe/jev-1.14-20261014", "hello"),
        ("jev-1.13.0", "other"),
    ]
    stats = db.one("SELECT n, total FROM phrase_stats WHERE qkey = 'qk' AND state = 'hello'")
    assert stats["n"] == 1
    assert stats["total"] == 0.4


def test_apply_entry_uses_the_goal_and_leaves_a_published_flag():
    entry = {"phrase": "hello", "units": 1, "status": "candidate", "beats": True, "board": HIGH_SCORES, "history": []}
    score = Score("hello", [0.2, 0.2])
    rescore.apply_entry(entry, score, Objective("no"), None, HIGH_SCORES)
    assert entry["p_mean"] == 0.8
    assert entry["n"] == 2
    assert entry["checkpoint"] == "1.14"
    assert entry["beats"] is True
    assert entry["history"][-1]["status"] == "rescored"

    published = {"phrase": "hello", "units": 1, "status": "published", "beats": False, "board": HIGH_SCORES}
    rescore.apply_entry(published, score, Objective("no"), Leader(0.9, 1, "Ada"), HIGH_SCORES)
    assert published["beats"] is False
    assert published["p_mean"] == 0.8
    assert published["leader"]["name"] == "Ada"

    rescore.apply_entry(entry, score, Objective("no"), None, HIGH_SCORES)
    assert len(entry["history"]) == 1


def test_apply_entry_marks_a_shortest_gamble():
    entry = {"phrase": "cup", "units": 1, "status": "candidate", "beats": False, "board": SHORTEST_YES, "history": []}
    score = Score("cup", [0.2, 0.2, 0.2, 0.2, 0.8])
    rescore.apply_entry(entry, score, Objective("yes", board=SHORTEST_YES), Leader(0.6, 3, "Ada"), SHORTEST_YES)
    assert entry["gamble"] is True
    assert entry["p_reach"] == 0.8
    assert entry["hits"] == "1/5"
