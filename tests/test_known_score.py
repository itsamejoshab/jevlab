"""A site score replaces the local estimate when deciding whether to submit again."""

import json

from jevlab import vault
from jevlab.boards import standings
from jevlab.crosspost import picks
from jevlab.db import DB
from jevlab.modes import GOLF, GOLF_HIGHEST, HIGH_SCORES, STRICT

PHRASE = "Accounts blur Ledgers Zig debit credit two phase pending transfers Pick"
FRESH = "edge sqlite filesystem"
MEASURED = "filesystem Fly"


def _db(tmp_path) -> DB:
    db = DB(tmp_path / "t.db")
    db.execute("INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES ('t', 'clef', 'me', '')")
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Edge sqlite. Pick:', '', 'noul', 'yes', 0.2, 0.5, 'm', '{}', '{}')",
        ("edge",),
    )
    ours = json.dumps(
        [
            {
                "userId": "me",
                "name": "Zach",
                "phrase": "Ledgers debit credit two phase pending",
                "probability": 0.74,
                "wordCount": 38,
            }
        ]
    )
    db.execute("INSERT INTO boards VALUES ('edge', 'golf', 'highScores', 0, ?)", (ours,))
    db.execute("INSERT INTO boards VALUES ('edge', 'strict_chain', 'highScores', 0, ?)", (ours,))
    return db


def _save(slug: str, mode: str, phrase: str, p: float, units: int) -> None:
    vault.save(
        slug,
        mode,
        phrase,
        p_mean=p,
        p_lcb=p,
        spread=0.01,
        n=5,
        units=units,
        leader=None,
        beats=True,
        title=slug,
        board=HIGH_SCORES,
    )


def test_a_losing_site_score_is_not_offered_again_as_the_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = _db(tmp_path)
    _save("edge", STRICT, PHRASE, 0.944, 71)
    _save("edge", STRICT, FRESH, 0.80, 22)
    _save("edge", GOLF, PHRASE, 0.944, 71)
    _save("edge", STRICT, MEASURED, 0.50, 14)
    _save("edge", GOLF, MEASURED, 0.50, 14)
    vault.set_status("edge", GOLF, PHRASE, "failed", HIGH_SCORES, server_p=0.0)
    vault.set_status("edge", STRICT, PHRASE, "failed", HIGH_SCORES, server_p=0.0)
    vault.set_status("edge", GOLF, MEASURED, "failed", HIGH_SCORES, server_p=0.99)

    golf = picks(db, modes=(GOLF_HIGHEST,), slugs=["edge"])
    assert [pick.entry["phrase"] for pick in golf] == [MEASURED]

    rows = {row.slug: row for row in standings(db, mode=STRICT, board=HIGH_SCORES)}
    assert rows["edge"].best_entry["phrase"] == FRESH
    assert rows["edge"].entry_beats is True

    _save("quiet", STRICT, PHRASE, 0.944, 71)
    db.execute("INSERT INTO questions VALUES ('quiet', 0, 'r1', 'Quiet', '', 'noul', 'yes', 0.2, 0.5, 'm', '{}', '{}')")
    db.execute("INSERT INTO boards VALUES ('quiet', 'strict_chain', 'highScores', 0, '[]')")
    vault.set_status("quiet", STRICT, PHRASE, "failed", HIGH_SCORES, server_p=0.0)
    quiet = {row.slug: row for row in standings(db, mode=STRICT, board=HIGH_SCORES)}
    assert quiet["quiet"].entry_beats is False
