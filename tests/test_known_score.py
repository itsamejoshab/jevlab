"""A site score replaces the local estimate when deciding whether to submit again."""

import json

from jevlab import vault
from jevlab.boards import standings
from jevlab.crosspost import casual_picks, picks
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


def test_casual_proxy_offers_the_strict_highest_line_where_it_takes_the_board(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    db.execute("INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES ('t', 'luna', 'me', '')")
    for slug, title in (("open", "Open"), ("short", "Short leader"), ("tied", "Tied ceiling")):
        db.execute(
            "INSERT INTO questions VALUES (?, 0, 'r1', ?, '', 'noul', 'yes', 0.2, 0.5, 'm', '{}', '{}')",
            (slug, title),
        )
    short_leader = json.dumps(
        [{"userId": "ada", "name": "Ada", "phrase": "yes please", "probability": 0.62, "wordCount": 2}]
    )
    ceiling = json.dumps(
        [{"userId": "ada", "name": "Ada", "phrase": "yes", "probability": 0.99, "wordCount": 1}]
    )
    db.execute("INSERT INTO boards VALUES ('short', 'word_chain', 'highScores', 0, ?)", (short_leader,))
    db.execute("INSERT INTO boards VALUES ('short', 'word_chain', 'shortestYes', 0, ?)", (short_leader,))
    db.execute("INSERT INTO boards VALUES ('tied', 'word_chain', 'highScores', 0, ?)", (ceiling,))
    db.execute("INSERT INTO boards VALUES ('tied', 'word_chain', 'shortestYes', 0, ?)", (ceiling,))
    _save("open", STRICT, "one two", 0.80, 2)
    _save("open", STRICT, "alpha beta gamma delta epsilon zeta eta theta", 0.95, 8)
    _save("short", STRICT, "alpha beta gamma delta epsilon zeta eta theta", 0.95, 8)
    _save("tied", STRICT, "alpha beta gamma delta epsilon zeta eta theta", 0.99, 8)

    found = {(pick.slug, pick.mode.name, pick.entry["phrase"]) for pick in casual_picks(db)}
    assert ("open", "casual-highest", "alpha beta gamma delta epsilon zeta eta theta") in found
    assert ("open", "casual-shortest", "alpha beta gamma delta epsilon zeta eta theta") in found
    assert ("short", "casual-highest", "alpha beta gamma delta epsilon zeta eta theta") in found
    assert not any(slug == "short" and mode == "casual-shortest" for slug, mode, _ in found)
    assert not any(slug == "tied" for slug, mode, _ in found)
    assert not any(phrase == "one two" for _, _, phrase in found)
