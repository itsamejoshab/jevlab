"""Import copies one measured Strict line per board, and nothing else."""

import json

from jevlab import vault
from jevlab.db import DB
from jevlab.modes import GOLF, HIGH_SCORES, SHORTEST_YES, STRICT
from jevlab.transfer import import_from


def _question(db: DB, slug: str, goal: str = "yes", kind: str = "noul", raw: dict | None = None) -> None:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', ?, '', ?, ?, 0.2, 0.5, 'm', '{}', ?)",
        (slug, slug, kind, goal, json.dumps(raw or {})),
    )


def _snapshot(db: DB) -> None:
    db.execute("INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES ('t', 'laya', '', '')")


def _save(
    phrase: str,
    p: float,
    board: str = HIGH_SCORES,
    status: str = "candidate",
    mode: str = STRICT,
    n: int = 5,
    estimated_from: str = "",
    target: str = "",
    slug: str = "q",
) -> None:
    vault.save(
        slug,
        mode,
        phrase,
        p_mean=p,
        p_lcb=p - 0.02,
        spread=0.01,
        n=n,
        units=len(phrase.split()),
        leader=None,
        beats=True,
        title=slug,
        board=board,
        target=target,
        estimated_from=estimated_from,
    )
    if status != "candidate":
        vault.set_status(slug, mode, phrase, status, board, target=target)


def test_import_from_keeps_the_best_measured_line_on_each_board(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.transfer.EDITION", "laya")
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    monkeypatch.setattr(vault, "VAULT", source)
    _save("alpha beta", 0.90)
    _save("worse line", 0.40)
    _save("failed crown", 0.99, status="failed")
    _save("borrowed elsewhere", 0.95, estimated_from="jev", n=0)
    _save("golf phrase", 0.99, mode=GOLF)
    _save("short winner", 0.80, board=SHORTEST_YES)
    monkeypatch.setattr(vault, "VAULT", dest)
    dest_db = DB(tmp_path / "dest.db")
    _snapshot(dest_db)
    _question(dest_db, "q")
    source_db = DB(tmp_path / "source.db")
    _question(source_db, "q")

    found = import_from(dest_db, "kev", source_vault=source, source_db=source_db, log=lambda *_: None)

    phrases = {(item.entry["phrase"], item.entry["board"]) for item in found}
    assert phrases == {("alpha beta", HIGH_SCORES), ("short winner", SHORTEST_YES)}
    saved = vault.all_entries()
    assert all(entry.get("estimated_from") == "kev" and entry["n"] == 0 for entry in saved)
    assert all(entry["status"] != "queued" for entry in saved)


def test_import_from_skips_a_different_goal_and_a_line_already_measured(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.transfer.EDITION", "laya")
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    monkeypatch.setattr(vault, "VAULT", source)
    _save("flipped goal", 0.99, slug="flip")
    _save("best here", 0.99, slug="kept")
    _save("second here", 0.70, slug="kept")
    monkeypatch.setattr(vault, "VAULT", dest)
    _save("best here", 0.50, slug="kept", n=8)
    dest_db = DB(tmp_path / "dest.db")
    _snapshot(dest_db)
    _question(dest_db, "flip", goal="no")
    _question(dest_db, "kept")
    source_db = DB(tmp_path / "source.db")
    _question(source_db, "flip", goal="yes")
    _question(source_db, "kept")

    found = import_from(dest_db, "kev", source_vault=source, source_db=source_db, log=lambda *_: None)

    assert [item.entry["phrase"] for item in found] == ["second here"]
    kept = vault.load("kept", STRICT)["entries"]
    measured = next(entry for entry in kept if entry["phrase"] == "best here")
    assert measured["n"] == 8
    assert not measured.get("estimated_from")
    assert vault.load("flip", STRICT)["entries"] == []
