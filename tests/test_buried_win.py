"""A confirmed win can sit under lines whose average beats the leader and whose lower bound does not."""

import asyncio

from jevlab import vault
from jevlab.oracle import Score
from jevlab.search.engine import Engine
from jevlab.tui.lab import LabScreen, QueueItem

from helpers import make_db


def test_a_win_under_ties_stays_a_win_and_is_vaulted(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    eng = Engine(make_db(tmp_path, leader=(0.95, 30)), "q", use_llm=False)
    for i in range(6):
        words = ["alpha"] * 39 + [f"tie{i}"]
        eng.archive.add(" ".join(words), Score("t", [0.99, 0.99, 0.99, 0.99, 0.99, 0.88]), "extend")
    win = " ".join(["beta"] * 12)
    eng.archive.add(win, Score("w", [0.952] * 6), "local_edit")
    ranked = eng.archive.top_by_board(8)
    assert ranked[0].phrase != win
    assert not eng.objective.wins(ranked[0].score, ranked[0].units, eng.leader)
    assert eng.objective.wins(eng.archive.items[win].score, 12, eng.leader)
    assert win not in {c.phrase for c in eng.archive.top_by_board(5)}

    item = QueueItem("q", "Would you trade your house")
    LabScreen.update_item(item, eng)
    assert item.win

    asyncio.run(eng.promote())
    saved = [entry for entry in vault.all_entries() if entry["phrase"] == win]
    assert saved and saved[0]["beats"]


def test_a_few_rolls_do_not_mark_the_queue_as_a_win(tmp_path):
    eng = Engine(make_db(tmp_path, leader=(0.95, 30)), "q", use_llm=False)
    phrase = " ".join(["gamma"] * 16)
    eng.archive.add(phrase, Score("s", [0.97, 0.97]), "extend")
    assert eng.objective.wins(eng.archive.items[phrase].score, 16, eng.leader)
    item = QueueItem("q", "Is it insufferable")
    LabScreen.update_item(item, eng)
    assert item.best == "0.970/16w"
    assert not item.win
