from jevlab import vault
from jevlab.boards import standings
from jevlab.modes import is_searchable
from jevlab.objective import SHORTEST_THRESHOLD
from jevlab.oracle import Score
from jevlab.search.engine import Engine
from jevlab.search.prompts import target_line

from helpers import make_db


def test_shortest_yes_qualifies_at_the_board_line(tmp_path):
    db = make_db(tmp_path)
    db.execute("UPDATE questions SET yes_threshold = 0.15 WHERE slug = 'q'")
    engine = Engine(db, "q", use_llm=False, board="shortestYes")
    assert engine.objective.threshold == 0.15
    assert engine.objective.qualifies(0.15)
    assert not engine.objective.qualifies(0.14)
    assert engine.objective.beats(0.20, 2, engine.leader)
    assert not engine.objective.beats(0.14, 1, engine.leader)

    db.execute("UPDATE questions SET yes_threshold = NULL WHERE slug = 'q'")
    plain = Engine(db, "q", use_llm=False, board="shortestYes")
    assert plain.objective.threshold == SHORTEST_THRESHOLD

    high = tmp_path / "high"
    high.mkdir()
    db2 = make_db(high)
    db2.execute("UPDATE questions SET yes_threshold = 0.15 WHERE slug = 'q'")
    highest = Engine(db2, "q", use_llm=False, board="highScores")
    assert highest.objective.threshold == SHORTEST_THRESHOLD


def test_scale_questions_join_the_existing_boards(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    assert is_searchable("score")
    assert not is_searchable("choice", ranked=True)

    db = make_db(tmp_path, kind="score", leader=(0.4, 3))
    db.execute(
        "UPDATE questions SET yes_threshold = ?, title = ?, raw = ? WHERE slug = 'q'",
        (7 / 9, "Who is more likely to grill meat: man or woman?",
         '{"levels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]}'),
    )
    highest = Engine(db, "q", use_llm=False)
    assert highest.objective.kind == "score"
    assert highest.objective.p(Score("woman grills", [0.8, 0.8])) == 0.8

    short = Engine(db, "q", use_llm=False, board="shortestYes")
    assert abs(short.objective.threshold - 7 / 9) < 1e-9
    assert short.objective.qualifies(7 / 9)
    assert not short.objective.qualifies(0.7)

    rows = standings(db)
    assert [(row.kind, row.searchable, row.should_search) for row in rows] == [("score", True, True)]

    line = target_line(short.ctx)
    assert "scale" in line.casefold()
    assert "10" in line
    assert "0.78" in line
