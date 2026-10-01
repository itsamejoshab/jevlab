from jevlab.objective import SHORTEST_THRESHOLD
from jevlab.search.engine import Engine

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
