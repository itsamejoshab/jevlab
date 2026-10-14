"""A deduction frame is tried while the search is still looking for a high score.

The shape is a closed set of rules: the subject belongs to some class, every member of that class gets the
target answer, and so the subject does too. Opening generator batches on Highest reserve one tactic slot for
it. Once those lines have been scored, or the opening batches are over, the slot goes back to the mix.
Shortest boards never reserve it.
"""

from jevlab.oracle import Score
from jevlab.search.archive import ARCHETYPES
from jevlab.search.engine import Engine
from jevlab.search.prompts import ARCHETYPE_GUIDE, gen_user, plan_user
from jevlab.search.strategies import LLMGenerate

from helpers import make_db


def engine(tmp_path, name, board="highScores", leader=(0.99, 40)):
    (tmp_path / name).mkdir()
    return Engine(make_db(tmp_path / name, leader=leader), "q", use_llm=False, board=board)


def picks(gen, ctx, n):
    return [gen.pick_archetypes(ctx) for _ in range(n)]


def test_opening_highest_batches_reserve_a_deduction(tmp_path):
    eng = engine(tmp_path, "high")
    gen = LLMGenerate()
    eng.ctx.focus = ["scene", "rename"]

    eng.archive.add(
        "old scene already scores", Score("old scene already scores", [0.98]), "llm:core", archetype="scene"
    )
    opening = picks(gen, eng.ctx, 6)
    assert all("syllogism" in chosen for chosen in opening)
    assert all("scene" in chosen or "rename" in chosen for chosen in opening)
    assert all(len(chosen) >= 2 for chosen in opening)


def test_reservation_releases_once_deductions_are_scored_or_the_opening_passes(tmp_path):
    eng = engine(tmp_path, "high")
    gen = LLMGenerate()
    for i in range(gen.count):
        eng.archive.add(f"rule line {i} alpha", Score(f"rule line {i} alpha", [0.5]), "llm:core", archetype="syllogism")
    released = picks(gen, eng.ctx, 30)
    assert any("syllogism" not in chosen for chosen in released)

    fresh = engine(tmp_path, "later")
    aged = LLMGenerate()
    aged.round = 8
    aged_picks = picks(aged, fresh.ctx, 30)
    assert any("syllogism" not in chosen for chosen in aged_picks)

    later_tier = picks(LLMGenerate(tier=1), fresh.ctx, 30)
    assert any("syllogism" not in chosen for chosen in later_tier)


def test_shortest_boards_do_not_reserve_the_deduction(tmp_path):
    short = engine(tmp_path, "short", board="shortestYes", leader=(0.7, 4))
    race = engine(tmp_path, "race", board="shortestYes", leader=(0.7, 1))
    for eng in (short, race):
        chosen = picks(LLMGenerate(), eng.ctx, 30)
        assert any("syllogism" not in batch for batch in chosen)


def test_writers_are_shown_the_deduction_shape(tmp_path):
    assert "syllogism" in ARCHETYPES
    guide = ARCHETYPE_GUIDE["syllogism"]
    assert "sponge" in guide and "microwave" in guide
    lowered = guide.casefold()
    assert "always true" in lowered or "always hold" in lowered
    assert "every" in lowered and "class" in lowered
    assert "(3)" in guide or "therefore" in lowered
    # A copied opener is what gets banned, so the guide has to demand a new one instead of quoting a stock phrase.
    assert "banned" in lowered
    assert any(word in lowered for word in ("invent", "new opener", "do not reuse", "vary"))

    eng = engine(tmp_path, "high")
    shown = gen_user(eng.ctx, ["syllogism"], 4)
    assert guide in shown
    stats = {"calls": 0, "size": 0, "arms": {}, "archetypes": {}}
    assert "syllogism" in plan_user(eng.ctx, stats)
