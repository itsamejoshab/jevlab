"""The writer catalog covers every frame that moves a classifier, and a hot frame cannot starve the rest.

Opening Highest batches still reserve a deduction. Coverage only swaps a slot once every tactic already
in the batch has scored lines, and it will not evict that reserved deduction.
"""

from jevlab.oracle import Score
from jevlab.search.archive import ARCHETYPES
from jevlab.search.prompts import ARCHETYPE_GUIDE, fragments_user, gen_system, plan_user
from jevlab.search.strategies import COVERAGE_TRIES, LLMGenerate

from test_syllogism import engine


def test_catalog_matches_the_guide():
    assert set(ARCHETYPE_GUIDE) == set(ARCHETYPES) - {"other"}
    assert "other" not in ARCHETYPE_GUIDE
    for name, text in ARCHETYPE_GUIDE.items():
        assert name.isidentifier()
        assert len(text.split()) >= 8


def test_planner_and_writers_see_the_literal_reading_rule(tmp_path):
    eng = engine(tmp_path, "high")
    shown = plan_user(eng.ctx, {"calls": 0, "size": 0, "arms": {}, "archetypes": {}})
    for name in ARCHETYPE_GUIDE:
        assert name in shown
    system = gen_system(eng.ctx)
    assert "literally" in system
    assert "double negative" in system
    fragments = fragments_user(eng.ctx, 4)
    assert sum(1 for name in ARCHETYPE_GUIDE if f"[{name}]" in fragments) == 8


def test_a_covered_batch_yields_one_slot_to_an_untried_tactic(tmp_path):
    eng = engine(tmp_path, "covered", board="shortestYes", leader=(0.7, 4))
    gen = LLMGenerate()
    hot = ["rename", "scene", "direct"]
    for name in hot:
        for i in range(COVERAGE_TRIES + 2):
            phrase = f"{name} covered line {i} alpha"
            eng.archive.add(phrase, Score(phrase, [0.99]), "llm:core", archetype=name)
    batches = [gen.pick_archetypes(eng.ctx) for _ in range(20)]
    assert all(any(name not in hot for name in chosen) for chosen in batches)


def test_coverage_does_not_drop_a_reserved_deduction(tmp_path):
    eng = engine(tmp_path, "held")
    gen = LLMGenerate()
    eng.ctx.focus = ["scene", "rename"]
    for i in range(COVERAGE_TRIES + 1):
        phrase = f"rule line {i} alpha"
        eng.archive.add(phrase, Score(phrase, [0.4]), "llm:core", archetype="syllogism")
    for name in ("scene", "rename", "direct", "eager"):
        for i in range(COVERAGE_TRIES + 4):
            phrase = f"{name} hot line {i} beta"
            eng.archive.add(phrase, Score(phrase, [0.99]), "llm:core", archetype=name)
    opening = [gen.pick_archetypes(eng.ctx) for _ in range(6)]
    assert all("syllogism" in chosen for chosen in opening)
