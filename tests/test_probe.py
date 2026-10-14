"""A search asks every generator model the same five tactics, then stays with the winner.

The winner writes every later tactic. A flat search asks again: after eight of that model's
batches and six rounds without a better line, or after four and four when half the tactics
are already covered. That stall locks a different model when another one scored. A model that
has gone cold, or a newly unlocked tier, asks immediately and may keep the same writer.
"""

import asyncio
import time

from jevlab.oracle import Score
from jevlab.search.probe import PROBE_TACTICS, Lock, choose, due, mean_top, searched_out
from jevlab.search.prompts import ARCHETYPE_GUIDE
from jevlab.search.strategies import COVERAGE_TRIES, LLMGenerate

from test_syllogism import engine


def test_the_probe_score_is_the_mean_of_the_best_three_lines():
    assert mean_top([]) == 0.0
    assert mean_top([0.5]) == 0.5
    assert mean_top([0.1, 0.2, 0.9, 0.8]) == (0.9 + 0.8 + 0.2) / 3


def test_the_best_model_wins_and_a_near_tie_goes_to_the_earlier_one():
    pool = ["cheap", "mid", "dear"]
    assert PROBE_TACTICS == ("rename", "scene", "syllogism", "direct", "wild")
    assert choose({"cheap": 0.40, "mid": 0.90, "dear": 0.70}, pool) == "mid"
    # 0.02 under the best still counts as a tie, and the earlier pool entry is the cheaper one.
    assert choose({"cheap": 0.93, "mid": 0.90, "dear": 0.95}, pool) == "cheap"
    assert choose({"cheap": 0.90, "dear": 0.95}, ["cheap", "dear"]) == "dear"
    assert choose({"cheap": 0.0, "dear": 0.10}, pool) == "dear"
    assert choose({"cheap": 0.0, "dear": 0.0}, pool) is None
    assert choose({}, []) is None


def test_a_board_is_searched_out_once_half_its_tactics_are_covered():
    tactics = ["rename", "scene", "direct", "wild"]
    assert searched_out({"rename": COVERAGE_TRIES}, tactics, COVERAGE_TRIES) is False
    assert searched_out({"rename": COVERAGE_TRIES, "scene": COVERAGE_TRIES}, tactics, COVERAGE_TRIES) is True
    assert searched_out({"rename": COVERAGE_TRIES - 1, "scene": COVERAGE_TRIES}, tactics, COVERAGE_TRIES) is False


def test_the_lock_holds_while_the_search_moves_and_releases_on_a_stall():
    assert due(Lock(), stall=0, exhausted=False, cold=False, force=False)
    assert not due(Lock("mid", batches=1), stall=0, exhausted=False, cold=False, force=False)
    assert not due(Lock("mid", batches=7), stall=6, exhausted=False, cold=False, force=False)
    assert due(Lock("mid", batches=8), stall=6, exhausted=False, cold=False, force=False)
    assert not due(Lock("mid", batches=8), stall=5, exhausted=False, cold=False, force=False)
    assert not due(Lock("mid", batches=3), stall=4, exhausted=True, cold=False, force=False)
    assert due(Lock("mid", batches=4), stall=4, exhausted=True, cold=False, force=False)
    assert not due(Lock("mid", batches=8), stall=0, exhausted=True, cold=False, force=False)
    assert due(Lock("mid", batches=0), stall=0, exhausted=False, cold=True, force=False)
    assert due(Lock("mid", batches=0), stall=0, exhausted=False, cold=False, force=True)


class _Writer:
    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    async def json(self, model, system, user, temperature=1.0, max_tokens=4000, attempts=3, reasoning=None):
        self.prompts.append((model, user))
        who = model.split("/")[-1]
        return {"phrases": [{"tactic": "rename", "text": f"{who} writes a fresh line"}]}


def _install(eng, writer, scores):
    eng.llm = writer

    async def evaluate(items, n=1):
        out = []
        for text, origin, parent, tactic in items:
            who = text.split()[0]
            cand, _ = eng.archive.add(text, Score(text, [scores[who]]), origin, parent, tactic)
            out.append(cand)
        return out

    eng.ctx.evaluate = evaluate


def _cover(eng):
    names = list(ARCHETYPE_GUIDE)
    need = (len(names) + 1) // 2
    for name in names[:need]:
        for i in range(COVERAGE_TRIES):
            eng.archive.add(
                f"{name} covered line {i}", Score(f"{name} covered line {i}", [0.2]), "seed", archetype=name
            )


def _stop_prefetch(gen):
    gen.cancel()
    gen.pending = None


def test_the_opening_batch_asks_every_model_then_later_batches_stay_there(tmp_path):
    eng = engine(tmp_path, "probe")
    writer = _Writer()
    # dear leads by 0.01, inside the 0.02 band, so the earlier model keeps the question.
    scores = {"cheap": 0.90, "dear": 0.91, "sonnet": 0.20}
    _install(eng, writer, scores)
    gen = LLMGenerate(["cheap", "dear"])
    prem = LLMGenerate(["sonnet"], tier=2)
    eng.strategies = {gen.name: gen, prem.name: prem}
    eng.level = 4

    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    probe_users = [user for _, user in writer.prompts if "Write 8 new" in user]
    probe_asks = {model for model, user in writer.prompts if "Write 8 new" in user}
    assert probe_asks == {"cheap", "dear", "sonnet"}
    assert probe_users and all(all(f"[{name}]" in user for name in PROBE_TACTICS) for user in probe_users)
    assert "cheap writes a fresh line" in eng.archive.items
    assert "dear writes a fresh line" in eng.archive.items
    assert "sonnet writes a fresh line" in eng.archive.items
    assert not prem.available(eng.ctx)
    assert prem.next_model(eng.ctx) is None
    assert gen.cold_until["cheap"] == 0.0

    held = len(writer.prompts)
    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    follow = writer.prompts[held:]
    assert follow and {model for model, _ in follow} == {"cheap"}
    assert all("Write 8 new" not in user for _, user in follow)


def test_a_flat_search_on_a_covered_board_asks_again_and_can_switch(tmp_path):
    eng = engine(tmp_path, "reprobe")
    writer = _Writer()
    scores = {"cheap": 0.40, "dear": 0.91}
    _install(eng, writer, scores)
    gen = LLMGenerate(["cheap", "dear"])
    eng.strategies = {gen.name: gen}

    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    _cover(eng)
    eng.stall = 4
    for _ in range(4):
        asyncio.run(gen.run(eng.ctx))
        _stop_prefetch(gen)
    opening = [model for model, user in writer.prompts if "Write 8 new" in user]
    assert set(opening) == {"cheap", "dear"}
    assert len(opening) == 2
    later = [model for model, user in writer.prompts if "Write 8 new" not in user]
    assert later and set(later) == {"dear"}

    scores["cheap"] = 0.97
    scores["dear"] = 0.10
    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    again = [model for model, user in writer.prompts if "Write 8 new" in user]
    assert set(again) == {"cheap", "dear"}
    assert len(again) == 4
    held = len(writer.prompts)
    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    switch = writer.prompts[held:]
    assert switch and {model for model, _ in switch} == {"cheap"}
    assert all("Write 8 new" not in user for _, user in switch)


def _stall_until_due(eng, gen):
    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    _cover(eng)
    eng.stall = 4
    for _ in range(4):
        asyncio.run(gen.run(eng.ctx))
        _stop_prefetch(gen)


def test_a_stall_hands_the_question_to_the_best_other_model(tmp_path):
    eng = engine(tmp_path, "swap")
    writer = _Writer()
    # dear stays well ahead of mid, and mid stays well ahead of cheap. A stall still has to leave dear.
    scores = {"cheap": 0.40, "mid": 0.80, "dear": 0.91}
    _install(eng, writer, scores)
    gen = LLMGenerate(["cheap", "mid", "dear"])
    eng.strategies = {gen.name: gen}

    _stall_until_due(eng, gen)
    assert eng.probe_lock.model == "dear"

    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    asked = [model for model, user in writer.prompts if "Write 8 new" in user]
    assert set(asked) == {"cheap", "mid", "dear"}
    assert len(asked) == 6
    assert eng.probe_lock.model == "mid"

    held = len(writer.prompts)
    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    follow = writer.prompts[held:]
    assert follow and {model for model, _ in follow} == {"mid"}
    assert all("Write 8 new" not in user for _, user in follow)


def test_a_stall_keeps_the_model_when_nobody_else_scored(tmp_path):
    eng = engine(tmp_path, "stay")
    writer = _Writer()
    scores = {"cheap": 0.0, "dear": 0.91}
    _install(eng, writer, scores)
    gen = LLMGenerate(["cheap", "dear"])
    eng.strategies = {gen.name: gen}

    _stall_until_due(eng, gen)
    assert eng.probe_lock.model == "dear"

    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    assert eng.probe_lock.model == "dear"


def test_a_cold_writer_asks_again_on_the_next_batch(tmp_path):
    eng = engine(tmp_path, "cold")
    writer = _Writer()
    scores = {"cheap": 0.20, "dear": 0.80}
    _install(eng, writer, scores)
    gen = LLMGenerate(["cheap", "dear"])
    eng.strategies = {gen.name: gen}

    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    gen.cold_until["dear"] = time.monotonic() + 500
    asyncio.run(gen.run(eng.ctx))
    _stop_prefetch(gen)
    asked = [model for model, user in writer.prompts if "Write 8 new" in user]
    assert set(asked) == {"cheap", "dear"}
    assert len(asked) == 4


def test_a_newly_unlocked_tier_is_included_the_next_time_we_ask(tmp_path):
    eng = engine(tmp_path, "tier")
    writer = _Writer()
    scores = {"cheap": 0.80, "dear": 0.30, "sonnet": 0.20}
    _install(eng, writer, scores)
    gen = LLMGenerate(["cheap", "dear"])
    prem = LLMGenerate(["sonnet"], tier=1)
    eng.strategies = {gen.name: gen, prem.name: prem}

    async def check():
        await gen.run(eng.ctx)
        _stop_prefetch(gen)
        eng.set_level(2, "plateau")
        for strategy in (gen, prem):
            if strategy.pending is not None:
                await strategy.pending
                strategy.pending = None

    asyncio.run(check())
    asked = [model for model, user in writer.prompts if "Write 8 new" in user]
    assert "sonnet" in asked
