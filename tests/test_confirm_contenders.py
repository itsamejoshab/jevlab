"""Lines whose displayed score already beats the leader are confirmed before any new guesses.

A batch is the next five in board order that still have fewer than five rolls. The following batch
waits until those five are done. A line under the leader does not hold up the next guess.
"""

import asyncio

from jevlab.oracle import Score
from jevlab.search.scheduler import Scheduler

from test_syllogism import engine


class _HoldingOracle:
    def __init__(self):
        self.calls = 0
        self.qkey = "k"
        self.batches: list[list[str]] = []

    async def score(self, states, n=1):
        self.batches.append(list(states))
        self.calls += len(states) * n
        return {state: Score(state, [0.93] * n) for state in states}

    async def close(self):
        pass


class _Guesser:
    name = "surrogate_bo"

    def __init__(self):
        self.runs = 0
        self.n_at_guess: dict[str, int] = {}
        self.calls_at_guess = 0

    def available(self, ctx):
        return True

    async def run(self, ctx):
        self.runs += 1
        self.calls_at_guess = ctx.engine.oracle.calls
        self.n_at_guess = {cand.phrase: cand.score.n for cand in ctx.engine.archive.items.values()}
        ctx.engine.stopping = True


def _phrase(units: int, tag: str) -> str:
    return " ".join([tag] + ["alpha"] * (units - 1))


def _run(eng, oracle, guesser):
    eng.strategies = {guesser.name: guesser}
    eng.scheduler = Scheduler([guesser.name])

    async def start():
        eng.oracle = oracle

    eng.start = start
    asyncio.run(eng.run())


def test_possible_winners_are_confirmed_in_batches_of_five_before_new_guesses(tmp_path):
    eng = engine(tmp_path, "confirm", leader=(0.92, 12))
    beaters = {}
    for units in (30, 28, 26, 24, 22, 20, 18, 16):
        phrase = _phrase(units, f"w{units}")
        eng.archive.add(phrase, Score(phrase, [0.93]), "surrogate_bo")
        beaters[units] = phrase
    weak = _phrase(12, "weak")
    eng.archive.add(weak, Score(weak, [0.90]), "surrogate_bo")
    oracle = _HoldingOracle()
    guesser = _Guesser()

    _run(eng, oracle, guesser)

    assert guesser.runs == 1
    assert all(guesser.n_at_guess[phrase] >= 5 for phrase in beaters.values())
    assert guesser.n_at_guess[weak] == 1
    assert [len(batch) for batch in oracle.batches] == [5, 3]
    assert set(oracle.batches[0]) == {beaters[units] for units in (16, 18, 20, 22, 24)}


def test_a_line_under_the_leader_does_not_delay_the_next_guess(tmp_path):
    eng = engine(tmp_path, "under", leader=(0.92, 12))
    weak = _phrase(12, "weak")
    eng.archive.add(weak, Score(weak, [0.90]), "surrogate_bo")
    held = _phrase(16, "held")
    eng.archive.add(held, Score(held, [0.93] * 5), "surrogate_bo")
    oracle = _HoldingOracle()
    guesser = _Guesser()

    _run(eng, oracle, guesser)

    assert guesser.runs == 1
    assert guesser.calls_at_guess == 0
    assert guesser.n_at_guess[weak] == 1
