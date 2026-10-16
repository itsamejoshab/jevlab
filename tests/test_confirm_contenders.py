"""Lines that could still beat the score to beat get one more roll before any new guesses.

Each pass adds a single roll and checks again. A line that falls out of range stops. Once a line is
confirmed, later lines are judged against that line, with oracle noise allowed for a short sample.
"""

import asyncio

from jevlab.oracle import Score
from jevlab.search.scheduler import Scheduler

from test_syllogism import engine


class _HoldingOracle:
    def __init__(self):
        self.calls = 0
        self.qkey = "k"
        self.batches: list[tuple[list[str], int]] = []

    async def score(self, states, n=1):
        self.batches.append((list(states), n))
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
    assert [n for _, n in oracle.batches] == [2, 3, 4, 5]
    assert all(set(phrases) == set(beaters.values()) for phrases, _ in oracle.batches)


def test_a_confirmed_line_becomes_the_score_to_beat(tmp_path):
    eng = engine(tmp_path, "ours", leader=(0.92, 12))
    sure = " ".join(["sure"] + ["alpha"] * 20)
    eng.archive.add(sure, Score(sure, [0.936] * 5), "surrogate_bo")
    behind = " ".join(["behind"] + ["beta"] * 29)
    eng.archive.add(behind, Score(behind, [0.93]), "surrogate_bo")
    ahead = " ".join(["ahead"] + ["gamma"] * 19)
    eng.archive.add(ahead, Score(ahead, [0.95]), "surrogate_bo")
    shorter = " ".join(["short"] + ["delta"] * 17)
    eng.archive.add(shorter, Score(shorter, [0.936]), "surrogate_bo")
    # One roll at 0.930 displays under 0.936, but the oracle noise can still carry it over that line.
    close = " ".join(["close"] + ["epsilon"] * 17)
    eng.archive.add(close, Score(close, [0.930]), "surrogate_bo")
    far = " ".join(["far"] + ["zeta"] * 9)
    eng.archive.add(far, Score(far, [0.90]), "surrogate_bo")

    batch = {cand.phrase for cand in eng.contender_batch()}

    assert ahead in batch
    assert shorter in batch
    assert close in batch
    assert behind not in batch
    assert far not in batch
    assert sure not in batch


def test_each_confirm_adds_one_roll_and_drops_a_line_that_falls_out_of_range(tmp_path):
    eng = engine(tmp_path, "step", leader=(0.92, 12))
    sure = " ".join(["sure"] + ["alpha"] * 23)
    eng.archive.add(sure, Score(sure, [0.95] * 5), "surrogate_bo")
    close = " ".join(["close"] + ["beta"] * 17)
    eng.archive.add(close, Score(close, [0.944]), "surrogate_bo")
    steady = " ".join(["steady"] + ["gamma"] * 15)
    eng.archive.add(steady, Score(steady, [0.96]), "surrogate_bo")

    class Oracle:
        def __init__(self):
            self.calls = 0
            self.qkey = "k"
            self.asked: list[tuple[list[str], int]] = []

        async def score(self, states, n=1):
            self.asked.append((list(states), n))
            self.calls += len(states)
            out = {}
            for state in states:
                if state == close:
                    out[state] = Score(state, [0.944, 0.70])
                else:
                    out[state] = Score(state, [0.96] * n)
            return out

    eng.oracle = Oracle()
    assert asyncio.run(eng.confirm_contenders())
    assert [n for _, n in eng.oracle.asked] == [2]
    assert eng.archive.items[close].score.n == 2
    assert eng.archive.items[steady].score.n == 2

    eng.oracle.asked.clear()
    assert asyncio.run(eng.confirm_contenders())
    assert close not in {phrase for phrases, _ in eng.oracle.asked for phrase in phrases}
    assert [n for _, n in eng.oracle.asked] == [3]
    assert eng.archive.items[steady].score.n == 3
    assert eng.archive.items[close].score.n == 2


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
