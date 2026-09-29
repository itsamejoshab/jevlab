import asyncio
import random

from jevlab.oracle import Score
from jevlab.search.archive import Archive
from jevlab.search.engine import Engine
from jevlab.search.mechanical import noise_sd, race
from jevlab.objective import Objective
from helpers import make_db


def test_plateau_needs_a_long_leader_at_the_ceiling(tmp_path):
    db = make_db(tmp_path, leader=(0.99, 86))
    engine = Engine(db, "q", use_llm=False)
    assert not engine.long and engine.plateau_candidate and not engine.plateau
    ordinary_cap = engine.length_cap
    engine.enter_plateau("test")
    assert engine.plateau and engine.length_cap == engine.grow_cap == 85 > ordinary_cap
    assert engine.allowed("extend") and engine.allowed("grow")

    grow = engine.strategies["grow"]
    grow.restore(["a b c", "a b d"])
    assert grow.depth == 3 and grow.state() == ["a b c", "a b d"]
    grow.restore([], {"banned": [0.06, 3], "fluffy": [-0.02, 2], "rare": [0.03, 1]})
    assert grow.best_words(5) == ["banned", "rare"]

    assert not Engine(db, "q", use_llm=False, board="shortestYes").plateau_candidate

    def other(name, leader):
        (tmp_path / name).mkdir()
        return Engine(make_db(tmp_path / name, leader=leader), "q", use_llm=False)

    assert not other("a", (0.98, 86)).plateau_candidate
    short_leader = other("b", (0.99, 20))
    assert not short_leader.plateau_candidate and not short_leader.allowed("extend")


def test_noise_shrinks_toward_the_ends():
    assert noise_sd(0.5) > noise_sd(0.85) > noise_sd(0.98)
    assert noise_sd(0.02) == noise_sd(0.98) < 0.005


class DitherCtx:
    """Every roll is the true value plus noise, rounded to 0.01 like the oracle."""

    def __init__(self, truth: dict[str, float], seed: int = 1):
        self.truth = truth
        self.rng = random.Random(seed)
        self.objective = Objective("yes", "noul")
        self.archive = Archive(self.objective)
        self.calls = 0

    def roll(self, phrase):
        return round(min(self.truth[phrase] + self.rng.gauss(0, 0.004), 0.99), 2)

    async def evaluate(self, items, n=1):
        out = []
        for phrase, origin, parent, archetype in items:
            have = self.archive.items[phrase].score.samples if phrase in self.archive.items else []
            new = [self.roll(phrase) for _ in range(max(n - len(have), 0))]
            self.calls += len(new)
            out.append(self.archive.add(phrase, Score(phrase, have + new), origin, parent, archetype)[0])
        return out


def test_race_finds_the_line_nearest_the_next_step():
    truth = {f"line {i}": 0.977 for i in range(30)}
    truth["line 7"] = 0.9835
    ctx = DitherCtx(truth)
    first = asyncio.run(ctx.evaluate([(p, "t", "", "other") for p in truth]))
    assert len({round(c.p, 2) for c in first}) <= 3  # one roll each: mostly ties
    raced = asyncio.run(race(ctx, first, (2, 4, 8)))
    assert raced[0].phrase == "line 7"
    assert ctx.calls < 30 * 8


def test_drift_starts_from_the_steadiest_top_line_and_walks_toward_steadier_ones(tmp_path):
    engine = Engine(make_db(tmp_path, leader=(0.99, 86)), "q", use_llm=False)
    engine.enter_plateau("test")
    drift = engine.strategies["drift"]
    shaky, steady = "a b c d", "e f g h"
    for line in (shaky, steady):
        engine.archive.add(line, Score(line, [0.98] * 4), "t")
    for i in range(12):
        engine.archive.add(f"{shaky} x{i}", Score("", [0.98 if i < 2 else 0.95]), "t", shaky)
        engine.archive.add(f"{steady} y{i}", Score("", [0.98 if i < 9 else 0.96]), "t", steady)
    assert drift.start(engine.ctx).phrase == steady
    assert drift.hold_rate(engine.ctx, steady, 0.98) == (10 / 14, 12)

    ctx = DitherCtx({})
    ctx.engine = type("E", (), {"grow_cap": 85, "strategies": {}, "remaining": lambda self: 10_000})()
    ctx.word_pool = lambda k: ["calm", "sure", "firm", "true"]
    ctx.climb_roots = lambda k: [ctx.archive.items["p q r"]]
    ctx.archive.top_by_board = lambda k: [ctx.archive.items["p q r"]]
    ctx.log = lambda message: None
    # Lines holding "firm" sit near the top of the 0.98 band, so their own edits keep holding.
    ctx.truth = type("T", (dict,), {"__missing__": lambda self, p: 0.983 if "firm" in p else 0.977})()
    ctx.archive.add("p q r", Score("p q r", [0.98] * 4), "t")
    drift.vocab, drift.line, drift.steps = [], None, 3
    asyncio.run(drift.run(ctx))
    assert "firm" in drift.line


def test_climb_roots_prefer_the_better_mean_over_fewer_words(tmp_path):
    engine = Engine(make_db(tmp_path, leader=(0.99, 86)), "q", use_llm=False)
    short = " ".join(["w"] * 23)
    longer = " ".join(["v"] * 40)
    lucky = " ".join(["u"] * 30)
    engine.archive.add(short, Score(short, [0.98] * 8), "t")
    engine.archive.add(longer, Score(longer, [0.98] * 5 + [0.99] * 3), "t")
    engine.archive.add(lucky, Score(lucky, [0.99]), "t")
    assert engine.archive.top_by_board(1)[0].phrase == lucky
    engine.archive.items.pop(lucky)
    assert engine.archive.top_by_board(1)[0].phrase == short
    assert engine.ctx.climb_roots(1)[0].phrase == longer


def test_one_word_at_one_on_every_roll_is_unbeatable(tmp_path):
    for board in ("highScores", "shortestYes"):
        (tmp_path / board).mkdir()
        engine = Engine(make_db(tmp_path / board), "q", use_llm=False, board=board)
        engine.archive.add("yes please", Score("yes please", [1.0] * 8), "t")
        engine.archive.add("lucky", Score("lucky", [1.0] * 4), "t")
        engine.archive.add("shaky", Score("shaky", [1.0] * 7 + [0.99]), "t")
        assert engine.unbeatable() is None
        engine.archive.add("yes", Score("yes", [0.999] * 5), "t")
        assert engine.unbeatable().phrase == "yes"
