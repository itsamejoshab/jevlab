import asyncio
import math
import random

from jevlab.objective import Objective
from jevlab.oracle import Score
from jevlab.rules import check_phrase
from jevlab.search.archive import Archive
from jevlab.search.chains import RECIPES, ChainCompact, ChainLibrary, Probe, split_phrase
from jevlab.search.engine import Engine
from jevlab.search.strategies import WORD_STAGE

from helpers import make_db


def long_objective(**kw) -> Objective:
    return Objective("yes", "noul", long_units=300, **kw)


def test_regime_only_for_highest_yes_no(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.search.engine.LONG_TARGET", 300)
    db = make_db(tmp_path)
    engine = Engine(db, "q", use_llm=False)
    assert engine.long and engine.objective.long
    assert engine.length_cap == 300 and engine.grow_cap == 300
    assert engine.objective.ceiling == 0.99
    assert "chain:stack" in engine.strategies and engine.ctx.chains is not None
    assert not engine.allowed("single_word") and not engine.allowed("compress")

    shortest = Engine(db, "q", use_llm=False, board="shortestYes")
    assert not shortest.long and not shortest.objective.long

    db.execute("UPDATE questions SET kind = 'choice'")
    assert not Engine(db, "q", use_llm=False).long


def test_ceiling_hold():
    obj = long_objective()
    assert obj.holds_ceiling(Score("x", [0.99] * 5))
    assert not obj.holds_ceiling(Score("x", [0.99, 0.99, 0.98, 0.99, 0.99]))
    assert not obj.holds_ceiling(Score("x", [0.99] * 4))
    no = Objective("no", "noul", long_units=300)
    assert no.holds_ceiling(Score("x", [0.01] * 5))
    assert obj.holds(Score("x", [0.99, 0.99]), 0.99, 2)
    assert not obj.holds(Score("x", [0.99, 0.98]), 0.99, 2)
    assert obj.holds(Score("x", [0.95, 0.96]), 0.95, 2)


def test_long_fitness_prefers_short_lines_at_the_ceiling():
    obj = long_objective()
    long_top = obj.fitness(Score("x", [0.99]), " ".join(["w"] * 250))
    short_top = obj.fitness(Score("x", [0.99]), " ".join(["w"] * 40))
    below = obj.fitness(Score("x", [0.97]), " ".join(["w"] * 10))
    assert short_top > long_top > below


def library(rng=None) -> ChainLibrary:
    lib = ChainLibrary([], 300, 300, rng or random.Random(3))
    for i in range(30):
        lib.add(" ".join(f"word{i}x{j}" for j in range(6 + i % 8)), "archive", cluster=i % 6, min_units=3)
    lib.add("yes absolutely", "booster")
    return lib


def test_recipes_stay_strict_valid_within_cap():
    lib = library()
    scored = [(lib.recipe("stack"), 0.5 + i / 100) for i in range(12)]
    for recipe in RECIPES:
        for _ in range(20):
            ids = lib.recipe(recipe, scored)
            phrase = lib.compose(ids)
            assert ids and phrase
            assert check_phrase(phrase, [], 300) == phrase
            assert len(phrase.split()) <= 300


def test_split_rejoins():
    phrase = " ".join(f"w{i}" for i in range(40)) + " and then " + " ".join(f"v{i}" for i in range(20))
    pieces = split_phrase(phrase)
    assert " ".join(pieces) == phrase
    assert all(len(p.split()) <= 12 for p in pieces)


def test_attribution_finds_planted_fragment():
    rng = random.Random(5)
    lib = library(rng)
    strong = lib.add("the magic keyword is here", "archive", cluster=0, min_units=3)
    archive = Archive(long_objective())
    for _ in range(120):
        ids = lib.stack(120)
        phrase = lib.register(ids)
        z = -0.5 + 2.0 * ids.count(strong.id) + rng.gauss(0, 0.3)
        archive.add(phrase, Score(phrase, [1 / (1 + math.exp(-z))]), "test")
    assert lib.fit(archive) >= 100
    assert lib.top_fragments(1) == [strong.text]


def test_novelty_compares_against_earlier_chains_only():
    lib = library(random.Random(9))
    archive = Archive(long_objective())
    archive.add("short old line", Score("short old line", [0.98] * 5), "history")
    earlier = lib.register([f.id for f in lib.frags if f.cluster == 0])
    archive.add(earlier, Score(earlier, [0.97]), "test")
    same = lib.register([f.id for f in lib.frags if f.cluster == 0][::-1])
    other = lib.register([f.id for f in lib.frags if f.cluster == 3])
    weak = lib.register([f.id for f in lib.frags if f.cluster == 4])
    new = [archive.add(same, Score(same, [0.97]), "t")[0], archive.add(other, Score(other, [0.97]), "t")[0],
           archive.add(weak, Score(weak, [0.80]), "t")[0]]
    assert lib.novel(archive, new) == 1


class FakeOracle:
    calls = 0


class FakeScheduler:
    boost: dict = {}


class FakeEngine:
    def __init__(self, ctx):
        self.ctx = ctx
        self.oracle = FakeOracle()
        self.scheduler = FakeScheduler()

    def remaining(self):
        return 100000

    def roots(self):
        obj = self.ctx.objective
        return [c for c in self.ctx.archive.items.values() if obj.holds_ceiling(c.score)]


class FakeCtx:
    """Scores 0.99 on every roll while the line keeps 'magic keyword', else 0.90."""

    def __init__(self):
        self.objective = long_objective()
        self.archive = Archive(self.objective)
        self.chains = library()
        self.leader = None
        self.engine = FakeEngine(self)

    def log(self, message):
        pass

    async def evaluate(self, items, n=1):
        out = []
        for phrase, origin, parent, archetype in items:
            value = 0.99 if "magic keyword" in phrase else 0.90
            have = self.archive.items[phrase].score.samples if phrase in self.archive.items else []
            samples = have + [value] * max(n - len(have), 0)
            self.engine.oracle.calls += max(n - len(have), 0)
            out.append(self.archive.add(phrase, Score(phrase, samples), origin, parent, archetype)[0])
        return out


class PairOracle:
    """Hidden strength: 'deep' lines 4.5 logits, others 4.0 (both round to 0.98-0.99 alone). Each copy of
    'answer is no' costs 1.5 logits."""

    def __init__(self):
        self.calls = 0
        self.qkeys = set()

    async def score(self, states, n=1, qkey=None):
        self.qkeys.add(qkey)
        out = {}
        for state in states:
            z = (7.0 if "sturdy" in state else 4.5 if "deep" in state else 4.0) - 1.5 * state.count("answer is no")
            self.calls += n
            out[state] = Score(state, [round(1 / (1 + math.exp(-z)), 2)] * n)
        return out


class ProbeCtx:
    def __init__(self):
        self.objective = long_objective()
        self.question = {"title": "Is it?"}
        self.llm = None
        self.engine = self
        self.oracle = PairOracle()

    def remaining(self):
        return 10000

    def log(self, message):
        pass


def test_probe_breaks_ties_on_a_level():
    ctx = ProbeCtx()
    probe = Probe("qk")
    probe.claims = lambda ctx: asyncio.sleep(0, ["the answer is no"])
    assert asyncio.run(probe.calibrate(ctx, ["plain line one", "plain line two", "plain line three"]))
    assert probe.claim.count("answer is no") >= 2
    got = asyncio.run(probe.measure(ctx, ["a deep line", "a plain line"]))
    assert got["a deep line"] > got["a plain line"]
    assert ctx.oracle.qkeys == {"qk:probe"}

    lib = library()
    lib.probe = probe
    archive = Archive(ctx.objective)
    deep = archive.add("a deep line", Score("a deep line", [0.98]), "t")[0]
    plain = archive.add("a plain line", Score("a plain line", [0.98]), "t")[0]
    above = archive.add("higher", Score("higher", [0.99]), "t")[0]
    assert lib.merit(above) > lib.merit(deep) > lib.merit(plain)


def test_probe_strengthens_when_the_best_chains_saturate_it():
    ctx = ProbeCtx()
    probe = Probe("qk")
    probe.claims = lambda ctx: asyncio.sleep(0, ["the answer is no"])
    asyncio.run(probe.calibrate(ctx, ["plain one", "plain two", "plain three"]))
    sturdy = ["sturdy one", "sturdy two", "sturdy three"]
    asyncio.run(probe.measure(ctx, sturdy + ["a deep line"], n=3))
    assert probe.saturated(sturdy)
    before = probe.claim.count("answer is no")
    assert asyncio.run(probe.strengthen(ctx, sturdy, sturdy + ["a deep line"]))
    assert probe.claim.count("answer is no") > before
    assert not probe.saturated(sturdy)
    assert probe.value("sturdy one") > probe.value("a deep line")


def test_fit_uses_probe_on_a_flat_level():
    rng = random.Random(7)
    lib = library(rng)
    lib.probe = Probe("qk")
    strong = lib.add("the magic keyword is here", "archive", cluster=0, min_units=3)
    archive = Archive(long_objective())
    for _ in range(120):
        ids = lib.stack(120)
        phrase = lib.register(ids)
        archive.add(phrase, Score(phrase, [0.98]), "test")
        z = 0.3 + 1.2 * ids.count(strong.id) + rng.gauss(0, 0.2)
        lib.probe.values[phrase] = 1 / (1 + math.exp(-z))
    assert lib.fit(archive) >= 100
    assert lib.top_fragments(1) == [strong.text]


def test_long_gain_ignores_probe_noise():
    assert Engine.long_gain((0, 0.98, 0.80), (0, 0.98, 0.81)) == 0.0
    assert Engine.long_gain((0, 0.98, 0.80), (0, 0.98, 0.86)) > 0
    assert Engine.long_gain((0, 0.98, 0.90), (0, 0.99, 0.0)) > 0
    assert Engine.long_gain((0, 0.99, 0.5), (1, 0.99, -280)) == 1.0


def test_compaction_keeps_the_needed_fragment():
    ctx = FakeCtx()
    lib = ctx.chains
    key = lib.add("the magic keyword stays", "archive", cluster=1, min_units=3)
    ids = lib.stack(240)
    ids.insert(len(ids) // 2, key.id)
    root = lib.register(ids)
    asyncio.run(ctx.evaluate([(root, "test", "", "other")], n=5))
    asyncio.run(ChainCompact().run(ctx))
    held = [c for c in ctx.engine.roots() if c.units < len(root.split())]
    best = min(held, key=lambda c: c.units)
    assert best.units <= WORD_STAGE
    assert "magic keyword" in best.phrase
