"""Hard Highest yes/no visits deal the catalog, grow the close frames, then compress.

Easy boards, Shortest yes, choice questions, and score questions stay on the ladder.
"""

import asyncio
import json
import re
import time

import httpx
import pytest

from jevlab.config import GEN_MODELS
from jevlab.llm import LLM
from jevlab.objective import Leader
from jevlab.oracle import Score
from jevlab.rules import RuleError, check_phrase
from jevlab.search.engine import BudgetExhausted, Engine
from jevlab.search.hardboard import CATALOG, HARD_LENGTH, HARD_LENGTH_STEP, HardSchedule, ScoredLine
from jevlab.search.prompts import gen_user
from jevlab.search.mechanical import BeamBuild, Grow
from jevlab.search import memory as lab_memory
from jevlab.search.probe import Lock
from jevlab.search.routes import assign
from jevlab.search.strategies import LLMGenerate

from helpers import make_db


def schedule(lines=(), counts=None, models=None, leader=(0.95, 40), **kw):
    return HardSchedule(
        kind=kw.get("kind", "noul"),
        shortest=kw.get("shortest", False),
        long=kw.get("long", False),
        leader=None if leader is None else Leader(leader[0], leader[1], ours=kw.get("ours", False)),
        we_lead=kw.get("we_lead", False),
        counts={} if counts is None else counts,
        lines=list(lines),
        models=["m1", "m2", "m3", "m4"] if models is None else list(models),
    )


def engine(tmp_path, name, leader=(0.95, 40), board="highScores", kind="noul"):
    path = tmp_path / name
    path.mkdir()
    return Engine(make_db(path, kind=kind, leader=leader), "q", use_llm=False, board=board)


def cover(counts=None):
    done = {name: 8 for name in CATALOG}
    if counts:
        done.update(counts)
    return done


class Writer:
    def __init__(self, reply="labeled"):
        self.prompts = []
        self.reply = reply

    async def json(self, model, system, user, temperature=1.0, max_tokens=4000, **kwargs):
        self.prompts.append((model, user))
        if self.reply == "empty":
            return {"phrases": []}
        if self.reply == "other":
            return {"phrases": [{"tactic": "other", "text": f"missed label {len(self.prompts)} alpha"}]}
        frames = re.findall(r"^- \[(\w+)\]", user, re.M)
        tactic = frames[0] if frames else "other"
        return {"phrases": [{"tactic": tactic, "text": f"dealt line {len(self.prompts)} alpha beta"}]}


class FakeOracle:
    def __init__(self):
        self.calls = 0
        self.qkey = "k"

    async def score(self, phrases, n):
        self.calls += len(phrases) * n
        return {phrase: Score(phrase, [0.91]) for phrase in phrases}


class FarOracle(FakeOracle):
    async def score(self, phrases, n):
        self.calls += len(phrases) * n
        return {phrase: Score(phrase, [0.40]) for phrase in phrases}


def test_catalog_is_the_thirty_one_frames():
    assert len(CATALOG) == 31
    assert "other" not in CATALOG


def test_a_hard_board_does_not_start_a_line_from_nothing(tmp_path):
    eng = engine(tmp_path, "hard", leader=(0.99, 86))
    assert eng.on_hard_board()
    assert isinstance(eng.strategies["beam"], BeamBuild)
    assert isinstance(eng.strategies["grow"], Grow)
    assert not eng.allowed("beam") and not eng.allowed("grow")
    eng.enter_plateau("test")
    assert eng.plateau and not eng.allowed("beam") and not eng.allowed("grow")

    short = engine(tmp_path, "short", leader=(0.99, 4), board="shortestYes")
    assert not short.on_hard_board()
    assert short.allowed("beam")
    short.plateau = True
    assert short.allowed("grow")


def test_eight_models_deal_disjoint_frames_least_written_first():
    models = [f"model-{i}" for i in range(8)]
    fresh = schedule(models=models).plan_batches()
    assert [batch.model for batch in fresh] == models
    frames = [frame for batch in fresh for frame in batch.frames]
    assert frames == list(CATALOG[:24])
    assert len(set(frames)) == 24
    left = list(CATALOG[24:])
    assert len(left) == 7
    counted = {name: 1 for name in frames}
    nxt = schedule(counts=counted, models=models).plan_batches()
    assert list(nxt[0].frames) == left[:3]

    uneven = {name: 5 for name in CATALOG}
    uneven["scene"] = 0
    uneven["rename"] = 1
    uneven["speaker"] = 1
    first = schedule(counts=uneven, models=models).plan_batches()[0]
    assert first.frames == ("scene", "rename", "speaker")


def test_pool_size_sets_the_deal_and_a_covered_catalog_deals_nothing():
    one = schedule(models=["only"]).plan_batches()
    assert len(one) == 4 and {batch.model for batch in one} == {"only"}
    two = schedule(models=["a", "b"]).plan_batches()
    assert [batch.model for batch in two] == ["a", "b", "a", "b"]
    assert len(two) == 4
    assert schedule(counts=cover(), models=["a", "b", "c", "d"]).plan_batches() == []
    # Fewer than three unfinished frames still get one batch, not four empty ones.
    almost = cover()
    last = CATALOG[-2:]
    for name in last:
        almost[name] = 7
    tail = schedule(counts=almost, models=["only"]).plan_batches()
    assert len(tail) == 1 and tail[0].frames == tuple(last)


def test_the_next_visit_uses_saved_generator_counts_only(tmp_path):
    eng = engine(tmp_path, "counts")
    eng.oracle = FakeOracle()
    asyncio.run(eng.evaluate([("alpha beta", "llm:core", "", "rename")]))
    asyncio.run(eng.evaluate([("alpha gamma", "sweep", "", "rename")]))
    asyncio.run(eng.evaluate([("alpha delta", "llm:core", "", "other")]))
    asyncio.run(eng.evaluate([("alpha epsilon", "llm:core", "", "speaker")]))
    assert eng.memory.generator_counts == {"rename": 1, "speaker": 1}
    assert eng.archive.style_stats()["rename"]["tries"] == 2

    lab_memory.save(eng.db, "k", "q", eng.memory)
    loaded = lab_memory.load(eng.db, "k")
    assert loaded.generator_counts == {"rename": 1, "speaker": 1}
    assert loaded.styles == {}

    models = [f"model-{i}" for i in range(8)]
    first = schedule(models=models).plan_batches()
    dealt = [frame for batch in first for frame in batch.frames]
    failed = first[0].frames
    counts = {frame: 1 for frame in dealt if frame not in failed}
    second = schedule(counts=counts, models=models).plan_batches()
    later = [frame for batch in first[1:] for frame in batch.frames]
    assert not set(failed) & set(later)
    assert list(second[0].frames) == list(failed)


def test_a_new_best_does_not_cancel_the_deal_or_drop_the_ladder(tmp_path):
    eng = engine(tmp_path, "deal", leader=(0.95, 30))
    eng.level = 2
    eng.probe_force = False
    eng.probe_lock = Lock(model="locked-writer", batches=4)
    # A model the ladder would call premium is still in the deal, and a cold model is not skipped.
    models = [GEN_MODELS[0], "premium/left-out"]
    arm = eng._generator_arm(models[0])
    arm.cold_until[models[0]] = time.monotonic() + 10_000
    eng.schedule = schedule(models=models, leader=(0.95, 30))
    eng.schedule.counts = eng.memory.generator_counts
    eng.schedule.begin()
    writer = Writer()
    eng.llm = writer
    seen = []

    async def evaluate(items, n=1):
        seen.append(items)
        phrase = f"better line {len(seen)} alpha"
        eng.archive.add(phrase, Score(phrase, [0.99]), "llm:core", archetype="direct")
        return []

    eng.evaluate = evaluate
    ran = asyncio.run(eng.run_hard_deal())
    assert ran == 4
    assert [model for model, _ in writer.prompts] == [models[0], models[1], models[0], models[1]]
    assert eng.level == 2
    assert eng.memory.exhausted == []
    assert eng.probe_force is False
    assert eng.probe_lock.model == "locked-writer"
    assert all("Write 40" in user for _, user in writer.prompts)
    assert "syllogism" not in writer.prompts[0][1]
    assert eng.schedule.phase == "grow"


def test_a_mislabeled_batch_uses_its_frames_without_advancing_them(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.search.routes.DEAR_ARCHETYPES", frozenset())
    monkeypatch.setattr("jevlab.search.routes.DEAR_MODELS", frozenset({"dear/model"}))
    eng = engine(tmp_path, "miss")
    eng.oracle = FakeOracle()
    eng.budget = 4
    eng.schedule = schedule(models=["dear/model"])
    eng.schedule.counts = eng.memory.generator_counts
    eng.schedule.begin()
    assigned = [frame for batch in eng.schedule.batches for frame in batch.frames]
    assert assign("dear/model", list(assigned[:3])) is None
    writer = Writer(reply="other")
    eng.llm = writer
    with pytest.raises(BudgetExhausted):
        asyncio.run(eng.run_hard_deal())
    assert len(writer.prompts) > 4
    assert all(model == "dear/model" for model, _ in writer.prompts)
    assert eng.memory.generator_counts == {}
    nxt = schedule(counts=dict(eng.memory.generator_counts), models=["dear/model"]).plan_batches()
    assert list(nxt[0].frames) == list(assigned[:3])


def test_displayed_hundredths_keep_a_frame_at_five_hundredths():
    # 0.849 displays as 0.85, which is 0.05 from 0.90. The raw gap is 0.051.
    # 0.844 displays as 0.84, which is 0.06 away. An edit can be the frame's best line.
    lines = [
        ScoredLine("direct", "sweep", 0.849, 6, "alpha beta gamma delta epsilon zeta"),
        ScoredLine("scene", "llm:core", 0.844, 6, "one two three four five six"),
        ScoredLine("speaker", "llm:core", 0.85, 4, "speaker line alpha beta"),
    ]
    sched = schedule(lines=lines, leader=(0.90, 20))
    sched.begin()
    assert sched.phase == "grow"
    assert sched.growth_frames == ["speaker", "direct"]
    assert "scene" not in sched.growth_frames
    assert "extend" in sched.jobs() and "sweep" in sched.jobs() and "surrogate_bo" in sched.jobs()
    assert "compress" not in sched.jobs() and "beam" not in sched.jobs() and "grow" not in sched.jobs()


def test_nothing_close_deals_again_until_a_frame_is_within_five_hundredths():
    """A far pass starts another generator pass. Coverage of eight does not stop it. 0.05 does."""
    far = ScoredLine("direct", "llm:core", 0.40, 4, "alpha beta gamma delta")
    sched = schedule(lines=[far], leader=(0.95, 40))
    sched.begin()
    assert sched.phase == "deal"
    first = [frame for batch in sched.batches for frame in batch.frames]
    for frame in first:
        sched.counts[frame] = 1
    sched.finish_deal()
    assert sched.deal_ran and sched.phase == "deal" and sched.growth_frames == []
    second = [frame for batch in sched.batches for frame in batch.frames]
    assert second and not set(second) & set(first)

    covered = schedule(
        lines=[
            ScoredLine("direct", "llm:core", 0.80, 5, "alpha beta gamma delta epsilon"),
            ScoredLine("scene", "sweep", 0.70, 5, "one two three four five"),
        ],
        counts=cover(),
        leader=(0.95, 40),
    )
    covered.begin()
    assert covered.phase == "deal" and covered.growth_frames == []
    covered.lines.append(ScoredLine("direct", "llm:core", 0.91, 5, "close alpha beta gamma delta"))
    covered.finish_deal()
    assert covered.phase == "grow" and covered.growth_frames == ["direct"]


def test_an_empty_generator_pass_does_not_deal_forever(tmp_path):
    eng = engine(tmp_path, "empty", leader=(0.95, 40))
    eng.oracle = FakeOracle()
    eng.schedule = schedule(models=["only"], leader=(0.95, 40))
    eng.schedule.counts = eng.memory.generator_counts
    eng.schedule.begin()
    eng.llm = Writer(reply="empty")
    ran = asyncio.run(eng.run_hard_deal())
    assert ran == 4
    assert eng.schedule.phase == "done"
    assert eng.oracle.calls == 0


def test_consecutive_hard_passes_ask_for_longer_lines(tmp_path):
    """Each length is tried twice: plain, then with the board lines that raise the score, then ten more words."""
    eng = engine(tmp_path, "stretch", leader=(0.95, 12))
    assert "mostly 6 to 18" in gen_user(eng.ctx, ["direct"], 4)
    eng.impacts = [
        {"word": "helpline", "averageImpact": 0.4},
        {"word": "flatline", "averageImpact": 0},
        {"word": "harmline", "averageImpact": -0.3},
    ]
    eng.budget = 8
    eng.oracle = FarOracle()
    eng.schedule = schedule(models=["m1", "m2"], leader=(0.95, 12))
    eng.schedule.counts = eng.memory.generator_counts
    eng.schedule.begin()
    eng.llm = Writer()
    with pytest.raises(BudgetExhausted):
        asyncio.run(eng.run_hard_deal())

    prompts = [user for _, user in eng.llm.prompts]
    assert len(prompts) >= 9
    assert all("about 60 words each" in prompt for prompt in prompts[:8])
    assert "mostly 6 to 18" not in prompts[0]
    assert "raise the score" not in prompts[0]
    injected = prompts[4].split("raise the score", 1)[1]
    assert "helpline" in injected
    assert "harmline" not in injected
    assert "flatline" not in injected
    assert "about 70 words each" in prompts[8]
    assert "raise the score" not in prompts[8]
    assert check_phrase(" ".join(["alpha"] * 70), eng.choices, eng.length_cap)

    # The site stores averageImpact in the direction of the shown score, including when the goal is no.
    no = engine(tmp_path, "no-goal", leader=(0.95, 12))
    no.objective.goal = "no"
    no.impacts = eng.impacts
    no.schedule = schedule(models=["m1"], leader=(0.95, 12))
    no.schedule.begin()
    no.schedule.deal_passes = 2
    no.apply_hard_deal_length()
    text = gen_user(no.ctx, ["direct"], 4)
    assert "about 60 words each" in text
    assert "raise the score" in text
    directed = text.split("raise the score", 1)[1]
    assert "helpline" in directed
    assert "harmline" not in directed


def test_a_longer_hard_pass_gives_the_model_proportionally_more_time(tmp_path, monkeypatch):
    """A hard deal waits one ordinary timeout per 20 words, including the first 60-word pass.

    z.ai was still running when an 80-word ask hit 60s, so 80 words gets four minutes and a
    180-word ask waits longer than the old one-second-per-word scale. The token cap still
    grows from the 60-word budget. A reply slower than that old scale still comes back.
    """
    base = 0.05
    monkeypatch.setattr("jevlab.llm.LLM_TIMEOUT", base)
    monkeypatch.setattr("jevlab.config.LLM_TIMEOUT", base)
    monkeypatch.setattr("jevlab.llm.OPENROUTER_KEY", "test-key")
    token_budget = 6000
    seen: list[tuple[int, float]] = []
    slow = {"on": False}

    async def handler(request):
        body = json.loads(request.content)
        seen.append((body["max_tokens"], request.extensions["timeout"]["read"]))
        if slow["on"]:
            await asyncio.sleep(base * 4)
        reply = '{"phrases": [{"tactic": "direct", "text": "alpha beta"}]}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply}}], "usage": {"cost": 0}},
        )

    async def call(eng):
        return await LLMGenerate(["m1"])._call(eng.ctx, "m1", ["direct"], [])

    async def run():
        llm = LLM()
        await llm.http.aclose()
        llm.http = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            transport=httpx.MockTransport(handler),
            timeout=base * 2,
        )
        try:
            plain = engine(tmp_path, "plain", leader=(0.95, 12))
            plain.llm = llm
            ordinary = await call(plain)

            opening = engine(tmp_path, "opening", leader=(0.95, 12))
            opening.llm = llm
            opening.schedule = schedule(models=["m1"], leader=(0.95, 12))
            opening.schedule.begin()
            opening.apply_hard_deal_length()
            first = await call(opening)

            words = HARD_LENGTH * 3
            stretched = engine(tmp_path, "stretched", leader=(0.95, 12))
            stretched.llm = llm
            stretched.schedule = schedule(models=["m1"], leader=(0.95, 12))
            stretched.schedule.begin()
            steps = (words - HARD_LENGTH) // HARD_LENGTH_STEP
            stretched.schedule.deal_passes = 1 + steps * 2
            stretched.apply_hard_deal_length()
            assert stretched.schedule.deal_length() == words
            slow["on"] = True
            longer = await call(stretched)
        finally:
            await llm.http.aclose()
        return ordinary, first, longer

    ordinary, first, longer = asyncio.run(run())
    assert ordinary
    assert first
    assert seen[0][0] == token_budget
    assert seen[1][0] == token_budget
    assert seen[1][1] == pytest.approx(base * (HARD_LENGTH / 20))
    assert seen[2][0] == token_budget * 3
    assert seen[2][1] == pytest.approx(base * (HARD_LENGTH * 3 / 20))
    assert longer


def test_a_close_frame_does_not_stretch_the_generator_length(tmp_path):
    eng = engine(tmp_path, "held", leader=(0.95, 12))
    phrase = " ".join(["alpha"] * 12)
    eng.archive.add(phrase, Score(phrase, [0.93]), "llm:core", archetype="direct")
    eng.memory.generator_counts = cover()
    eng.open_hard_schedule()
    assert eng.schedule.phase == "grow"
    assert eng.length_cap == 30
    assert "mostly 6 to 18" in gen_user(eng.ctx, ["direct"], 4)


def test_a_far_deal_keeps_going_until_the_budget_runs_out(tmp_path):
    eng = engine(tmp_path, "loop", leader=(0.95, 40))
    eng.budget = 6
    eng.oracle = FarOracle()
    eng.schedule = schedule(models=["m1", "m2"], leader=(0.95, 40))
    eng.schedule.counts = eng.memory.generator_counts
    eng.schedule.begin()
    eng.llm = Writer()
    with pytest.raises(BudgetExhausted):
        asyncio.run(eng.run_hard_deal())
    assert eng.rounds > 4
    assert eng.schedule.phase == "deal"


def test_growth_edits_only_the_kept_frame(tmp_path):
    lines = [
        ScoredLine("direct", "llm:core", 0.91, 5, "alpha beta gamma delta epsilon"),
        ScoredLine("scene", "sweep", 0.70, 5, "one two three four five"),
    ]
    sched = schedule(lines=lines, counts=cover(), leader=(0.95, 40))
    sched.begin()
    assert not sched.deal_ran and sched.phase == "grow" and sched.growth_frames == ["direct"]

    eng = engine(tmp_path, "closest", leader=(0.95, 40))
    for line in lines:
        eng.archive.add(line.phrase, Score(line.phrase, [line.p]), line.origin, archetype=line.archetype)
    stray = "other line alpha beta"
    eng.archive.add(stray, Score(stray, [0.99]), "sweep", archetype="other")
    eng.memory.generator_counts = cover()
    eng.open_hard_schedule()
    assert eng.schedule.growth_frames == ["direct"]
    assert [cand.phrase for cand in eng.ctx.parents(4)] == [lines[0].phrase]
    assert eng.ctx.climb_roots(2)[0].phrase == lines[0].phrase
    assert eng.strategies["local_edit"].pick_parent(eng.ctx).phrase == lines[0].phrase
    assert eng.level == 0 and eng.strategies["sweep"].available(eng.ctx)


def test_a_kept_line_may_grow_past_the_leader(tmp_path):
    eng = engine(tmp_path, "long", leader=(0.95, 60))
    phrase = " ".join(["alpha"] * 60)
    eng.archive.add(phrase, Score(phrase, [0.90]), "llm:core", archetype="direct")
    eng.memory.generator_counts = cover()
    longer = " ".join(["alpha"] * 61)
    with pytest.raises(RuleError):
        check_phrase(longer, eng.choices, eng.length_cap)
    eng.open_hard_schedule()
    assert eng.schedule.phase == "grow" and "direct" in eng.schedule.growth_frames
    assert eng.length_cap >= 62 and eng.grow_cap >= 62
    assert check_phrase(longer, eng.choices, eng.length_cap) == longer
    assert not eng.plateau and eng.strategies["extend"].available(eng.ctx)


def test_highest_compression_waits_for_the_score_and_for_growth(tmp_path):
    eng = engine(tmp_path, "cut", leader=(0.95, 10))
    behind = " ".join(["alpha"] * 14)
    level = " ".join(["beta"] * 12)
    eng.archive.add(behind, Score(behind, [0.80]), "t", archetype="direct")
    eng.archive.add(level, Score(level, [0.95]), "t", archetype="scene")
    eng.memory.generator_counts = cover()
    eng.open_hard_schedule()
    assert eng.schedule.phase == "grow"
    compress = eng.strategies["compress"]
    assert not compress.worth_cutting(eng.ctx, eng.archive.items[behind])
    assert not compress.worth_cutting(eng.ctx, eng.archive.items[level])
    eng.schedule.finish_growth()
    assert eng.schedule.phase == "compress"
    assert not compress.worth_cutting(eng.ctx, eng.archive.items[behind])
    assert compress.worth_cutting(eng.ctx, eng.archive.items[level])
    compress.done.add(level)
    assert not compress.available(eng.ctx)

    easy = engine(tmp_path, "easy", leader=(0.50, 10))
    easy.archive.add(behind, Score(behind, [0.40]), "t", archetype="direct")
    easy.archive.add(level, Score(level, [0.50]), "t", archetype="scene")
    assert not easy.on_hard_board()
    assert not easy.strategies["compress"].worth_cutting(easy.ctx, easy.archive.items[behind])
    assert easy.strategies["compress"].worth_cutting(easy.ctx, easy.archive.items[level])


def test_easy_choice_and_score_boards_do_not_deal(tmp_path):
    easy = engine(tmp_path, "low", leader=(0.89, 40))
    assert not easy.on_hard_board()
    assert easy.open_hard_schedule().phase == "off"
    assert easy.schedule.plan_batches() == []

    (tmp_path / "empty").mkdir()
    empty = make_db(tmp_path / "empty", leader=(0.95, 10))
    empty.execute("UPDATE boards SET rows = '[]'")
    bare = Engine(empty, "q", use_llm=False)
    assert bare.leader is None and bare.open_hard_schedule().plan_batches() == []

    choice = engine(tmp_path, "choice", leader=(0.95, 10), kind="choice")
    score = engine(tmp_path, "score", leader=(0.95, 10), kind="score")
    assert choice.objective.kind == "choice" and not choice.on_hard_board()
    assert score.objective.kind == "score" and not score.on_hard_board()
    assert choice.open_hard_schedule().plan_batches() == []
    long_choice = " ".join(["alpha"] * 12)
    choice.archive.add(long_choice, Score(long_choice, [0.40]), "t")
    assert choice.strategies["compress"].worth_cutting(choice.ctx, choice.archive.items[long_choice])

    short = engine(tmp_path, "shortest", leader=(0.95, 4), board="shortestYes")
    assert not short.on_hard_board()
    caught = "alpha beta gamma"
    short.archive.add(caught, Score(caught, [0.40, 0.80]), "t")
    assert short.strategies["compress"].worth_cutting(short.ctx, short.archive.items[caught])


def test_a_hard_visit_does_not_resume_the_ladder_or_enter_plateau(tmp_path):
    eng = engine(tmp_path, "resume", leader=(0.99, 86))
    eng.fresh = True
    eng.memory.runs = 2
    eng.memory.level = 4
    assert eng.plateau_candidate
    assert eng.opening_level() == 0
    assert not eng.should_enter_plateau(500)
    assert eng.probe_force is False and eng.memory.exhausted == []

    easy = engine(tmp_path, "ladder", leader=(0.50, 30))
    easy.fresh = True
    easy.memory.level = 2
    assert easy.opening_level() == 2


def test_leading_the_board_keeps_the_ordinary_search(tmp_path):
    (tmp_path / "ours").mkdir()
    db = make_db(tmp_path / "ours", leader=(0.95, 40))
    db.execute("INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES ('t', 'jev', 'me', '')")
    rows = [
        {"userId": "me", "name": "us", "probability": 0.97, "wordCount": 8},
        {"userId": "other", "name": "rival", "probability": 0.95, "wordCount": 40},
    ]
    db.execute("UPDATE boards SET rows = ?", (json.dumps(rows),))
    eng = Engine(db, "q", use_llm=False)
    assert eng.we_lead() and not eng.on_hard_board()
    assert eng.open_hard_schedule().phase == "off"
