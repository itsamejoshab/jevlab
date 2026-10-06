"""Shortest yes against a one-word leader: only a one-word line can place, so the search stays there."""

import asyncio

from jevlab.objective import Leader
from jevlab.oracle import Score
from jevlab.rules.strict import MAX_WORDS
from jevlab.search.engine import Engine
from jevlab.search.prompts import gen_user, rewrite_user, single_words_user
from jevlab.search.triage import length_can_win

from helpers import make_db


def engine(tmp_path, name, leader, board="shortestYes"):
    (tmp_path / name).mkdir()
    return Engine(make_db(tmp_path / name, leader=leader), "q", use_llm=False, board=board)


def test_one_word_leader_on_shortest_yes_is_a_one_word_race(tmp_path):
    race = engine(tmp_path, "race", (0.7, 1))
    longer = engine(tmp_path, "longer", (0.7, 2))
    highest = engine(tmp_path, "highest", (0.7, 1), board="highScores")
    empty = engine(tmp_path, "empty", (0.7, 1))
    empty.leader = None

    assert race.one_word_race and race.grow_cap == 1 and race.length_cap == MAX_WORDS
    assert race.scheduler.arms["single_word"].prior == 4.0
    assert not race.allowed("boosters")
    assert not race.allowed("local_edit")

    assert not longer.one_word_race and longer.grow_cap == 8
    assert longer.scheduler.arms["single_word"].prior == 2.5
    assert longer.allowed("boosters") and longer.allowed("local_edit")
    assert not highest.one_word_race and highest.allowed("boosters")
    assert not empty.one_word_race

    race.archive.add("yes", Score("yes", [0.6]), "history")
    assert race.allowed("local_edit")
    assert not highest.one_word_race


def test_prompts_ask_for_one_word_only_in_that_race(tmp_path):
    race = engine(tmp_path, "race", (0.7, 1))
    longer = engine(tmp_path, "longer", (0.7, 2))
    highest = engine(tmp_path, "highest", (0.7, 1), board="highScores")

    race_gen = gen_user(race.ctx, ["direct"], 10)
    assert "exactly 1 word" in race_gen
    longer_gen = gen_user(longer.ctx, ["direct"], 10)
    assert "exactly 1 word" not in longer_gen
    assert "1 to 3 words" in longer_gen
    assert "exactly 1 word" not in gen_user(highest.ctx, ["direct"], 10)

    assert "exactly one word" in single_words_user(race.ctx, 10)
    assert "two-word phrases" in single_words_user(longer.ctx, 10)
    assert "exactly one word" in rewrite_user(race.ctx, ["alpha beta"], 2)
    assert "one to three words" in rewrite_user(longer.ctx, ["alpha beta"], 2)


def test_seeds_and_proposals_stay_one_word(tmp_path):
    race = engine(tmp_path, "race", (0.7, 1))
    longer = engine(tmp_path, "longer", (0.7, 2))
    for eng in (race, longer):
        eng.archive.add("alpha beta gamma", Score("alpha beta gamma", [0.8, 0.8]), "history")

    race_seeds = {row[0] for row in race.prefix_seeds()}
    assert race_seeds == {"alpha", "beta", "gamma"}
    longer_seeds = {row[0] for row in longer.prefix_seeds()}
    assert "alpha beta" in longer_seeds and "alpha beta gamma" not in longer_seeds

    assert race.ctx.valid_only(["yes", "yes please", "no"]) == ["yes", "no"]
    assert "yes please" in longer.ctx.valid_only(["yes", "yes please"])


def test_scoring_skips_multi_word_tries_except_a_manual_one(tmp_path):
    race = engine(tmp_path, "race", (0.7, 1))
    longer = engine(tmp_path, "longer", (0.7, 2))

    class Oracle:
        def __init__(self):
            self.calls = 0
            self.qkey = "k"
            self.scored: list[str] = []

        async def score(self, phrases, n=1):
            self.scored.extend(phrases)
            self.calls += len(phrases) * n
            return {phrase: Score(phrase, [0.6]) for phrase in phrases}

    race.oracle = Oracle()
    longer.oracle = Oracle()
    items = [("yes", "single_word", "", "wild"), ("yes please", "llm:cheap", "", "direct"),
             ("typed phrase", "manual", "", "other")]
    asyncio.run(race.evaluate(items))
    asyncio.run(longer.evaluate(items))
    assert race.oracle.scored == ["yes", "typed phrase"]
    assert longer.oracle.scored == ["yes", "yes please", "typed phrase"]


def test_compress_mines_single_words_from_a_longer_line(tmp_path):
    race = engine(tmp_path, "race", (0.7, 1))
    longer = engine(tmp_path, "longer", (0.7, 2))
    seen = {"race": [], "longer": []}

    def watch(name, eng):
        async def evaluate(items, n=1):
            seen[name].extend(item[0] for item in items)
            return []

        eng.evaluate = evaluate

    watch("race", race)
    watch("longer", longer)

    class Oracle:
        calls = 0

    race.oracle = longer.oracle = Oracle()
    root_score = Score("alpha beta gamma", [0.8, 0.8])
    for eng in (race, longer):
        eng.archive.add("alpha beta gamma", root_score, "history")
    root = race.archive.items["alpha beta gamma"]
    asyncio.run(race.strategies["compress"].compress(race.ctx, root))
    asyncio.run(longer.strategies["compress"].compress(longer.ctx, longer.archive.items["alpha beta gamma"]))
    assert seen["race"] == ["alpha", "beta", "gamma"]
    assert any(len(phrase.split()) > 1 for phrase in seen["longer"])


def test_length_filter_keeps_one_word_lines_only_against_a_one_word_leader():
    one = Leader(0.8, 1, "rival")
    two = Leader(0.8, 2, "rival")
    three = Leader(0.8, 3, "rival")
    assert length_can_win(1, one)
    assert not length_can_win(2, one)
    assert length_can_win(1, two) and not length_can_win(2, two)
    assert length_can_win(1, three) and length_can_win(2, three)
    assert not length_can_win(3, three)
    assert length_can_win(4, None)
