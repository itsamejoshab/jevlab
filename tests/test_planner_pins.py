"""The planner's pin list is applied when the best lines show the word carries the score.

A pin is required in every later phrase. A suggestion is kept only when the word is in the
best line, is not a function word or a ban, and appears in a strict majority of the lines
within 0.02 of the best. Each plan replaces the set, so a pin the planner drops comes off.
"""

import asyncio

from jevlab.oracle import Score

from test_syllogism import engine


class _Planner:
    def __init__(self, data):
        self.data = data

    async def json(self, *args, **kwargs):
        return self.data


class _Oracle:
    def __init__(self):
        self.calls = 0
        self.qkey = "k"
        self.scored: list[str] = []

    async def score(self, states, n=1):
        self.scored.extend(states)
        self.calls += len(states) * n
        return {state: Score(state, [0.9] * n) for state in states}


def _plan(eng, data):
    eng.llm = _Planner(data)
    eng.oracle = _Oracle()
    asyncio.run(eng.plan())


def _add(eng, phrase, p):
    eng.archive.add(phrase, Score(phrase, [p]), "llm")


def test_a_pin_sticks_only_when_the_lines_near_the_best_share_it(tmp_path):
    eng = engine(tmp_path, "pin")
    _add(eng, "ledger engines avert avoid is stores tables", 0.93)
    _add(eng, "ledger piles avert avoid is stores tables", 0.92)
    _add(eng, "ledger heaps avert avoid is stores tables", 0.915)
    _add(eng, "unrelated chicken soup recipe", 0.40)
    _add(eng, "another weak line here", 0.30)
    eng.ctx.banned.add("stores")
    events = []
    eng.on_event = lambda event: events.append(event)

    _plan(
        eng,
        {
            "directive": "Stay on the ledger frame",
            "focus": ["scene"],
            "mode": "exploit",
            "ban": ["ledger"],
            "pin": ["avoid", "ledger", "is", "stores", "chicken", "engines", "avert", "tables", "not a word"],
            "seed_phrases": [],
        },
    )

    # is is a function word, stores is banned, chicken is absent from the best line,
    # engines is only in the best line, tables is a fourth shared word past the cap,
    # and "not a word" is not one word.
    assert {word.casefold() for word in eng.ctx.pinned} == {"avoid", "ledger", "avert"}
    assert "ledger" not in eng.ctx.banned
    plan = next(event for event in events if event.kind == "plan")
    assert {word.casefold() for word in plan.data["pin"]} == {"avoid", "ledger", "avert"}


def test_a_lone_best_line_can_pin_a_word_the_runners_up_lack(tmp_path):
    eng = engine(tmp_path, "lone")
    _add(eng, "ledger engines avert avoid", 0.93)
    _add(eng, "unrelated chicken soup recipe", 0.40)
    _add(eng, "another weak line here", 0.30)
    _add(eng, "third weak line there", 0.20)
    _add(eng, "fourth weak line now", 0.10)

    _plan(eng, {"directive": "hold ledger", "mode": "exploit", "pin": ["ledger"], "seed_phrases": []})

    assert {word.casefold() for word in eng.ctx.pinned} == {"ledger"}


def test_the_next_plan_drops_a_pin_it_no_longer_asks_for(tmp_path):
    eng = engine(tmp_path, "drop")
    for phrase, p in (
        ("ledger engines avert avoid", 0.93),
        ("ledger stores avert avoid", 0.92),
        ("plain alpha beta gamma", 0.50),
        ("plain delta epsilon zeta", 0.40),
        ("plain eta theta iota", 0.30),
    ):
        _add(eng, phrase, p)
    _plan(eng, {"directive": "hold", "mode": "exploit", "pin": ["ledger"], "seed_phrases": []})
    assert eng.ctx.pinned

    _plan(eng, {"directive": "look elsewhere", "mode": "explore", "pin": [], "seed_phrases": []})

    assert eng.ctx.pinned == set()


def test_a_new_phrase_without_the_pinned_words_is_not_scored(tmp_path):
    eng = engine(tmp_path, "filter")
    _add(eng, "ledger engines avert avoid", 0.93)
    _add(eng, "ledger stores avert avoid", 0.92)
    _add(eng, "plain alpha beta gamma", 0.50)
    _add(eng, "plain delta epsilon zeta", 0.40)
    _add(eng, "plain eta theta iota", 0.30)
    _plan(eng, {"directive": "hold", "mode": "exploit", "pin": ["ledger", "avoid"], "seed_phrases": []})
    oracle = eng.oracle

    scored = asyncio.run(
        eng.evaluate(
            [
                ("ledger engines avert avoid extra", "llm", "", "scene"),
                ("engines avert extra", "llm", "", "scene"),
            ]
        )
    )

    assert [cand.phrase for cand in scored] == ["ledger engines avert avoid extra"]
    assert oracle.scored == ["ledger engines avert avoid extra"]
