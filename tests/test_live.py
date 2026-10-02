from jevlab import vault
from jevlab.live import (
    SENTENCE_WORDS,
    LiveSession,
    best_saved,
    carried_question,
    chain_token,
    live_question,
    post_phrase,
    question_from_round,
    search_question,
    sentence_words,
)
from jevlab.objective import Leader
from jevlab.modes import from_name
from jevlab.oracle import Score
from jevlab.search.engine import Engine

from helpers import make_db

ROUND = {
    "id": "r1",
    "status": "live",
    "playMode": "word_chain",
    "question": {
        "slug": "round-avocados",
        "revisionId": "rev1",
        "title": "Is it stealing to ring up avocados as bananas?",
        "kind": "noul",
        "goal": "no",
        "yesThreshold": 0.15,
        "jevRequest": (
            '{"state":"","model":"jev-latest","questions":{"q":{"type":"noul",'
            '"instructions":"Is it stealing to ring up avocados as bananas?"}}}'
        ),
        "choices": [],
        "ranked": False,
    },
    "leaders": [],
    "milestones": [{"id": "finish", "at": 0.15}],
}


def test_round_state_unwraps_the_live_envelope():
    """The site wraps the round as {state, servedAt}. Live mode reads status on the round itself."""
    from jevlab.site.client import SiteClient

    client = SiteClient()
    client.call = lambda name, data, method="GET": {"state": dict(ROUND), "servedAt": 1}
    state = client.round_state("live")
    assert state["status"] == "live"
    assert state["id"] == "r1"
    session = LiveSession()
    assert session.observe(state)
    assert session.question["title"].startswith("Is it stealing")


def test_typed_live_question_never_enters_the_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    mode = from_name("live")
    assert mode.label == "Live Mode"
    assert mode.searchable
    assert not mode.publishable

    question = live_question("Is cereal a soup?")
    assert question["jev_request"]["questions"]["q"]["instructions"] == "Is cereal a soup?"
    assert question["jev_request"]["questions"]["q"]["type"] == "noul"

    db = make_db(tmp_path)
    engine = Engine(db, question["slug"], use_llm=False, question=question)
    assert engine.question["jev_request"]["questions"]["q"]["instructions"] == "Is cereal a soup?"
    cand, _ = engine.archive.add("cereal is soup", Score("cereal is soup", [0.9] * 5), "manual")
    engine.save(cand)
    engine.save(cand, auto=True)
    assert vault.all_entries() == []


def test_site_live_search_saves_under_the_question_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    question = question_from_round(ROUND)
    db = make_db(tmp_path)
    engine = Engine(db, question["slug"], use_llm=False, question=question)
    cand, _ = engine.archive.add("ring them as bananas", Score("ring them as bananas", [0.42] * 5), "manual")
    engine.save(cand)

    entries = vault.all_entries()
    assert [(e["slug"], e["mode"], e["title"], e["phrase"], e["p_mean"]) for e in entries] == [
        (
            "round-avocados",
            "strict_chain",
            "Is it stealing to ring up avocados as bananas?",
            "ring them as bananas",
            0.58,
        )
    ]
    found = best_saved(question, entries)
    assert found == {"phrase": "ring them as bananas", "p": 0.58, "units": 4}


def test_gap_between_rounds_still_names_the_question():
    ended = {**ROUND, "status": "ended", "repeats": True, "nextComes": True}
    question = carried_question(ended)
    assert question is not None
    assert question["slug"] == "round-avocados"
    assert question["title"] == ROUND["question"]["title"]
    assert carried_question({"status": "ended", "question": {}}) is None
    assert carried_question({"status": "idle"}) is None


def test_unknown_next_round_keeps_searching_the_previous_question():
    previous = question_from_round(ROUND)
    assert search_question({"status": "ended", "question": {}}, previous)["slug"] == "round-avocados"
    assert search_question({"status": "idle"}, previous)["slug"] == "round-avocados"
    assert search_question({"status": "live"}, previous) is None
    assert search_question({"status": "ended"}, None) is None


def test_live_search_keeps_climbing_after_the_call_budget(tmp_path):
    question = question_from_round(ROUND)
    engine = Engine(make_db(tmp_path), question["slug"], budget=100, use_llm=False, question=question)
    engine.idle_when_done = True
    engine.oracle = type("Calls", (), {"calls": 100})()
    engine.stall = 12
    engine.level = 0
    assert engine.keeps_searching()
    assert engine.patience() == 12
    assert engine.refill_search()
    assert engine.budget == 200
    assert engine.oracle.calls < engine.budget
    assert not engine.refill_search()

    engine.live = False
    engine.oracle.calls = engine.budget
    assert not engine.keeps_searching()
    assert not engine.refill_search()


def test_live_search_vaults_a_finish_line_that_does_not_beat_the_leader(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    question = question_from_round(ROUND)
    engine = Engine(make_db(tmp_path), question["slug"], use_llm=False, question=question)
    engine.leader = Leader(0.99, 1, "rival")
    cand, _ = engine.archive.add("exactly right", Score("exactly right", [0.20] * 5), "manual")
    assert not engine.objective.wins(cand.score, cand.units, engine.leader)
    assert engine.worth_vaulting(cand)
    engine.save(cand, auto=True)
    assert [e["phrase"] for e in vault.all_entries()] == ["exactly right"]
    weak, _ = engine.archive.add("nope", Score("nope", [0.90] * 5), "manual")
    assert not engine.worth_vaulting(weak)


def test_live_round_posts_when_not_leading():
    question = question_from_round(ROUND)
    assert question["slug"] == "round-avocados"
    assert question["revision_id"] == "rev1"
    assert question["jev_request"]["questions"]["q"]["instructions"] == (
        "Is it stealing to ring up avocados as bananas?"
    )
    assert question["raw"]["live"]
    assert question["raw"]["playMode"] == "word_chain"
    assert question["goal"] == "no"
    unnamed = {**ROUND, "question": {k: v for k, v in ROUND["question"].items() if k != "goal"}}
    assert question_from_round(unnamed)["goal"] == "yes"
    assert LiveSession().objective.goal == "yes"
    session_goal = LiveSession()
    session_goal.observe(ROUND)
    assert session_goal.objective.goal == "no"

    session = LiveSession()
    session.observe(ROUND, me="me")
    assert session.should_post(0.15, 4)
    session.post_enabled = False
    assert not session.should_post(0.20, 1)
    session.post_enabled = True
    assert session.should_post(0.15, 4)
    assert session.should_post(0.20, 4)
    assert not session.should_post(0.10, 1)
    session.mark_posted("first line")
    assert not session.should_post(0.20, 4, "first line")
    assert session.should_post(0.20, 3, "shorter line")

    occupied = {
        **ROUND,
        "leaders": [{"userId": "rival", "name": "rival", "probability": 0.20, "wordCount": 4}],
    }
    session.observe(occupied, me="me")
    assert not session.we_lead
    assert session.should_post(0.22, 5)
    assert not session.should_post(0.20, 5)
    assert session.should_post(0.20, 3)

    leading = {
        **ROUND,
        "leaders": [{"userId": "me", "name": "us", "probability": 0.22, "wordCount": 5}],
    }
    session.observe(leading, me="me")
    assert session.we_lead
    assert not session.should_post(0.30, 2)


def test_live_fast_posts_immediately_and_on_every_gain():
    fast = LiveSession(pace="fast")
    slow = LiveSession(pace="slow")

    fast.observe(ROUND, me="me")
    assert fast.should_post(0.05, 1, "please")
    fast.mark_posted("please", p=0.05, units=1)
    assert not fast.should_post(0.05, 1, "please")
    assert not fast.should_post(0.05, 1, "other")
    assert not fast.should_post(0.04, 1, "worse")
    assert fast.should_post(0.12, 1, "yes")
    fast.mark_posted("yes", p=0.12, units=1)

    occupied = {
        **ROUND,
        "leaders": [{"userId": "rival", "name": "rival", "probability": 0.20, "wordCount": 4}],
    }
    first_shot = LiveSession(pace="fast")
    first_shot.observe(occupied, me="me")
    assert first_shot.should_post(0.05, 1, "please")

    leading = {
        **ROUND,
        "leaders": [{"userId": "me", "name": "us", "probability": 0.22, "wordCount": 1}],
    }
    fast.observe(leading, me="me")
    assert fast.we_lead
    assert fast.should_post(0.15, 1, "mid")
    assert fast.should_post(0.30, 1, "better")
    assert not fast.should_post(0.10, 1, "worse")
    fast.mark_posted("better", p=0.30, units=1)
    assert not fast.should_post(0.30, 1, "better")
    assert not fast.should_post(0.25, 1, "still-worse")

    slow.observe(occupied, me="me")
    assert not slow.we_lead
    assert slow.should_post(0.22, 1, "take")
    slow.mark_posted("take", p=0.22, units=1)
    slow.observe(
        {**ROUND, "leaders": [{"userId": "me", "name": "us", "probability": 0.22, "wordCount": 1}]},
        me="me",
    )
    assert slow.we_lead
    assert not slow.should_post(0.30, 1, "spam")

    assert fast.poll_interval() < 1.0
    assert slow.poll_interval() >= 3.0


def _chain_client(finish_on: str | None = None, closed: bool = False):
    class Client:
        def __init__(self):
            self.ops = []
            self.chain = {
                "id": "a",
                "revisionId": "rev1",
                "playMode": "strict_chain",
                "nextTurn": 1,
                "tokens": [],
                "turns": [],
            }

        def attempt(self, revision_id, mode):
            return self.chain

        def start(self, revision_id, mode):
            return self.chain

        def turn(self, attempt, operation):
            self.ops.append(operation)
            if closed and operation.get("kind") == "append" and len(attempt.get("tokens") or []) >= 1:
                from jevlab.site.client import SiteError

                raise SiteError("round ended", code="round_ended")
            tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
            if operation.get("kind") == "clear":
                tokens = []
            elif operation.get("kind") == "append":
                tokens.append({"id": str(len(tokens)), "text": operation.get("text"), "active": True})
            text = tokens[-1]["text"] if tokens else ""
            reached = finish_on is not None and text == finish_on
            turn = {"probability": 0.18 if reached else 0.05, "reachedYes": reached, "id": "t"}
            self.chain = {
                **attempt,
                "tokens": tokens,
                "turns": [turn],
                "nextTurn": attempt.get("nextTurn", 1) + 1,
            }
            return self.chain, turn

    return Client()


def test_live_post_holds_once_a_word_crosses_the_finish_line():
    question = question_from_round(ROUND)
    client = _chain_client(finish_on="exactly")
    logs = []
    scored = post_phrase(client, question, "strict_chain", "exactly more words here", logs.append, stop_at=0.15)
    assert scored == 0.18
    assert client.ops == [{"kind": "append", "text": "exactly"}]
    assert any("holding" in line for line in logs)

    session = LiveSession(pace="fast", chain="strict")
    session.observe(ROUND, me="me")
    session.finished = True
    assert not session.should_post(0.9, 1, "another line")


def test_live_post_does_not_clear_a_chain_that_already_finished():
    question = question_from_round(ROUND)
    client = _chain_client()
    client.chain = {
        **client.chain,
        "tokens": [{"id": "0", "text": "exactly", "active": True}],
        "turns": [{"probability": 0.18, "reachedYes": True, "id": "t"}],
    }
    logs = []
    scored = post_phrase(client, question, "strict_chain", "a different phrase entirely", logs.append, stop_at=0.15)
    assert scored == 0.18
    assert client.ops == []
    assert any("already crossed" in line for line in logs)


def test_live_post_holds_when_the_site_closes_the_round():
    question = question_from_round(ROUND)
    client = _chain_client(closed=True)
    logs = []
    scored = post_phrase(client, question, "strict_chain", "exactly more words", logs.append, stop_at=0.15)
    assert scored == 0.05
    assert [op["text"] for op in client.ops] == ["exactly", "more"]
    assert any("round closed" in line for line in logs)


def test_live_chain_toggle_keeps_strict_word_by_word():
    strict_round = {**ROUND, "playMode": "strict_chain"}
    casual = LiveSession(pace="fast", chain="casual")
    strict = LiveSession(pace="fast", chain="strict")
    casual.observe(strict_round, me="me")
    strict.observe(ROUND, me="me")
    assert casual.play_mode == "word_chain"
    assert strict.play_mode == "strict_chain"
    assert chain_token(casual.play_mode, "pitch black sky") == "pitch-black-sky"
    assert chain_token(strict.play_mode, "pitch black sky") is None

    class Client:
        def __init__(self):
            self.modes = []
            self.ops = []

        def attempt(self, revision_id, mode):
            self.modes.append(mode)
            return {
                "id": "a",
                "revisionId": revision_id,
                "playMode": mode,
                "nextTurn": 1,
                "tokens": [],
                "turns": [],
            }

        def start(self, revision_id, mode):
            return self.attempt(revision_id, mode)

        def turn(self, attempt, operation):
            self.ops.append(operation)
            tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
            if operation.get("kind") == "append":
                tokens.append({"id": str(len(tokens)), "text": operation.get("text"), "active": True})
            turn = {"probability": 0.2, "id": "t"}
            return {**attempt, "tokens": tokens, "turns": [turn], "nextTurn": 2}, turn

    question = question_from_round(ROUND)
    client = Client()
    post_phrase(client, question, strict.play_mode, "pitch black sky", lambda _message: None)
    assert client.modes == ["strict_chain"]
    assert [op["text"] for op in client.ops] == ["pitch", "black", "sky"]


def test_live_posts_a_phrase_as_one_word_and_prefers_a_saved_line():
    assert chain_token("word_chain", "pitch black volcanic sky") == "pitch-black-volcanic-sky"
    assert chain_token("word_chain", "pitch  black") == "pitch-black"
    assert chain_token("word_chain", "please") == "please"
    assert chain_token("strict_chain", "pitch black volcanic sky") is None

    class Client:
        def __init__(self):
            self.ops = []

        def attempt(self, revision_id, mode):
            return {
                "id": "a",
                "revisionId": revision_id,
                "playMode": mode,
                "nextTurn": 1,
                "tokens": [],
                "turns": [],
            }

        def start(self, revision_id, mode):
            return self.attempt(revision_id, mode)

        def turn(self, attempt, operation):
            self.ops.append(operation)
            tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
            if operation.get("kind") == "append":
                tokens.append({"id": str(len(tokens)), "text": operation.get("text"), "active": True})
            turn = {"probability": 0.22, "id": "t"}
            attempt = {**attempt, "tokens": tokens, "turns": [turn], "nextTurn": attempt.get("nextTurn", 1) + 1}
            return attempt, turn

    question = question_from_round(ROUND)
    client = Client()
    scored = post_phrase(client, question, "word_chain", "pitch black volcanic sky", lambda _message: None)
    assert scored == 0.22
    assert client.ops == [{"kind": "append", "text": "pitch-black-volcanic-sky"}]

    strict = Client()
    post_phrase(strict, question, "strict_chain", "pitch black sky", lambda _message: None)
    assert [op["text"] for op in strict.ops] == ["pitch", "black", "sky"]

    entries = [
        {
            "slug": "round-avocados",
            "title": "Is it stealing to ring up avocados as bananas?",
            "phrase": "pitch black volcanic sky",
            "p_mean": 0.99,
            "units": 4,
            "status": "published",
        },
        {
            "slug": "round-avocados",
            "title": "Is it stealing to ring up avocados as bananas?",
            "phrase": "blue sky overhead",
            "p_mean": 0.2,
            "units": 3,
            "status": "candidate",
        },
        {
            "slug": "other",
            "title": "Something else",
            "phrase": "unrelated",
            "p_mean": 0.995,
            "units": 1,
            "status": "candidate",
        },
    ]
    saved = best_saved(question, entries)
    assert saved["phrase"] == "pitch black volcanic sky"
    assert saved["p"] == 0.99
    assert saved["units"] == 4

    recycled = best_saved({**question, "slug": "round-brand-new"}, entries)
    assert recycled["phrase"] == "pitch black volcanic sky"
    assert recycled["p"] == 0.99

    assert best_saved({**question, "slug": "missing", "title": "Nope"}, entries) is None


def test_sentence_round_submits_the_phrase_in_one_turn():
    assert sentence_words("  Hello,   world! ") == ["Hello", "world"]
    assert sentence_words("yes+") == ["yes+"]

    state = {**ROUND, "playMode": "strict_chain", "sentences": True}
    session = LiveSession(pace="fast", chain="casual")
    session.observe(state, me="me")
    assert session.sentences
    assert session.play_mode == "strict_chain"
    question = session.question
    assert question["raw"]["sentences"]
    assert question["raw"]["playMode"] == "strict_chain"

    class Client:
        def __init__(self):
            self.ops = []

        def attempt(self, revision_id, mode):
            return {
                "id": "a",
                "revisionId": revision_id,
                "playMode": mode,
                "nextTurn": 1,
                "tokens": [],
                "turns": [],
            }

        def start(self, revision_id, mode):
            return self.attempt(revision_id, mode)

        def turn(self, attempt, operation):
            self.ops.append(operation)
            tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
            if operation.get("kind") == "append":
                tokens.extend(
                    {"id": str(len(tokens) + i), "text": word, "active": True}
                    for i, word in enumerate(operation["text"].split())
                )
            turn = {"probability": 0.22, "id": "t"}
            return {**attempt, "tokens": tokens, "turns": [turn], "nextTurn": 2}, turn

    client = Client()
    scored = post_phrase(client, question, session.play_mode, "pitch black sky", lambda _message: None)
    assert scored == 0.22
    assert client.ops == [{"kind": "append", "text": "pitch black sky"}]

    long = " ".join(f"w{i}" for i in range(SENTENCE_WORDS + 3))
    longer = Client()
    post_phrase(longer, question, session.play_mode, long, lambda _message: None)
    assert [op["text"] for op in longer.ops] == [" ".join(f"w{i}" for i in range(SENTENCE_WORDS))]
