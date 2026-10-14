import jevlab.config as config
from jevlab.objective import Objective
from jevlab.publish import (
    SENTENCE_WORDS,
    append_texts,
    build_chain,
    pause_for_fresh_score,
    reroll,
    reroll_limits,
    row_listed,
    should_reroll,
    wait_for_row,
)


def test_requested_rerolls_run_after_a_win():
    assert should_reroll(0, asked=3, ceiling=3, ahead=True)
    assert should_reroll(2, asked=3, ceiling=3, ahead=True)
    assert not should_reroll(3, asked=3, ceiling=3, ahead=True)


def test_extra_budget_continues_only_while_short():
    assert not should_reroll(3, asked=3, ceiling=12, ahead=True)
    assert should_reroll(3, asked=3, ceiling=12, ahead=False)
    assert should_reroll(11, asked=3, ceiling=12, ahead=False)
    assert not should_reroll(12, asked=3, ceiling=12, ahead=False)


def test_zero_rerolls_does_not_chase_a_win():
    assert not should_reroll(0, asked=0, ceiling=0, ahead=False)
    assert should_reroll(0, asked=0, ceiling=12, ahead=False)
    assert not should_reroll(0, asked=0, ceiling=12, ahead=True)


def test_deterministic_editions_do_not_chase(monkeypatch):
    monkeypatch.setattr(config, "EDITION", "laya")
    assert config.default_rerolls() == 0
    assert reroll_limits(0, 5, gamble=True, long_shot=False) == (0, 0)
    assert reroll_limits(0, 5, gamble=False, long_shot=True) == (0, 0)
    assert reroll_limits(4, 5, gamble=True, long_shot=False) == (4, 4)


def test_noisy_editions_still_chase(monkeypatch):
    for edition in ("jev", "kev", "microsoft"):
        monkeypatch.setattr(config, "EDITION", edition)
        assert config.default_rerolls() == 5
        assert reroll_limits(5, 5, gamble=True, long_shot=False) == (5, 12)
        assert reroll_limits(5, 5, gamble=False, long_shot=True) == (5, 5)
        assert reroll_limits(0, 5, gamble=False, long_shot=True) == (0, 5)


def _row(p: float, units: int = 4, user: str = "me") -> dict:
    return {"userId": user, "probability": p, "wordCount": units}


def test_a_row_is_listed_when_ours_is_at_least_the_score_we_posted():
    objective = Objective("yes")
    assert row_listed([_row(0.97)], "me", 0.97, 4, objective)
    assert row_listed([_row(0.99, 40)], "me", 0.90, 4, objective)
    assert not row_listed([_row(0.97, 40)], "me", 0.97, 4, objective)
    assert not row_listed([_row(0.97, user="them")], "me", 0.97, 4, objective)
    assert not row_listed([], "me", 0.97, 4, objective)


def test_the_board_is_read_until_the_row_shows_up(monkeypatch):
    boards = {"highScores": [_row(0.50)]}
    reads = {"n": 0}

    class Client:
        def leaderboards(self, revision_id: str, mode: str) -> dict:
            reads["n"] += 1
            if reads["n"] >= 3:
                boards["highScores"] = [_row(0.97)]
            return {"leaderboards": boards}

    slept: list[float] = []
    clock = {"t": 0.0}
    monkeypatch.setattr(
        "jevlab.publish.time.sleep",
        lambda seconds: slept.append(seconds) or clock.__setitem__("t", clock["t"] + seconds),
    )
    monkeypatch.setattr("jevlab.publish.time.monotonic", lambda: clock["t"])
    notes: list[str] = []

    class DB:
        def latest_snapshot(self):
            return {"id": 1}

        def executemany(self, sql, rows):
            self.saved = list(rows)

    db = DB()
    wait_for_row(db, Client(), {"revisionId": "r", "slug": "q"}, "golf", "me", 0.97, 4, Objective("yes"), notes.append)
    assert reads["n"] == 3
    assert slept == [0.5, 1.0]
    assert any("listed" in note for note in notes)
    assert db.saved


def test_a_board_that_never_lists_the_row_stops_at_the_cap(monkeypatch):
    class Client:
        def leaderboards(self, revision_id: str, mode: str) -> dict:
            return {"leaderboards": {"highScores": [_row(0.10)]}}

    clock = {"t": 0.0}

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    monkeypatch.setattr("jevlab.publish.time.sleep", sleep)
    monkeypatch.setattr("jevlab.publish.time.monotonic", lambda: clock["t"])
    notes: list[str] = []
    wait_for_row(
        None, Client(), {"revisionId": "r", "slug": "q"}, "golf", "me", 0.97, 4, Objective("yes"), notes.append
    )
    assert clock["t"] <= 20.0
    assert clock["t"] >= 19.0
    assert any("had not listed" in note for note in notes)


def test_only_microsoft_waits_between_rerolls(monkeypatch):
    slept = []
    monkeypatch.setattr("jevlab.publish.time.sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(config, "EDITION", "kev")
    pause_for_fresh_score(lambda _message: None)
    assert slept == []
    assert config.reroll_gap() == 0
    monkeypatch.setattr(config, "EDITION", "microsoft")
    pause_for_fresh_score(lambda _message: None)
    assert slept == [config.MICROSOFT_REROLL_GAP]
    assert config.reroll_gap() == config.MICROSOFT_REROLL_GAP


class _Chain:
    def __init__(self):
        self.ops: list[str] = []
        self.attempt = {
            "id": "a",
            "revisionId": "r",
            "playMode": "strict_chain",
            "nextTurn": 1,
            "tokens": [],
            "turns": [],
        }

    def turn(self, attempt, operation):
        self.ops.append(operation["kind"] + ":" + str(operation.get("text") or operation.get("tokenId") or ""))
        tokens = [token for token in attempt.get("tokens") or [] if token.get("active", True)]
        if operation["kind"] == "append":
            tokens.append({"id": f"t{len(self.ops)}", "text": operation["text"], "active": True})
        elif operation["kind"] == "remove":
            tokens = [token for token in tokens if token["id"] != operation["tokenId"]]
        elif operation["kind"] == "clear":
            tokens = []
        turn = {"id": f"n{len(self.ops)}", "probability": 0.5}
        self.attempt = {**attempt, "tokens": tokens, "turns": [turn], "nextTurn": int(attempt.get("nextTurn") or 1) + 1}
        return self.attempt, turn


def test_publish_chain_sends_the_sentence_then_the_last_word():
    words = [f"w{i}" for i in range(SENTENCE_WORDS + 2)]
    pieces = append_texts(words, hold_last=True)
    assert pieces == [" ".join(words[:SENTENCE_WORDS]), words[-2], words[-1]]
    assert append_texts(["only"], hold_last=True) == ["only"]
    assert append_texts(["pitch", "black", "sky"], hold_last=False) == ["pitch", "black", "sky"]

    site = _Chain()
    attempt, score = build_chain(site, site.attempt, ["alpha", "beta", "gamma"], lambda _message: None, hold_last=True)
    assert score == 0.5
    assert site.ops == ["append:alpha beta", "append:gamma"]
    attempt, again = reroll(site, attempt, lambda _message: None)
    assert again == 0.5
    assert site.ops[-2].startswith("remove:")
    assert site.ops[-1] == "append:gamma"
    assert [token["text"] for token in attempt["tokens"]] == ["alpha beta", "gamma"]
