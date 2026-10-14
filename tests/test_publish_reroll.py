import jevlab.config as config
from jevlab.objective import Objective
from jevlab.publish import pause_for_fresh_score, reroll_limits, row_listed, should_reroll, wait_for_row


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
