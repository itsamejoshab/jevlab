import jevlab.config as config
from jevlab.publish import pause_for_fresh_score, reroll_limits, should_reroll


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
