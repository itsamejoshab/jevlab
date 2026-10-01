from jevlab.publish import should_reroll


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
