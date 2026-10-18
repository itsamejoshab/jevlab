"""The search bar reports the last minute of network trouble, and stays blank after a single blip."""

import jevlab.netlog as netlog


def _clock(monkeypatch, start=0.0):
    now = {"t": start}
    monkeypatch.setattr(netlog.time, "monotonic", lambda: now["t"])
    return now


def _quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(netlog, "PATH", tmp_path / "net.log")
    netlog.counts.clear()
    netlog._recent.clear()


def _fail(status="timeout", url="https://openrouter.ai/api/v1", service="jev-oracle"):
    netlog.record(service, url, "/systemone", status, "no reply")


def test_one_or_two_failures_stay_off_the_bar(monkeypatch, tmp_path):
    _quiet(monkeypatch, tmp_path)
    _fail()
    _fail()
    assert netlog.live_line() == ""
    assert netlog.live_line((48, 96)) == ""
    assert "x2" in netlog.summary()


def test_a_burst_shows_the_last_minute_then_clears(monkeypatch, tmp_path):
    _quiet(monkeypatch, tmp_path)
    now = _clock(monkeypatch)
    for _ in range(3):
        _fail("timeout")
    for _ in range(2):
        _fail("ConnectError")
    assert netlog.live_line() == "net/min openrouter timeout×3 ConnectError×2"
    now["t"] += 61
    assert netlog.live_line() == ""
    assert "x3" in netlog.summary()


def test_other_services_and_hosts_keep_their_names(monkeypatch, tmp_path):
    _quiet(monkeypatch, tmp_path)
    for _ in range(4):
        _fail(503, "https://api.typesafe.ai/v1")
    for _ in range(3):
        _fail("timeout", "https://openrouter.ai/api/v1", service="llm")
    assert netlog.live_line() == "net/min typesafe 503×4 · llm openrouter timeout×3"


def test_concurrency_shows_only_after_the_limiter_cuts(monkeypatch, tmp_path):
    _quiet(monkeypatch, tmp_path)
    for _ in range(3):
        _fail()
    assert netlog.live_line((96, 96)) == "net/min openrouter timeout×3"
    assert netlog.live_line((48, 96)) == "net/min openrouter timeout×3 · conc 48/96"
