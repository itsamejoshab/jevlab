"""Remote deterministic editions fill a requested sample count from one API call."""

import asyncio
import json
import time

import pytest

from jevlab.oracle import Oracle, OracleError, build_request, request_hash

from helpers import make_db


def _record(oracle, state, value=0.4):
    request = build_request(oracle.question_request, state)
    key = request_hash(request)
    answer = {"type": "noul", "noul": value}
    oracle.calls += 1
    oracle.db.add_oracle_samples(
        [
            (
                key,
                "perplexity/pplx-decider-v1.2",
                state,
                value,
                json.dumps(answer),
                1,
                0.0,
                time.time(),
                oracle.qkey,
            )
        ]
    )
    oracle._cached(key).append(value)
    return value


def test_a_deterministic_edition_fills_n_from_one_remote_call(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["openrouter"])
    monkeypatch.setattr("jevlab.oracle.OPENROUTER_KEY", "test-key")
    monkeypatch.setattr("jevlab.config.EDITION", "decider")
    oracle = Oracle(db, db.question("q")["jev_request"])
    calls = []

    async def once(state, qkey=None):
        calls.append(state)
        return _record(oracle, state)

    monkeypatch.setattr(oracle, "_one", once)
    try:
        score = asyncio.run(oracle.score_one("woman grills", n=5))
        assert score.n == 5
        assert score.mean == pytest.approx(0.4)
        assert score.spread == 0.0
        assert calls == ["woman grills"]
        assert oracle.calls == 1
        again = asyncio.run(oracle.score_one("woman grills", n=5))
        assert again.n == 5
        assert calls == ["woman grills"]
    finally:
        asyncio.run(oracle.close())
    rows = db.all("SELECT noul FROM oracle_samples")
    assert [row["noul"] for row in rows] == [0.4, 0.4, 0.4, 0.4, 0.4]


def test_a_stored_reading_is_copied_without_another_call(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["openrouter"])
    monkeypatch.setattr("jevlab.oracle.OPENROUTER_KEY", "test-key")
    monkeypatch.setattr("jevlab.config.EDITION", "luna")
    oracle = Oracle(db, db.question("q")["jev_request"])
    _record(oracle, "woman grills", 0.7)
    oracle.calls = 0
    calls = []

    async def once(state, qkey=None):
        calls.append(state)
        return _record(oracle, state)

    monkeypatch.setattr(oracle, "_one", once)
    try:
        score = asyncio.run(oracle.score_one("woman grills", n=5))
    finally:
        asyncio.run(oracle.close())
    assert score.n == 5
    assert score.mean == pytest.approx(0.7)
    assert calls == []
    assert oracle.calls == 0


def test_a_failed_call_is_not_copied(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["openrouter"])
    monkeypatch.setattr("jevlab.oracle.OPENROUTER_KEY", "test-key")
    monkeypatch.setattr("jevlab.config.EDITION", "decider")
    oracle = Oracle(db, db.question("q")["jev_request"])

    async def fail(state, qkey=None):
        raise OracleError("gave up scoring")

    monkeypatch.setattr(oracle, "_one", fail)
    try:
        score = asyncio.run(oracle.score_one("woman grills", n=5))
    finally:
        asyncio.run(oracle.close())
    assert score.n == 0
    assert db.all("SELECT noul FROM oracle_samples") == []


def test_a_noisy_edition_still_asks_for_every_sample(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["openrouter"])
    monkeypatch.setattr("jevlab.oracle.OPENROUTER_KEY", "test-key")
    monkeypatch.setattr("jevlab.config.EDITION", "kev")
    oracle = Oracle(db, db.question("q")["jev_request"])
    calls = []

    async def once(state, qkey=None):
        calls.append(state)
        return _record(oracle, state, 0.2 + 0.01 * len(calls))

    monkeypatch.setattr(oracle, "_one", once)
    try:
        score = asyncio.run(oracle.score_one("woman grills", n=3))
    finally:
        asyncio.run(oracle.close())
    assert calls == ["woman grills", "woman grills", "woman grills"]
    assert score.n == 3
    assert oracle.calls == 3
