"""Laya scores through a local Hugging Face checkpoint, with weights kept outside the repo."""

import asyncio
from pathlib import Path

import pytest

from jevlab import laya_local
from jevlab.config import ROOT
from jevlab.laya_local import LayaError, cache_dir, weights_message
from jevlab.oracle import Oracle, backend_specs

from helpers import make_db


def test_default_cache_is_outside_the_repo():
    path = cache_dir()
    assert path.is_absolute()
    assert not path.is_relative_to(ROOT)
    if "JEV_LAYA_CACHE" not in __import__("os").environ:
        assert path == Path.home() / ".cache" / "jevlab" / "huggingface"


def test_missing_runtime_names_the_install(monkeypatch):
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("no hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(LayaError, match="uv sync --extra laya"):
        laya_local._snapshot(download=False)


def test_huggingface_backend_needs_no_key(monkeypatch):
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["huggingface"])
    assert backend_specs() == [("huggingface", "huggingface", "")]


def test_local_oracle_fills_every_sample_from_one_forward(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["huggingface"])
    calls = []

    def predict_many(states, questions):
        calls.append((list(states), questions))
        return [
            {"model": "convaiinnovations/laya", "answers": {"q": {"type": "noul", "noul": 0.4}}, "usage": {}}
            for _ in states
        ]

    monkeypatch.setattr(laya_local, "predict_many", predict_many)
    db = make_db(tmp_path)
    question = db.question("q")
    oracle = Oracle(db, question["jev_request"])
    try:
        score = asyncio.run(oracle.score_one("woman grills", n=3))
    finally:
        asyncio.run(oracle.close())
    assert score.n == 3
    assert score.mean == pytest.approx(0.4)
    assert oracle.calls == 1
    assert calls[0][0] == ["woman grills"]
    again = asyncio.run(oracle.score_one("woman grills", n=3))
    assert again.n == 3
    assert len(calls) == 1
    rows = db.all("SELECT noul FROM oracle_samples")
    assert [row["noul"] for row in rows] == [0.4, 0.4, 0.4]


def test_missing_weights_message_names_the_cache():
    text = weights_message()
    assert "jevlab install-laya" in text
    assert str(cache_dir()) in text


def test_predict_many_bounds_each_forward(monkeypatch):
    """A search round of ~90 phrases must not be one forward. That is what filled RAM and swap."""
    cap = laya_local.LAYA_BATCH
    assert cap == 8
    states = [f"p{i}" for i in range(cap * 2 + 1)]
    forwards = []

    class Agent:
        def system_one(self, state, questions):
            raise AssertionError("several states share predict_batch")

        def predict_batch(self, batch, questions, batch_size=None):
            step = batch_size if batch_size and batch_size > 0 else len(batch)
            if step > cap:
                pytest.fail(f"one forward would score {step} states; the cap is {cap}")
            out = []
            for start in range(0, len(batch), step):
                chunk = batch[start : start + step]
                if len(chunk) > cap:
                    pytest.fail(f"one forward would score {len(chunk)} states; the cap is {cap}")
                forwards.append(list(chunk))
                out.extend({"model": "laya", "answers": {"q": state}, "usage": {}} for state in chunk)
            return out

    monkeypatch.setattr(laya_local, "load_agent", lambda: Agent())
    results = laya_local.predict_many(states, {"q": {"type": "noul"}})
    assert [row["answers"]["q"] for row in results] == states
    scored = [state for chunk in forwards for state in chunk]
    assert scored == states
    assert forwards and all(1 <= len(chunk) <= cap for chunk in forwards)
