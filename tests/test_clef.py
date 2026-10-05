"""Clef scores through a local Hugging Face checkpoint, with weights kept outside the repo."""

import asyncio
from pathlib import Path

import pytest

from jevlab import clef_local
from jevlab.config import ROOT
from jevlab.clef_local import ClefError, cache_dir, prepare, resolve_quant, weights_message
from jevlab.oracle import Oracle, backend_specs

from helpers import make_db


def test_default_cache_is_outside_the_repo():
    path = cache_dir()
    assert path.is_absolute()
    assert not path.is_relative_to(ROOT)
    if "JEV_CLEF_CACHE" not in __import__("os").environ:
        assert path == Path.home() / ".cache" / "jevlab" / "huggingface"


def test_missing_runtime_names_the_install(monkeypatch):
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("no hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(ClefError, match="uv sync --extra clef"):
        clef_local._snapshot(download=False)


def test_small_gpu_uses_4bit(monkeypatch):
    monkeypatch.setattr(clef_local, "CLEF_QUANT", "auto")
    monkeypatch.setattr(clef_local, "_cuda_total_gib", lambda: 12.0)
    assert resolve_quant("cuda") == "4bit"


def test_tiny_gpu_offloads(monkeypatch):
    monkeypatch.setattr(clef_local, "CLEF_QUANT", "auto")
    monkeypatch.setattr(clef_local, "_cuda_total_gib", lambda: 4.0)
    assert resolve_quant("cuda") == "offload"


def test_mid_gpu_uses_4bit(monkeypatch):
    monkeypatch.setattr(clef_local, "CLEF_QUANT", "auto")
    monkeypatch.setattr(clef_local, "_cuda_total_gib", lambda: 24.0)
    assert resolve_quant("cuda") == "4bit"


def test_large_gpu_keeps_full_weights(monkeypatch):
    monkeypatch.setattr(clef_local, "CLEF_QUANT", "auto")
    monkeypatch.setattr(clef_local, "_cuda_total_gib", lambda: 80.0)
    assert resolve_quant("cuda") == "none"


def test_4bit_refuses_cpu(monkeypatch):
    monkeypatch.setattr(clef_local, "CLEF_QUANT", "4bit")
    with pytest.raises(ClefError, match="CUDA"):
        prepare("cpu")


def test_local_oracle_fills_every_sample_from_one_forward(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.oracle.ORACLE_BACKENDS", ["huggingface"])
    monkeypatch.setattr("jevlab.config.EDITION", "clef")
    calls = []

    def predict_many(states, questions):
        calls.append((list(states), questions))
        return [
            {"model": "clef", "answers": {"q": {"type": "noul", "noul": 0.4}}, "usage": {}}
            for _ in states
        ]

    monkeypatch.setattr(clef_local, "predict_many", predict_many)
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
    assert backend_specs() == [("huggingface", "huggingface", "")]


def test_missing_weights_message_names_the_cache():
    text = weights_message()
    assert "jevlab install-clef" in text
    assert str(cache_dir()) in text
