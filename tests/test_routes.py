"""Dear models only write tactics they have beaten the cheap pool at."""

from jevlab.search import routes
from jevlab.search.engine import Engine
from jevlab.search.strategies import LLMGenerate

from test_syllogism import engine


def test_an_untuned_pool_still_sends_every_batch_to_the_model_on_turn():
    assert routes.DEAR_ARCHETYPES is None
    assert routes.assign("anthropic/claude-sonnet-5.5", ["scene", "rename"]) == ["scene", "rename"]


def test_a_dear_model_keeps_only_the_tactics_it_wins(monkeypatch):
    monkeypatch.setattr(routes, "DEAR_ARCHETYPES", frozenset({"syllogism", "reductio"}))
    assert routes.assign("anthropic/claude-sonnet-5.5", ["scene", "syllogism", "rename"]) == ["syllogism"]
    assert "qwen/qwen3.8-max-0902" not in routes.DEAR_MODELS
    assert routes.assign("qwen/qwen3.8-max-0902", ["scene", "rename"]) == ["scene", "rename"]
    assert routes.assign("deepseek/deepseek-v4-flash", ["scene", "rename"]) == ["scene", "rename"]


def test_a_skipped_dear_model_hands_the_batch_to_the_next_one(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "DEAR_ARCHETYPES", frozenset({"syllogism"}))
    eng = engine(tmp_path, "route")
    gen = LLMGenerate(["anthropic/claude-sonnet-5.5", "deepseek/deepseek-v4-flash"])
    notes = []
    eng.ctx.log = notes.append
    assert gen.next_model(eng.ctx) == "anthropic/claude-sonnet-5.5"
    model, archetypes = gen.route(eng.ctx, "anthropic/claude-sonnet-5.5", ["scene", "rename"])
    assert model == "deepseek/deepseek-v4-flash"
    assert archetypes == ["scene", "rename"]
    assert notes and "skipped" in notes[0]


def test_an_empty_decision_never_calls_the_dear_model(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "DEAR_ARCHETYPES", frozenset())
    eng: Engine = engine(tmp_path, "none")
    gen = LLMGenerate(["anthropic/claude-sonnet-5.5"])
    eng.ctx.log = lambda _message: None
    model, _archetypes = gen.route(eng.ctx, "anthropic/claude-sonnet-5.5", ["scene"])
    assert model is None
