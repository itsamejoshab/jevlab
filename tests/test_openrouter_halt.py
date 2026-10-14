"""A search stops when OpenRouter answers 403 Forbidden because the key is out of budget.

The writers, the planner, and the embedder all use that key. A forbidden reply is not a
bad phrase: the run ends, and a queue does not start the next question.
"""

import asyncio

import httpx

from jevlab.llm import LLMError
from jevlab.search.scheduler import Scheduler

from test_syllogism import engine

HALT = "openrouter key limit"
FORBIDDEN = "Client error '403 Forbidden' for url 'https://openrouter.ai/api/v1/embeddings'"
KEY_LIMIT = 'llm HTTP 403 from openrouter.ai/chat/completions: {"error":{"message":"Key limit exceeded (total limit)"}}'


class _Calls:
    calls = 0

    async def close(self):
        pass


def _run(eng, strategy):
    eng.strategies = {strategy.name: strategy}
    eng.scheduler = Scheduler([strategy.name])
    logs = []

    async def start():
        eng.oracle = _Calls()

    eng.start = start
    eng.on_event = lambda event: logs.append(event.data.get("message", "")) if event.kind == "log" else None
    asyncio.run(eng.run())
    return logs


class _Raise:
    name = "boom"
    runs = 0

    def available(self, ctx):
        return True

    async def run(self, ctx):
        type(self).runs += 1
        if type(self).runs >= 3:
            ctx.engine.stopping = True
            return
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError(FORBIDDEN, request=request, response=response)


class _Swallow:
    name = "boom"
    runs = 0

    def available(self, ctx):
        return True

    async def run(self, ctx):
        type(self).runs += 1
        if type(self).runs >= 3:
            ctx.engine.stopping = True
            return
        try:
            raise LLMError(KEY_LIMIT)
        except LLMError:
            return


class _Ordinary:
    name = "boom"
    runs = 0

    def available(self, ctx):
        return True

    async def run(self, ctx):
        type(self).runs += 1
        if type(self).runs >= 2:
            ctx.engine.stopping = True
            return
        raise LLMError("no JSON in reply", transient=True)


def test_an_ordinary_failure_keeps_searching(tmp_path):
    eng = engine(tmp_path, "ordinary")
    _Ordinary.runs = 0
    _run(eng, _Ordinary())
    assert _Ordinary.runs == 2
    assert eng.end_reason != HALT


def test_a_forbidden_embedding_stops_the_search(tmp_path):
    eng = engine(tmp_path, "embed")
    _Raise.runs = 0
    logs = _run(eng, _Raise())
    assert _Raise.runs == 1
    assert eng.end_reason == HALT
    assert any("key limit" in message.lower() for message in logs)


def test_a_swallowed_writer_refusal_still_stops_the_search(tmp_path):
    eng = engine(tmp_path, "writer")
    _Swallow.runs = 0
    logs = _run(eng, _Swallow())
    assert _Swallow.runs == 1
    assert eng.end_reason == HALT
    assert any("key limit" in message.lower() for message in logs)
