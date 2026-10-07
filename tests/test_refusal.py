"""A Luna refusal is kept for a later run and is not retried."""

import asyncio
import json

from jevlab.db import DB
from jevlab.oracle import Oracle

PHRASE = "post the essays"
REQUEST = {
    "model": "openai/gpt-6-luna-decisions",
    "state": "",
    "questions": {"q": {"type": "noul", "instructions": "Will you go out with me?"}},
}
REFUSAL = json.dumps({"error": {"message": 'OpenAI refused to answer question "q"', "code": 502}})


class _Http:
    def __init__(self):
        self.calls = 0

    async def post(self, path, json=None):
        self.calls += 1
        return _Response(502, REFUSAL)

    async def aclose(self):
        return None


class _Response:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text

    def json(self):
        return json.loads(self.text)


def test_a_refusal_is_kept_and_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr("jevlab.oracle.backend_specs", lambda: [("openrouter", "https://openrouter.ai/api/v1", "test")])
    monkeypatch.setattr("jevlab.oracle.asyncio.sleep", lambda *_args, **_kwargs: _asleep())
    http = _Http()
    db = DB(tmp_path / "t.db")
    oracle = Oracle(db, REQUEST, concurrency=4)
    oracle.limiter.backends[0].http = http

    async def run():
        return await oracle.score([PHRASE])

    scores = asyncio.run(run())

    assert http.calls == 1
    assert scores[PHRASE].n == 0
    assert oracle.limiter.backends[0].limiter.limit == 4
    kept = db.all("SELECT state FROM oracle_refusals")
    assert [row["state"] for row in kept] == [PHRASE]


async def _asleep():
    return None
