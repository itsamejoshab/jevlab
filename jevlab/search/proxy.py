"""ProxyScreen: a cheap stand-in model reads each candidate and its yes/no token logprobs estimate P(goal).

`jevlab bench-proxy` found qwen3-235b-a22b-2507 ranks phrases within a question at rho 0.67 against the
oracle, for about $0.009 per 1000 phrases. The trained per-question predictor does better once it exists, so
the stand-in only ranks candidate pools before that predictor is trusted (new questions, fresh starts)."""

from __future__ import annotations

import asyncio

from ..bench import p_yes_from_logprobs, proxy_prompt
from ..config import PROXY_MODEL
from ..objective import goal_p


class ProxyScreen:
    MAX_POOL = 240  # phrases scored per screen; larger pools are sampled down first
    CONCURRENCY = 16
    TIMEOUT = 45.0

    def __init__(self, llm, question: dict, goal: str, model: str = PROXY_MODEL):
        self.llm = llm
        self.question = question
        self.goal = goal
        self.model = model
        self.cache: dict[str, float] = {}
        self.gate = asyncio.Semaphore(self.CONCURRENCY)
        self.failures = 0
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.model) and self.llm is not None and self.failures < 50

    async def _one(self, phrase: str) -> float | None:
        body = {"model": self.model, "messages": proxy_prompt(self.question, phrase), "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": 5,
                "provider": {"require_parameters": True}}
        async with self.gate:
            try:
                response = await self.llm.http.post("/chat/completions", json=body)
            except Exception:
                self.failures += 1
                return None
        if response.status_code != 200:
            self.failures += 1
            return None
        self.calls += 1
        p = p_yes_from_logprobs((response.json().get("choices") or [{}])[0])
        return None if p is None else goal_p(p, self.goal)

    async def scores(self, phrases: list[str]) -> dict[str, float]:
        """P(goal) per phrase from the stand-in; phrases it could not score are left out."""
        todo = [p for p in phrases if p not in self.cache]
        if todo and self.enabled:
            try:
                got = await asyncio.wait_for(asyncio.gather(*(self._one(p) for p in todo)), self.TIMEOUT)
            except (TimeoutError, asyncio.TimeoutError):
                got = [None] * len(todo)
            for phrase, p in zip(todo, got):
                if p is not None:
                    self.cache[phrase] = p
        return {p: self.cache[p] for p in phrases if p in self.cache}
