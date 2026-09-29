"""OpenRouter chat completions for phrase generation and planning."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from . import netlog
from .config import GEN_REASONING, LLM_TIMEOUT, OPENROUTER_BASE, OPENROUTER_KEY, PLAN_MODEL, PROVIDER_SORT, gen_model


class LLMError(RuntimeError):
    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient  # worth retrying: timeouts, network errors, 429/5xx, unparseable replies


TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no JSON in reply: {text[:200]!r}", transient=True)


class LLM:
    def __init__(self):
        if not OPENROUTER_KEY:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.http = httpx.AsyncClient(
            base_url=OPENROUTER_BASE,
            timeout=LLM_TIMEOUT * 2,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                "X-Title": "jevlab",
            },
        )
        self.calls = 0
        self.failures = 0
        self.on_error = None  # callable(str), told about every failed call
        self.cost = 0.0

    async def close(self) -> None:
        await self.http.aclose()

    def failed(self, model: str, status: int | str, detail: str) -> str:
        message = netlog.record("llm", OPENROUTER_BASE, "/chat/completions", status, f"[{model}] {detail}")
        if self.on_error:
            self.on_error(f"[net] {message}")
        return message

    async def complete(self, model: str, system: str, user: str, temperature: float = 1.0,
                       max_tokens: int = 4000, json_mode: bool = True) -> str:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if model != PLAN_MODEL:
            spec = gen_model(model)
            effort = spec.reasoning if spec else GEN_REASONING
            if spec and spec.max_tokens:
                body["max_tokens"] = max(max_tokens, spec.max_tokens)
            if effort:
                body["reasoning"] = {"effort": effort}
            if PROVIDER_SORT:
                body["provider"] = {"sort": PROVIDER_SORT}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        timeout = LLM_TIMEOUT * (2 if model == PLAN_MODEL else 1)
        try:
            response = await asyncio.wait_for(self.http.post("/chat/completions", json=body), timeout)
            if response.status_code in (400, 422) and "valid model" not in response.text:
                if "mandatory" in response.text and body.get("reasoning", {}).get("effort") == "none":
                    body["reasoning"] = {"effort": "minimal"}
                elif response.status_code == 422:
                    body.pop("reasoning", None)
                else:
                    body.pop("response_format", None)
                    body.pop("reasoning", None)
                response = await asyncio.wait_for(self.http.post("/chat/completions", json=body), timeout)
        except (TimeoutError, asyncio.TimeoutError) as error:
            self.failures += 1
            raise LLMError(self.failed(model, "timeout", f"no reply after {timeout:.0f}s"), transient=True) from error
        except httpx.HTTPError as error:
            self.failures += 1
            raise LLMError(self.failed(model, type(error).__name__, str(error)), transient=True) from error
        if response.status_code != 200:
            self.failures += 1
            raise LLMError(self.failed(model, response.status_code, response.text[:300]),
                           transient=response.status_code in TRANSIENT_STATUS)
        try:
            data = response.json()
        except ValueError as error:
            raise LLMError(f"{model} non-JSON body: {response.text[:200]!r}", transient=True) from error
        self.calls += 1
        self.cost += float((data.get("usage") or {}).get("cost") or 0)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"{model} bad reply: {str(data)[:300]}", transient=True) from error

    async def json(self, model: str, system: str, user: str, temperature: float = 1.0, max_tokens: int = 4000,
                   attempts: int = 3):
        """complete() + extract_json(), retrying unparseable or empty replies and transient HTTP errors
        (timeouts, 429, 5xx). Other 4xx errors are real request problems and fail at once."""
        for attempt in range(attempts):
            try:
                return extract_json(await self.complete(model, system, user, temperature, max_tokens))
            except LLMError as error:
                if not error.transient or attempt == attempts - 1:
                    raise
                await asyncio.sleep(1 + attempt * 2)
