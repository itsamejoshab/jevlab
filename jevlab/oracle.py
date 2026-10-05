"""OpenRouter System One oracle: the same request the site sends to Jev, scored locally.

Answers come back rounded to 0.01 and vary between calls, so every sample is
kept. `score(states, n)` tops each state up to n samples and returns the mean.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field

import httpx

from . import netlog
from .config import (JEV_MODEL, OPENROUTER_BASE, OPENROUTER_KEY, ORACLE_BACKENDS, ORACLE_CONCURRENCY,
                     ORACLE_MAX_CONCURRENCY, ORACLE_TIMEOUT, TYPESAFE_BASE, TYPESAFE_KEY)
from .db import DB


class OracleError(RuntimeError):
    pass


@dataclass
class Score:
    state: str
    samples: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")

    @property
    def spread(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((s - m) ** 2 for s in self.samples) / (len(self.samples) - 1))

    def lcb(self, z: float = 1.0, floor_sd: float = 0.01) -> float:
        """Lower confidence bound; one sample assumes the measured per-call noise."""
        sd = max(self.spread, floor_sd)
        return self.mean - z * sd / math.sqrt(max(self.n, 1))


def build_request(question_request: dict, state: str, model: str = "") -> dict:
    """The site's `jevRequest` with our phrase as `state`, model pinned to JEV_MODEL (or `model`, to read another
    edition's samples)."""
    request = json.loads(json.dumps(question_request))
    request["state"] = state
    request["model"] = model or JEV_MODEL
    return request


def request_hash(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def question_key(question_request: dict, model: str = "") -> str:
    """Identifies the question independent of the phrase."""
    return request_hash(build_request(question_request, "", model))[:24]


def backfill_qkeys(db: DB, question_request: dict) -> None:
    """Older samples were stored without qkey; match them by recomputing the hash."""
    qkey = question_key(question_request)
    rows = db.all("SELECT DISTINCT state, request_hash FROM oracle_samples WHERE qkey IS NULL")
    updates = [
        (qkey, r["request_hash"])
        for r in rows
        if request_hash(build_request(question_request, r["state"])) == r["request_hash"]
    ]
    if updates:
        db.executemany("UPDATE oracle_samples SET qkey = ? WHERE request_hash = ?", updates)


def target_value(answer_json: str | None, target: str) -> float | None:
    """P(target) from a stored choice answer."""
    try:
        answer = json.loads(answer_json or "")
    except ValueError:
        return None
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict):
        return None
    return float(probabilities.get(target) or 0.0)


def history(db: DB, question_request: dict, target: str = "") -> dict[str, list[float]]:
    """Every oracle sample for this question, grouped by phrase; with `target`, P(target) per sample."""
    backfill_qkeys(db, question_request)
    out: dict[str, list[float]] = {}
    for r in db.all("SELECT state, noul, answer FROM oracle_samples WHERE qkey = ?",
                    (question_key(question_request),)):
        value = target_value(r["answer"], target) if target else r["noul"]
        if value is not None:
            out.setdefault(r["state"], []).append(float(value))
    return out


NOTICE_EVERY = 10.0
# Typesafe's documented slow-down signals (docs.typesafe.ai/api#handling-rate-limits), plus OpenRouter's 503.
THROTTLE_STATUS = {429: "rate limited", 529: "overloaded", 503: "unavailable"}
THROTTLE_BASE = 2.0
THROTTLE_CAP = 60.0
# How long one call keeps waiting out back-offs before giving up; waiting does not use up its ordinary retries.
THROTTLE_MAX_WAIT = 900.0
RETRIES = 6


def retry_after(response: httpx.Response) -> float | None:
    """Seconds from a Retry-After header (a number or an HTTP date)."""
    value = (response.headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return max(parsedate_to_datetime(value).timestamp() - time.time(), 0.0)
    except (TypeError, ValueError):
        return None


class AdaptiveLimiter:
    """AIMD concurrency for one backend: grow slowly on success, halve on failure. A burst of failures from
    calls that were already in flight is one congestion signal, so it halves at most once per BACKOFF_WINDOW."""

    BACKOFF_WINDOW = 2.0

    def __init__(self, start: int, maximum: int):
        self.limit = float(start)
        self.maximum = maximum
        self.cut_at = 0.0
        self.in_flight = 0

    @property
    def free(self) -> int:
        return int(self.limit) - self.in_flight

    def done(self, ok: bool) -> None:
        self.in_flight -= 1
        if ok:
            self.limit = min(self.maximum, self.limit + 1.0 / max(self.limit, 1))
        elif time.monotonic() - self.cut_at >= self.BACKOFF_WINDOW:
            self.limit = max(2.0, self.limit / 2)
            self.cut_at = time.monotonic()


class Backend:
    """One place that serves Jev's /systemone (Typesafe direct or OpenRouter), with its own concurrency."""

    def __init__(self, name: str, base: str, key: str, concurrency: int):
        self.name = name
        self.base = base
        self.local = name == "huggingface"
        self.limiter = AdaptiveLimiter(concurrency, ORACLE_MAX_CONCURRENCY)
        self.http = None
        if not self.local:
            self.http = httpx.AsyncClient(
                base_url=base,
                timeout=ORACLE_TIMEOUT,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                limits=httpx.Limits(max_connections=ORACLE_MAX_CONCURRENCY + 8),
            )
        self.calls = 0
        self.errors = 0
        self.streak = 0  # consecutive failures
        self.resting_until = 0.0
        self.throttles = 0  # consecutive 429/529s
        self.throttled_until = 0.0

    @property
    def resting(self) -> bool:
        return time.monotonic() < self.resting_until

    @property
    def throttled(self) -> bool:
        return time.monotonic() < self.throttled_until

    def throttle(self, retry_after: float | None) -> float:
        """The server asked us to slow down: no calls go here until the wait is over. The wait is what
        Retry-After asks for, else exponential (2s, 4s, ... up to THROTTLE_CAP) with jitter."""
        backoff = min(THROTTLE_CAP, THROTTLE_BASE * 2 ** self.throttles) * random.uniform(0.8, 1.2)
        wait = max(retry_after or 0.0, backoff)
        self.throttles += 1
        self.throttled_until = max(self.throttled_until, time.monotonic() + wait)
        return wait

    def done(self, ok: bool) -> None:
        self.limiter.done(ok)
        if ok:
            self.calls += 1
            self.streak = 0
            self.throttles = 0
            return
        self.errors += 1
        self.streak += 1
        if self.streak >= 3:
            self.resting_until = time.monotonic() + 5.0 * min(self.streak - 2, 6)


def backend_specs() -> list[tuple[str, str, str]]:
    """(name, base, key). ``huggingface`` is a local checkpoint (Laya or Clef) and needs no key."""
    known = {"typesafe": (TYPESAFE_BASE, TYPESAFE_KEY), "openrouter": (OPENROUTER_BASE, OPENROUTER_KEY)}
    specs = []
    for name in ORACLE_BACKENDS:
        if name == "huggingface":
            specs.append(("huggingface", "huggingface", ""))
        elif name in known and known[name][1]:
            specs.append((name, *known[name]))
    return specs


class BackendPool:
    """Spreads calls over every backend with a key: each call goes to the healthy backend with the most free
    slots (round robin on ties), and a backend failing three times in a row rests while the others carry on."""

    def __init__(self, concurrency: int):
        specs = backend_specs()
        if not specs:
            raise OracleError(
                "no Jev backend key: set TYPESAFE_API_KEY and/or OPENROUTER_API_KEY in .env. "
                "Laya and Clef score locally: `uv sync --extra laya` then `jevlab install-laya`, "
                "or `uv sync --extra clef` then `jevlab install-clef`."
            )
        self.backends = [Backend(name, base, key, concurrency) for name, base, key in specs]
        self.cond = asyncio.Condition()
        self.turn = 0

    @property
    def limit(self) -> float:
        return sum(b.limiter.limit for b in self.backends)

    def describe(self) -> str:
        now = time.monotonic()

        def state(b: Backend) -> str:
            if b.throttled:
                return f" backing off {b.throttled_until - now:.0f}s"
            return " resting" if b.resting else ""

        return " ".join(f"{b.name} {int(b.limiter.limit)}{state(b)}" for b in self.backends)

    def _pick(self, avoid: Backend | None) -> Backend | None:
        """None means wait. Throttled backends (429/529) are never used until their wait is over; resting ones
        (failure streaks) only when every other backend is resting too and none is merely backing off, since a
        backend that asked us to wait will be back soon while a resting one is failing."""
        usable = [b for b in self.backends if not b.throttled]
        alive = [b for b in usable if not b.resting]
        if not alive:
            if len(usable) < len(self.backends):
                return None
            alive = [b for b in usable if b.resting_until != float("inf")] or usable
        if avoid is not None and len(alive) > 1:
            fresh = [b for b in alive if b is not avoid and b.limiter.free > 0]
            if fresh:
                alive = fresh
        ready = [b for b in alive if b.limiter.free > 0]
        if not ready:
            return None
        self.turn = (self.turn + 1) % len(self.backends)
        order = self.backends[self.turn:] + self.backends[:self.turn]
        return max(ready, key=lambda b: (b.limiter.free / max(b.limiter.limit, 1), -order.index(b)))

    async def acquire(self, avoid: Backend | None = None) -> Backend:
        async with self.cond:
            while (backend := self._pick(avoid)) is None:
                # A back-off ends on the clock, not on a release, so wake up when the first one is over.
                ends = [b.throttled_until for b in self.backends if b.throttled]
                timeout = max(min(ends) - time.monotonic(), 0.05) if ends else None
                try:
                    await asyncio.wait_for(self.cond.wait(), timeout)
                except TimeoutError:
                    pass
            backend.limiter.in_flight += 1
            return backend

    async def release(self, backend: Backend, ok: bool) -> None:
        async with self.cond:
            backend.done(ok)
            self.cond.notify_all()

    async def close(self) -> None:
        for b in self.backends:
            if b.http is not None:
                await b.http.aclose()


class Oracle:
    """Scores phrases. With `target` (a choice question's answer option) every value is P(target) instead of
    Jev's top-option probability; samples are stored the same way either way. Remote calls are pooled over
    Typesafe and OpenRouter. Laya and Clef use a local Hugging Face checkpoint instead, one forward per phrase."""

    def __init__(self, db: DB, question_request: dict, concurrency: int = ORACLE_CONCURRENCY, target: str = ""):
        self.db = db
        self.question_request = question_request
        self.target = target
        self.limiter = BackendPool(concurrency)
        self.calls = 0
        self.errors = 0
        self.last_error = ""
        self.on_error = None  # callable(str); told about failures, at most once per NOTICE_EVERY per kind
        self._noticed: dict[str, tuple[float, int]] = {}
        self.cost = 0.0
        self.latencies: list[float] = []
        self.started = time.monotonic()
        self.cache: dict[str, list[float]] = {}
        self.qkey = question_key(question_request)

    async def close(self) -> None:
        await self.limiter.close()

    @property
    def rate(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.calls / elapsed if elapsed > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        recent = sorted(self.latencies[-200:])
        return recent[len(recent) // 2] * 1000 if recent else 0.0

    def _cached(self, key: str) -> list[float]:
        if key not in self.cache:
            rows = self.db.all("SELECT noul, answer FROM oracle_samples WHERE request_hash = ?", (key,))
            if self.target:
                values = [target_value(r["answer"], self.target) for r in rows]
            else:
                values = [r["noul"] for r in rows]
            self.cache[key] = [float(v) for v in values if v is not None]
        return self.cache[key]

    @staticmethod
    def answer_value(answer: dict) -> float:
        """One number per answer, in the site's P(yes) sense before goal flipping."""
        kind = answer.get("type")
        if kind == "noul":
            return float(answer["noul"])
        if kind == "choice":
            return float(max((answer.get("probabilities") or {}).values() or [answer.get("confidence") or 0]))
        if kind == "score":
            levels = max(len(answer.get("legend") or {}) - 1, 1)
            return float(answer["score"]) / levels
        raise OracleError(f"unknown answer type {kind!r}")

    async def _one(self, state: str, qkey: str | None = None) -> float:
        request = build_request(self.question_request, state)
        key = request_hash(request)
        last: Backend | None = None
        attempt = 0
        started = time.monotonic()
        while attempt < RETRIES:
            backend = await self.limiter.acquire(avoid=last)
            if backend.local:
                await self.limiter.release(backend, False)
                raise OracleError(
                    "the huggingface backend is local; it does not share a pool with remote /systemone calls"
                )
            ok = False
            throttled = False
            t0 = time.monotonic()
            try:
                response = await backend.http.post("/systemone", json=request)
                if response.status_code == 200:
                    body = response.json()
                    answer = (body.get("answers") or {}).get("q") or next(iter((body.get("answers") or {}).values()), None)
                    if not answer:
                        raise OracleError(f"no answer in {str(body)[:200]}")
                    value = self.answer_value(answer)
                    latency = time.monotonic() - t0
                    cost = float((body.get("usage") or {}).get("cost") or 0)
                    self.calls += 1
                    self.cost += cost
                    self.latencies.append(latency)
                    self.db.execute(
                        "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (key, body.get("model"), state, value, json.dumps(answer), int(latency * 1000), cost,
                         time.time(), qkey or self.qkey),
                    )
                    if self.target:
                        value = float((answer.get("probabilities") or {}).get(self.target) or 0.0)
                    self._cached(key).append(value)
                    ok = True
                    return value
                if response.status_code == 400:
                    raise OracleError(self.failed(backend, response.status_code, response.text))
                if response.status_code in (401, 402, 403):
                    # Bad key or no credit: this backend is out for good; the others can still serve.
                    message = self.failed(backend, response.status_code, response.text)
                    backend.resting_until = float("inf")
                    if all(b.resting_until == float("inf") for b in self.limiter.backends):
                        raise OracleError(message)
                elif response.status_code in THROTTLE_STATUS:
                    asked = retry_after(response)
                    wait = backend.throttle(asked)
                    throttled = True
                    why = THROTTLE_STATUS[response.status_code]
                    self.failed(backend, response.status_code,
                                f"{why}; {backend.name} backs off {wait:.0f}s"
                                f"{f' (Retry-After {asked:.0f}s)' if asked is not None else ''}: {response.text}",
                                action="waiting it out")
                else:
                    self.failed(backend, response.status_code, response.text)
            except httpx.TimeoutException as error:
                self.failed(backend, "timeout", f"no reply within {ORACLE_TIMEOUT:g}s, so we stopped waiting "
                                                f"(the server sent no status; {type(error).__name__})")
            except httpx.TransportError as error:
                self.failed(backend, type(error).__name__, str(error))
            finally:
                await self.limiter.release(backend, ok)
            last = backend
            if throttled:
                # Being told to slow down is not a failed try; keep waiting, up to THROTTLE_MAX_WAIT per call.
                if time.monotonic() - started < THROTTLE_MAX_WAIT:
                    continue
            attempt += 1
            others = [b for b in self.limiter.backends if b is not backend and not b.resting]
            await asyncio.sleep(0.1 if others else 0.5 * 2**attempt)
        raise OracleError(f"gave up scoring {state[:60]!r} after {RETRIES} tries; last: {self.last_error}")

    def failed(self, backend: Backend, status: int | str, detail: str, action: str = "retrying elsewhere") -> str:
        self.errors += 1
        self.last_error = netlog.record("jev-oracle", backend.base, "/systemone", status, detail)
        kind = str(status)
        at, quiet = self._noticed.get(kind, (0.0, 0))
        now = time.monotonic()
        if now - at >= NOTICE_EVERY:
            more = f" (+{quiet} more since last notice)" if quiet else ""
            if self.on_error:
                self.on_error(f"[net] {self.last_error}{more}; {action}, "
                              f"concurrency {self.limiter.describe()}")
            self._noticed[kind] = (now, 0)
        else:
            self._noticed[kind] = (at, quiet + 1)
        return self.last_error

    def _answer_of(self, body: dict) -> dict:
        answer = (body.get("answers") or {}).get("q") or next(iter((body.get("answers") or {}).values()), None)
        if not isinstance(answer, dict):
            raise OracleError(f"no answer in {str(body)[:200]}")
        return answer

    def _store(self, key: str, state: str, stored: float, answer: dict, model: str, qkey: str | None,
               latency: float, copies: int, shown: float) -> None:
        """Write `copies` identical samples. The in-memory cache keeps `shown` (P(target) when aiming)."""
        now = time.time()
        qk = qkey or self.qkey
        payload = json.dumps(answer)
        ms = int(latency * 1000)
        self.db.executemany(
            "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(key, model, state, stored, payload, ms, 0.0, now, qk)] * copies,
        )
        self._cached(key).extend([shown] * copies)

    def _shown(self, answer: dict, stored: float) -> float:
        if self.target:
            return float((answer.get("probabilities") or {}).get(self.target) or 0.0)
        return stored

    async def _score_local(self, states: list[str], n: int, qkey: str | None) -> dict[str, Score]:
        """A local checkpoint is deterministic, so one forward fills every sample a caller asked for."""
        from .config import EDITION

        if EDITION == "clef":
            from .clef_local import ClefError as LocalError, predict_many
            who = "Clef"
        else:
            from .laya_local import LayaError as LocalError, predict_many
            who = "Laya"

        unique = list(dict.fromkeys(states))
        keys = {state: request_hash(build_request(self.question_request, state)) for state in unique}
        pending = []
        for state in unique:
            have = self._cached(keys[state])
            if len(have) >= n:
                continue
            if have:
                self._duplicate(keys[state], n - len(have))
            else:
                pending.append(state)
        if pending:
            questions = build_request(self.question_request, "").get("questions") or {}
            started = time.monotonic()
            try:
                bodies = await asyncio.to_thread(predict_many, pending, questions)
            except LocalError as error:
                raise OracleError(str(error)) from error
            if len(bodies) != len(pending):
                raise OracleError(f"{who} returned {len(bodies)} answers for {len(pending)} phrases")
            latency = time.monotonic() - started
            for state, body in zip(pending, bodies):
                answer = self._answer_of(body)
                stored = self.answer_value(answer)
                self._store(
                    keys[state], state, stored, answer, str(body.get("model") or ""), qkey, latency, n,
                    self._shown(answer, stored),
                )
                self.calls += 1
                self.latencies.append(latency)
        return {state: Score(state, list(self._cached(keys[state]))) for state in unique}

    def _duplicate(self, key: str, copies: int) -> None:
        """Repeat a stored local reading so a later run sees the same sample count, without another forward."""
        row = self.db.one(
            "SELECT model, state, noul, answer, latency_ms, qkey FROM oracle_samples "
            "WHERE request_hash = ? ORDER BY rowid DESC LIMIT 1",
            (key,),
        )
        if row is None or copies <= 0:
            return
        now = time.time()
        self.db.executemany(
            "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(key, row["model"], row["state"], row["noul"], row["answer"], row["latency_ms"], 0.0, now, row["qkey"])]
            * copies,
        )
        shown = self._cached(key)[-1]
        self._cached(key).extend([shown] * copies)

    async def score(self, states: list[str], n: int = 1, qkey: str | None = None) -> dict[str, Score]:
        """Top every state up to n samples. Duplicate states are scored once. `qkey` files the samples under
        another key, so measurement lines stay out of the question's history."""
        if self.limiter.backends and all(backend.local for backend in self.limiter.backends):
            return await self._score_local(states, n, qkey)
        unique = list(dict.fromkeys(states))
        keys = {s: request_hash(build_request(self.question_request, s)) for s in unique}
        tasks = []
        for state in unique:
            need = n - len(self._cached(keys[state]))
            tasks.extend(self._one(state, qkey) for _ in range(max(need, 0)))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            fatal = [r for r in results if isinstance(r, OracleError) and "HTTP 4" in str(r)]
            if fatal:
                raise fatal[0]
        return {s: Score(s, list(self._cached(keys[s]))) for s in unique}

    async def score_one(self, state: str, n: int = 1) -> Score:
        return (await self.score([state], n))[state]
