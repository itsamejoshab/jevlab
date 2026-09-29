"""Offline benchmarks on data we already have: embedding models for the predictor, stand-in models for Jev."""

from __future__ import annotations

import asyncio
import json
import math
import random
import time

import numpy as np

from .config import DATA
from .db import DB
from .objective import goal_p, logit
from .oracle import history

EMBED_CANDIDATES = [
    "local",
    "openrouter:qwen/qwen3-embedding-8b",
    "openrouter:google/gemini-embedding-2",
    "openrouter:voyageai/voyage-4",
    "openrouter:baai/bge-m3",
    "openrouter:openai/text-embedding-3-large",
]
PROXY_CANDIDATES = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "deepseek/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.5-plus-20260420",
    "x-ai/grok-4.7",
    "typesafe/jev-router",
]


def sample_questions(db: DB, per_question: int, questions: int = 6, seed: int = 7) -> list[tuple[dict, dict]]:
    """Questions with the most scored phrases, each with up to `per_question` (phrase -> P(goal)) pairs."""
    rng = random.Random(seed)
    rows = []
    for q in db.all("SELECT slug FROM questions WHERE kind = 'noul'"):
        question = db.question(q["slug"])
        if not question or not question.get("jev_request"):
            continue
        samples = history(db, question["jev_request"])
        if len(samples) >= 300:
            rows.append((len(samples), question, samples))
    rows.sort(key=lambda r: -r[0])
    out = []
    for _, question, samples in rows[:questions]:
        goal = question.get("goal") or "yes"
        items = list(samples.items())
        rng.shuffle(items)
        data = {s: goal_p(sum(v) / len(v), goal) for s, v in items[:per_question] if v}
        out.append((question, data))
    return out


def holdout_rho(embedder, data: dict[str, float], folds: int = 5, seed: int = 0) -> float:
    from .search.surrogate import PhraseSurrogate, spearman

    phrases = list(data)
    X = PhraseSurrogate(embedder).features(phrases)
    y = np.array([logit(data[p], 0.005) for p in phrases])
    from sklearn.linear_model import BayesianRidge

    order = np.random.default_rng(seed).permutation(len(phrases))
    rhos = []
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        model = BayesianRidge().fit(X[train], y[train])
        rhos.append(spearman(model.predict(X[test]), y[test]))
    return float(np.mean(rhos))


async def bench_embed(models: list[str] | None = None, phrases: int = 3000) -> int:
    from .search.surrogate import Embedder

    db = DB()
    sets = sample_questions(db, max(100, phrases // 6))
    print(f"{len(sets)} questions, {sum(len(d) for _, d in sets)} phrases; 5-fold holdout Spearman rho of the "
          "phrase predictor (BayesianRidge on embedding + length)\n")
    results = {}
    for name in models or EMBED_CANDIDATES:
        started = time.time()
        try:
            embedder = await asyncio.to_thread(Embedder, name)
            if name != "local" and embedder.kind in ("bge-small", "hash"):
                raise RuntimeError("backend unavailable (fell back to local)")
            rhos = [await asyncio.to_thread(holdout_rho, embedder, data) for _, data in sets]
        except Exception as error:
            print(f"  {name:<45} failed: {str(error)[:100]}")
            continue
        cost = getattr(embedder.model, "cost", 0.0)
        results[name] = float(np.mean(rhos))
        per = " ".join(f"{r:.2f}" for r in rhos)
        print(f"  {name:<45} rho {results[name]:.3f}  [{per}]  dim {embedder.dim}  "
              f"{time.time() - started:5.1f}s  ${cost:.4f}")
    if results:
        best = max(results, key=results.get)
        print(f"\nbest: {best} (rho {results[best]:.3f}). Set JEV_EMBED_MODEL={best} in .env to use it.")
        (DATA / "bench_embed.json").write_text(json.dumps({"at": time.time(), "results": results}, indent=2))
    return 0


def proxy_prompt(question: dict, phrase: str) -> list[dict]:
    instructions = question["jev_request"]["questions"]["q"]["instructions"]
    return [
        {"role": "system", "content": "You read a context and answer a yes/no question about it. "
                                      "Reply with exactly one word: yes or no."},
        {"role": "user", "content": f"Context: {phrase}\n\nQuestion: {instructions}\nAnswer (yes or no):"},
    ]


def p_yes_from_logprobs(choice: dict) -> float | None:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    if not content:
        return None
    yes = no = 0.0
    for alt in content[0].get("top_logprobs") or []:
        token = str(alt.get("token", "")).strip().lower()
        if token.startswith("yes"):
            yes += math.exp(alt["logprob"])
        elif token.startswith("no"):
            no += math.exp(alt["logprob"])
    if yes + no == 0:
        return None
    return yes / (yes + no)


async def bench_proxy(models: list[str] | None = None, phrases: int = 400) -> int:
    import httpx

    from .config import OPENROUTER_BASE, OPENROUTER_KEY
    from .search.surrogate import spearman

    db = DB()
    sets = sample_questions(db, max(40, phrases // 5), questions=5, seed=11)
    items = [(q, s, p) for q, data in sets for s, p in data.items()]
    print(f"{len(items)} phrases over {len(sets)} questions. Stand-in P(goal) from yes/no token logprobs vs oracle.\n")
    http = httpx.AsyncClient(base_url=OPENROUTER_BASE, timeout=60,
                             headers={"Authorization": f"Bearer {OPENROUTER_KEY}"})
    gate = asyncio.Semaphore(24)
    results = {}
    try:
        for model in models or PROXY_CANDIDATES:
            cost = 0.0
            texts: list[str] = []

            async def one(question, phrase):
                nonlocal cost
                body = {"model": model, "messages": proxy_prompt(question, phrase), "max_tokens": 1,
                        "temperature": 0, "logprobs": True, "top_logprobs": 5, "usage": {"include": True},
                        "reasoning": {"effort": "none"}, "provider": {"require_parameters": True}}
                async with gate:
                    try:
                        response = await http.post("/chat/completions", json=body)
                        if response.status_code == 404 and "No endpoints" in response.text:
                            body.pop("reasoning")  # non-thinking models: no provider "supports" the knob
                            response = await http.post("/chat/completions", json=body)
                        if response.status_code == 400 and "mandatory" in response.text:
                            # Thinking can't be turned off: let it think, then read the answer text.
                            body.pop("reasoning")
                            body["max_tokens"] = 2000
                            response = await http.post("/chat/completions", json=body)
                        data = response.json()
                    except Exception:
                        return None
                if response.status_code != 200:
                    if len(texts) < 1:
                        texts.append(str(data)[:160])
                    return None
                cost += float((data.get("usage") or {}).get("cost") or 0)
                choice = (data.get("choices") or [{}])[0]
                p = p_yes_from_logprobs(choice)
                if p is None:
                    reply = str((choice.get("message") or {}).get("content") or "").strip().lower()
                    if len(texts) < 3:
                        texts.append(reply[:40])
                    if reply.startswith("yes"):
                        p = 0.99
                    elif reply.startswith("no"):
                        p = 0.01
                    else:
                        return None
                goal = question.get("goal") or "yes"
                return goal_p(p, goal)

            started = time.time()
            got = await asyncio.gather(*(one(q, s) for q, s, _ in items))
            pairs = [(g, p) for g, (_, _, p) in zip(got, items) if g is not None]
            if len(pairs) < 20:
                print(f"  {model:<36} unusable ({len(pairs)} answers) {texts[:2]}")
                continue
            per_q = []
            for question, _ in sets:
                sub = [(g, p) for g, (q, s, p) in zip(got, items) if q is question and g is not None]
                if len(sub) >= 10:
                    per_q.append(spearman(np.array([a for a, _ in sub]), np.array([b for _, b in sub])))
            rho = float(np.mean(per_q)) if per_q else 0.0
            results[model] = rho
            print(f"  {model:<36} rho {rho:.3f} per question [{' '.join(f'{r:.2f}' for r in per_q)}]  "
                  f"answered {len(pairs)}/{len(items)}  {time.time() - started:5.1f}s  ${cost:.4f}"
                  + (f"  (no logprobs, sample replies {texts[:2]})" if texts else ""))
    finally:
        await http.aclose()
    if results:
        best = max(results, key=results.get)
        verdict = "passes the 0.6 gate" if results[best] >= 0.6 else "below the 0.6 gate; ProxyScreen stays off"
        print(f"\nbest stand-in: {best} rho {results[best]:.3f} ({verdict})")
        (DATA / "bench_proxy.json").write_text(json.dumps({"at": time.time(), "results": results}, indent=2))
    return 0
