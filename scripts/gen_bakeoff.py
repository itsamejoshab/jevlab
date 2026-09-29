"""Head-to-head of generator models: same questions, same prompt per round, every phrase scored by the oracle."""

import asyncio
import json
import statistics
import sys
import time

from jevlab.db import DB
from jevlab.search.engine import Engine
from jevlab.search.strategies import LLMGenerate

MODELS = [
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.5-flash-02-23",
    "google/gemini-2.5-flash-lite",
    "mistralai/mistral-nemo",
    "cohere/command-r7b-12-2024",
    "z-ai/glm-5.3-flash",
]
QUESTIONS = sys.argv[1:] or [
    "do-you-think-dogs-know-we-re-naked",
    "point-break-is-the-greatest-movie-of-all-time",
    "is-the-cake-a-lie",
    "should-we-nerf-whoever-is-winning",
]
ROUNDS = 3


async def bake(slug: str) -> dict:
    engine = Engine(DB(), slug, budget=8000, use_llm=True)
    await engine.start()
    ctx = engine.ctx
    gen = LLMGenerate(MODELS, 0)
    out = {m: {"p": [], "secs": [], "empty": 0} for m in MODELS}
    for rnd in range(ROUNDS):
        archetypes = gen.pick_archetypes(ctx, fresh=rnd == 0)
        variants = [] if rnd == 0 else [c.phrase for c in ctx.archive.top_by_board(3)]

        async def one(model):
            t = time.monotonic()
            got = await gen._call(ctx, model, archetypes, variants, fresh=rnd == 0)
            return model, got, time.monotonic() - t

        for model, proposals, secs in await asyncio.gather(*(one(m) for m in MODELS)):
            out[model]["secs"].append(secs)
            if not proposals:
                out[model]["empty"] += 1
                continue
            scored = await ctx.evaluate([(text, "bakeoff:" + model.split("/")[-1], "", tactic)
                                         for text, tactic in proposals])
            out[model]["p"] += [c.p for c in scored]
    lead = engine.leader
    await engine.close()
    return {"slug": slug, "leader": lead.probability if lead else 0, "models": out}


def top_mean(ps, k=5):
    return statistics.mean(sorted(ps, reverse=True)[:k]) if ps else 0.0


async def main():
    results = [await bake(slug) for slug in QUESTIONS]
    json.dump(results, open("data/gen_bakeoff.json", "w"), indent=1)
    ranks = {m: [] for m in MODELS}
    for res in results:
        print(f"\n{res['slug']} (leader {res['leader']:.2f})")
        rows = []
        for m, d in res["models"].items():
            ps = d["p"]
            rows.append((top_mean(ps), m, len(ps), statistics.mean(ps) if ps else 0, max(ps, default=0),
                         statistics.mean(d["secs"]), d["empty"]))
        rows.sort(reverse=True)
        for rank, (top5, m, n, mean, best, secs, empty) in enumerate(rows, 1):
            ranks[m].append(rank)
            print(f"  {rank}. {m.split('/')[-1]:<28} top5 {top5:.3f}  best {best:.3f}  mean {mean:.3f}  "
                  f"n {n:3}  {secs:5.1f}s/batch" + (f"  empty {empty}" if empty else ""))
    print("\naverage rank by top-5 score (lower is better):")
    for m, r in sorted(ranks.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"  {m.split('/')[-1]:<28} {statistics.mean(r):.2f}  {r}")


asyncio.run(main())
