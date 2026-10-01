"""Head-to-head of generator models: same questions, same prompt per round, every phrase scored by the oracle."""

import asyncio
import json
import statistics
import sys
import time

from jevlab.db import DB
from jevlab.search.engine import Engine
from jevlab.search.strategies import LLMGenerate

# One model per voice. Versions of the same family are skipped. Cost is a later cut.
MODELS = [
    # fluent scene-builders
    "google/gemini-3.8-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemma-4-31b-it",
    # flat and competent
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3.8-max-0902",
    # short, spoken, slang
    "x-ai/grok-4.3",
    "x-ai/grok-4.7",
    # clause-heavy, careful
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-fable-5",
    # stacked, literary
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k3",
    # stiff, definitional
    "cohere/command-r7b-12-2024",
    "cohere/command-r-plus-08-2024",
    "cohere/command-a",
    # compact, slightly translated
    "mistralai/mistral-nemo",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-saba",
    "mistralai/mixtral-8x22b-instruct",
    # chatty product voices
    "openai/gpt-4o-mini",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "amazon/nova-pro-v1",
    # open instruct, terse or purple
    "meta-llama/llama-4-maverick",
    "microsoft/phi-4",
    "gryphe/mythomax-l2-13b",
    "nousresearch/hermes-3-llama-3.1-70b",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    "writer/palmyra-x5",
    "minimax/minimax-m2-her",
    "upstage/solar-pro-3",
    # GLM
    "z-ai/glm-5.3-flash",
    "z-ai/glm-5.3",
]
# Chosen from the current snapshot and vault, not from the first boards we happened to search.
# Four are behind a 0.99 leader with our best still short of that score, so models have room to separate.
# The steak question is one we already lead with two words: a model that cannot get near that known win is out.
QUESTIONS = sys.argv[1:] or [
    "is-git-rebase-gaslighting",  # 0.81 vs 0.99/31w, four runs, still the widest gap
    "round-should-you-iron-your-shirts-with-a-waffle-iron-01e79468",  # live round, 0.83 vs 0.99/40w
    "is-a-skeleton-just-a-body-doing-minimalism",  # one run, 0.92 vs 0.99/34w
    "does-the-following-prompt-require-us-to-spend-a-zillion-doll",  # long prompt, 0.91 vs 0.98/231w
    "is-a-well-done-steak-acceptable",  # we lead 0.99/2w; the solved check
]
ROUNDS = 3


async def bake(slug: str) -> dict:
    engine = Engine(DB(), slug, budget=25000, use_llm=True)
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

        print(f"  {slug} round {rnd + 1}/{ROUNDS}", flush=True)
        for model, proposals, secs in await asyncio.gather(*(one(m) for m in MODELS)):
            out[model]["secs"].append(secs)
            if not proposals:
                out[model]["empty"] += 1
                print(f"    empty {model.split('/')[-1]} {secs:.0f}s", flush=True)
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
