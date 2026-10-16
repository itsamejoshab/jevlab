"""Which generator models earn their price, per archetype.

Each cell is one tactic, one model, one question: the same prompt the search would
send, scored by the oracle. Rows append to jevlab/data/route_bakeoff.jsonl, so a stopped
run resumes. Qwen3.8 Max is not in the pool.

    uv run python scripts/route_bakeoff.py
    uv run python scripts/route_bakeoff.py --set tough
    uv run python scripts/route_bakeoff.py --summarize
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from jevlab.config import DATA
from jevlab.db import DB
from jevlab.llm import LLMError, call_costs
from jevlab.modes import is_searchable
from jevlab.rules import RuleError, check_phrase, normalize
from jevlab.search.engine import Engine
from jevlab.search.prompts import ARCHETYPE_GUIDE, gen_system, gen_user
from jevlab.search.strategies import LLMGenerate

OUT = DATA / "route_bakeoff.jsonl"

# Price bands. Qwen3.8 Max is intentionally absent.
CHEAP = (
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4-flash",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-haiku-4.5",
)
MID = ("google/gemini-3.8-flash",)
# Fast stand-ins for Qwen3.8 Max. Neither is the Max model.
REPLACE = (
    "qwen/qwen3.6-flash",
    "google/gemini-3.1-flash-lite",
)
DEAR = ("anthropic/claude-sonnet-5.5",)
# GLM and Gemini reject reasoning "none"; the others waste money if thinking is on.
REASONING = {
    "z-ai/glm-5.3-flash": "minimal",
    "google/gemini-3.8-flash": "minimal",
    "google/gemini-3.1-flash-lite": "minimal",
}
# Leader still ahead of our best measured line, after a real search.
TOUGH = (
    "round-should-you-iron-your-shirts-with-a-waffle-iron-01e79468",
    "is-a-skeleton-just-a-body-doing-minimalism",
    "does-the-following-prompt-require-us-to-spend-a-zillion-doll",
    "is-git-rebase-gaslighting",
)
# Searchable boards outside the first round. The zero-sample ones are all scales.
# The choice question is the least-sampled unranked choice (ranked choices cannot be searched).
SPARSE = (
    "how-good-is-sharknado",
    "how-do-you-like-your-steak",
    "youtube-or-twitch",
    "man-or-woman-lifts-weights",
    "someone-says-let-s-take-this-offline-they-mean",
)
# Point Break is where Qwen Max's one wide win happened. The replacement models get that board too.
COVER = ("point-break-is-the-greatest-movie-of-all-time",)
SETS = {
    "next": SPARSE + TOUGH + COVER,
    "sparse": SPARSE,
    "tough": TOUGH,
    "all": SPARSE + TOUGH + COVER,
}
# Two rounded points, above the oracle's 0.01 step and its per-call noise.
WORTH_IT = 0.02


def models() -> list[str]:
    return list(CHEAP) + list(MID) + list(REPLACE) + list(DEAR)


def pool_models() -> set[str]:
    return set(models())


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add(f"{row['slug']}|{row['archetype']}|{row['model']}")
    return done


def top_mean(values: list[float], k: int = 3) -> float:
    if not values:
        return 0.0
    return statistics.mean(sorted(values, reverse=True)[:k])


def phrases_of(data) -> list[str]:
    rows = data.get("phrases") if isinstance(data, dict) else data
    out = []
    for row in rows or []:
        if isinstance(row, dict):
            text = str(row.get("text") or row.get("phrase") or "")
        else:
            text = str(row)
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


async def call_model(llm, model: str, system: str, user: str) -> tuple[list[str], float, float, str]:
    bucket: list[float] = []
    token = call_costs.set(bucket)
    started = time.monotonic()
    try:
        data = await llm.json(
            model, system, user, temperature=1.0, max_tokens=2500, reasoning=REASONING.get(model, "none")
        )
        return phrases_of(data), sum(bucket), time.monotonic() - started, ""
    except (LLMError, Exception) as error:
        return [], sum(bucket), time.monotonic() - started, str(error)[:240]
    finally:
        call_costs.reset(token)


def legal(engine, text: str) -> str:
    try:
        return check_phrase(normalize(text), engine.choices, engine.length_cap)
    except RuleError:
        return ""


async def bake(questions: list[str], archetypes: list[str], pool: list[str], count: int) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUT)
    handle = OUT.open("a")
    spent = 0.0
    try:
        for slug in questions:
            print(f"\n{slug}", flush=True)
            db = DB()
            question = db.question(slug)
            ranked = bool((question or {}).get("raw", {}).get("ranked"))
            if question is None or not is_searchable(question.get("kind"), ranked):
                print(f"  skip {slug}: not a searchable yes/no, scale, or unranked choice", flush=True)
                continue
            engine = Engine(db, slug, budget=80000, use_llm=True)
            await engine.start()
            for strategy in engine.strategies.values():
                if isinstance(strategy, LLMGenerate):
                    strategy.cancel()
            if engine.train_task is not None and not engine.train_task.done():
                engine.train_task.cancel()
            system = gen_system(engine.ctx)
            try:
                for archetype in archetypes:
                    user = gen_user(engine.ctx, [archetype], count)
                    needed = [model for model in pool if f"{slug}|{archetype}|{model}" not in done]
                    if not needed:
                        continue
                    print(f"  {archetype} ({len(needed)} models)", flush=True)
                    calls = await asyncio.gather(
                        *(call_model(engine.llm, model, system, user) for model in needed)
                    )
                    owned: list[tuple[str, list[str]]] = []
                    every: list[str] = []
                    for model, (texts, _cost, _secs, _error) in zip(needed, calls):
                        phrases = []
                        for text in texts:
                            phrase = legal(engine, text)
                            if phrase and phrase not in phrases:
                                phrases.append(phrase)
                                if phrase not in every:
                                    every.append(phrase)
                        owned.append((model, phrases))
                    scored_map = {}
                    if every:
                        raw = await engine.oracle.score(every, 1)
                        for phrase, score in raw.items():
                            if score.n:
                                scored_map[phrase] = (engine.objective.p(score), engine.objective.units(phrase))
                    for model, (_texts, cost, secs, error) in zip(needed, calls):
                        phrases = dict(owned)[model]
                        scored = [(phrase, *scored_map[phrase]) for phrase in phrases if phrase in scored_map]
                        values = [p for _, p, _ in scored]
                        row = {
                            "slug": slug,
                            "archetype": archetype,
                            "model": model,
                            "cost": round(cost, 6),
                            "secs": round(secs, 1),
                            "n": len(scored),
                            "error": error,
                            "best": round(max(values, default=0.0), 4),
                            "top3": round(top_mean(values), 4),
                            "mean": round(statistics.mean(values), 4) if values else 0.0,
                            "phrases": [{"text": text, "p": round(p, 4), "units": units} for text, p, units in scored],
                        }
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        done.add(f"{slug}|{archetype}|{model}")
                        spent += cost
                        flag = " empty" if error or not values else ""
                        print(
                            f"    {model.split('/')[-1]:<24} top3 {row['top3']:.3f} best {row['best']:.3f} "
                            f"n {row['n']:2} ${cost:.4f} {secs:4.0f}s{flag}",
                            flush=True,
                        )
            finally:
                engine.rounds = 0
                await engine.close()
    finally:
        handle.close()
    print(f"\nspent ${spent:.2f} this process", flush=True)
    summarize(OUT)


def band_top3(rows: list[dict], band: tuple[str, ...]) -> float:
    values = [row["top3"] for row in rows if row["model"] in band and row["n"]]
    return max(values) if values else 0.0


def summarize(path: Path) -> None:
    if not path.exists():
        print("no results")
        return
    grouped: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["archetype"], []).append(row)
    print("\narchetype gaps (dear top3 − best of cheap or mid; positive means the dear model won):")
    premium = []
    for archetype in ARCHETYPE_GUIDE:
        rows = grouped.get(archetype) or []
        if not rows:
            continue
        by_question: dict[str, list[dict]] = {}
        for row in rows:
            by_question.setdefault(row["slug"], []).append(row)
        gaps = []
        for question_rows in by_question.values():
            dear_rows = [row for row in question_rows if row["model"] in DEAR and row["n"]]
            other_rows = [
                row for row in question_rows if row["model"] in pool_models() and row["model"] not in DEAR and row["n"]
            ]
            if not dear_rows or not other_rows:
                continue
            dear = max(row["top3"] for row in dear_rows)
            other = max(row["top3"] for row in other_rows)
            gaps.append(dear - other)
        if not gaps:
            continue
        gap = statistics.mean(gaps)
        mark = "DEAR" if gap >= WORTH_IT else "cheap"
        if gap >= WORTH_IT:
            premium.append(archetype)
        print(f"  {mark:<5} {archetype:<12} {gap:+.3f} over {len(gaps)} questions")
    cost = sum(row["cost"] for rows in grouped.values() for row in rows)
    print(f"\nlogged cost ${cost:.2f}")
    print("dear-only archetypes:", ", ".join(premium) or "(none)")
    recommend(grouped)


def recommend(grouped: dict[str, list[dict]]) -> None:
    """Per tactic, the fastest model within 0.02 of the best top-3. That is the speed-to-leader pick."""
    print("\nspeed-to-leader pick (fastest model within 0.02 of the best top-3 on that tactic):")
    picks: dict[str, list[str]] = {}
    for archetype in ARCHETYPE_GUIDE:
        rows = [row for row in grouped.get(archetype) or [] if row["model"] in pool_models() and row["n"]]
        if not rows:
            continue
        by_model: dict[str, list[dict]] = {}
        for row in rows:
            by_model.setdefault(row["model"], []).append(row)
        summary = []
        for model, group in by_model.items():
            summary.append(
                (
                    statistics.mean(row["top3"] for row in group),
                    statistics.mean(row["secs"] for row in group),
                    statistics.mean(row["cost"] for row in group),
                    len(group),
                    model,
                )
            )
        best = max(item[0] for item in summary)
        close = [item for item in summary if item[0] >= best - WORTH_IT]
        top3, secs, cost, n, model = min(close, key=lambda item: (item[1], item[2]))
        short = model.split("/")[-1]
        picks.setdefault(short, []).append(archetype)
        gap = best - top3
        print(f"  {short:<24} {archetype:<12} top3 {top3:.3f}  {secs:4.0f}s  ${cost:.4f}  nq {n}  behind best {gap:.3f}")
    print("\npicks:")
    for short, names in sorted(picks.items(), key=lambda item: -len(item[1])):
        print(f"  {len(names):2} {short}: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=tuple(SETS), default="next")
    parser.add_argument("--questions", default="", help="comma-separated slugs; overrides --set")
    parser.add_argument("--archetypes", default=",".join(ARCHETYPE_GUIDE))
    parser.add_argument("--models", default=",".join(models()))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(OUT)
        return
    questions = [part for part in args.questions.split(",") if part] or list(SETS[args.set])
    archetypes = [part for part in args.archetypes.split(",") if part]
    pool = [part for part in args.models.split(",") if part]
    unknown = [name for name in archetypes if name not in ARCHETYPE_GUIDE]
    if unknown:
        raise SystemExit(f"unknown archetypes: {', '.join(unknown)}")
    asyncio.run(bake(questions, archetypes, pool, args.count))


if __name__ == "__main__":
    main()
