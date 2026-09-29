"""Universal boosters (universal adversarial triggers): short word sequences that raise P(goal) across
many questions. Mined for free from every sample we already have, optimised across questions with the
oracle, and applied to the top lines of any board as prefix, suffix, or clause insert."""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict

from ..db import DB
from ..objective import Objective, goal_p, logit
from ..oracle import Oracle, history
from ..rules import RuleError, check_phrase
from .strategies import CLAUSE_BREAKS, FUNCTION_WORDS
from .mechanical import LevelStrategy

SCHEMA = """
CREATE TABLE IF NOT EXISTS boosters (
    goal TEXT,
    text TEXT,
    lift REAL,
    questions INTEGER,
    source TEXT,
    tries INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    updated REAL,
    PRIMARY KEY (goal, text)
)
"""


def ensure(db: DB) -> None:
    db.execute(SCHEMA)


def library(db: DB, goal: str, k: int = 30) -> list[dict]:
    """Best boosters for a goal: optimised and proven ones first, then mined lift."""
    ensure(db)
    rows = db.all("SELECT * FROM boosters WHERE goal = ?", (goal,))
    def rank(r):
        proven = (r["wins"] + 1) / (r["tries"] + 2)
        return (r["lift"] > 0, r["source"] == "optimised", proven, r["lift"])
    return [dict(r) for r in sorted(rows, key=rank, reverse=True)[:k] if r["lift"] > 0]


def credit(db: DB, goal: str, text: str, won: bool) -> None:
    db.execute("UPDATE boosters SET tries = tries + 1, wins = wins + ?, updated = ? WHERE goal = ? AND text = ?",
               (int(won), time.time(), goal, text))


def _question_rows(db: DB) -> list[dict]:
    out = []
    for q in db.all("SELECT slug FROM questions WHERE kind = 'noul'"):
        question = db.question(q["slug"])
        if question and question.get("jev_request"):
            out.append(question)
    return out


def mine(db: DB, max_n: int = 3, min_questions: int = 3, min_phrases: int = 15, keep: int = 60,
         log=print) -> dict[str, int]:
    """Within each question, compare mean logit P(goal) of phrases containing an n-gram against those that
    do not; average that lift over the questions where it appears. Confounded, but a strong prior."""
    ensure(db)
    stats: dict[str, dict[str, list[float]]] = {"yes": defaultdict(list), "no": defaultdict(list)}
    counts: dict[str, dict[str, int]] = {"yes": defaultdict(int), "no": defaultdict(int)}
    for question in _question_rows(db):
        goal = question.get("goal") or "yes"
        samples = history(db, question["jev_request"])
        if len(samples) < 60:
            continue
        values = {s: logit(goal_p(sum(v) / len(v), goal)) for s, v in samples.items() if v}
        mean_all = sum(values.values()) / len(values)
        grams: dict[str, list[float]] = defaultdict(list)
        for phrase, value in values.items():
            words = phrase.lower().split()
            seen = set()
            for n in range(1, max_n + 1):
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i : i + n])
                    if gram not in seen and not all(w in FUNCTION_WORDS for w in gram.split()):
                        seen.add(gram)
                        grams[gram].append(value)
        total = sum(values.values())
        for gram, vals in grams.items():
            if len(vals) < 5 or len(vals) >= len(values):
                continue
            with_mean = sum(vals) / len(vals)
            without_mean = (total - sum(vals)) / (len(values) - len(vals))
            stats[goal][gram].append(with_mean - without_mean)
            counts[goal][gram] += len(vals)
        del mean_all
    saved = {}
    for goal in ("yes", "no"):
        ranked = []
        for gram, lifts in stats[goal].items():
            if len(lifts) >= min_questions and counts[goal][gram] >= min_phrases:
                lift = sum(lifts) / len(lifts)
                if lift <= 0:
                    continue
                ranked.append((lift * math.sqrt(len(lifts)), lift, len(lifts), gram))
        ranked.sort(reverse=True)
        for _, lift, nq, gram in ranked[:keep]:
            db.execute("INSERT INTO boosters (goal, text, lift, questions, source, updated) VALUES (?, ?, ?, ?, ?, ?) "
                       "ON CONFLICT(goal, text) DO UPDATE SET lift = excluded.lift, questions = excluded.questions, "
                       "updated = excluded.updated WHERE boosters.source = 'mined'",
                       (goal, gram, lift, nq, "mined", time.time()))
        saved[goal] = min(len(ranked), keep)
        log(f"mined {goal}: {len(ranked)} candidate n-grams, kept {saved[goal]}; top: "
            + ", ".join(f"{g} {lift:+.2f}x{n}" for _, lift, n, g in ranked[:8]))
    return saved


async def optimise(db: DB, goal: str, k: int = 3, questions: int = 12, budget: int = 20000, log=print) -> dict:
    """Coordinate ascent on a k-word trigger, as prefix and suffix, maximising mean logit gain over
    `questions` boards (each using our best line there as the base)."""
    from . import vocab

    ensure(db)
    pool = []
    for question in _question_rows(db):
        if (question.get("goal") or "yes") != goal:
            continue
        samples = history(db, question["jev_request"])
        if len(samples) < 40:
            continue
        objective = Objective(goal)
        scored = sorted(((s, goal_p(sum(v) / len(v), goal)) for s, v in samples.items()
                         if v and len(s.split()) <= 40), key=lambda sv: sv[1])
        best = scored[-1][0]
        mid = scored[int(len(scored) * 0.6)][0]
        pool.append((question, [best, mid] if mid != best else [best], objective))
    pool = pool[:questions]
    if len(pool) < 3:
        log(f"only {len(pool)} {goal} questions with data; run some searches first")
        return {}
    oracles = [Oracle(db, q["jev_request"]) for q, _, _ in pool]
    spent = 0

    async def gain(triggers: list[str]) -> dict[str, float]:
        nonlocal spent
        per_trigger = {t: [] for t in triggers}
        async def one(index: int):
            question, bases, objective = pool[index]
            states = list(bases)
            for base in bases:
                for t in triggers:
                    states += [f"{t} {base}", f"{base} {t}"]
            scores = await oracles[index].score(states, 1)
            for base in bases:
                b = logit(objective.p(scores[base]))
                for t in triggers:
                    best = max(logit(objective.p(scores[f"{t} {base}"])),
                               logit(objective.p(scores[f"{base} {t}"])))
                    per_trigger[t].append(best - b)
        await asyncio.gather(*(one(i) for i in range(len(pool))))
        spent += len(triggers) * 2 * sum(len(b) for _, b, _ in pool)
        return {t: sum(v) / len(v) for t, v in per_trigger.items()}

    mined = library(db, goal, 10)
    words_vocab = vocab.global_words(db)
    start = (mined[0]["text"].split() if mined else ["hereafter"])[:k]
    trigger = start + ["the"] * (k - len(start))
    current = (await gain([" ".join(trigger)]))[" ".join(trigger)]
    log(f"optimise {goal}: {len(pool)} questions, start '{' '.join(trigger)}' gain {current:+.3f}")
    try:
        for sweep in range(2):
            for slot in range(k):
                remaining = budget - spent
                per = remaining // max(1, (2 * k - slot) * 4 * len(pool))
                if per < 10:
                    break
                options = [w for w in words_vocab[:per] if w != trigger[slot]]
                cands = {" ".join(trigger[:slot] + [w] + trigger[slot + 1 :]): w for w in options}
                valid = {}
                for text, w in cands.items():
                    try:
                        valid[check_phrase(text)] = w
                    except RuleError:
                        continue
                results = await gain(list(valid))
                best_text = max(results, key=results.get) if results else None
                if best_text and results[best_text] > current:
                    trigger[slot] = valid[best_text]
                    current = results[best_text]
                    log(f"  sweep {sweep + 1} slot {slot + 1}: '{' '.join(trigger)}' gain {current:+.3f} "
                        f"({spent} calls)")
    finally:
        for oracle in oracles:
            await oracle.close()
    text = " ".join(trigger)
    if current <= 0:
        log(f"best trigger '{text}' has gain {current:+.3f} (hurts on average); not saved, {spent} calls")
        return {"text": text, "gain": current, "calls": spent}
    db.execute("INSERT OR REPLACE INTO boosters (goal, text, lift, questions, source, tries, wins, updated) "
               "VALUES (?, ?, ?, ?, 'optimised', 0, 0, ?)", (goal, text, current, len(pool), time.time()))
    log(f"saved booster [{goal}] '{text}' mean logit gain {current:+.3f} over {len(pool)} questions, {spent} calls")
    return {"text": text, "gain": current, "calls": spent}


class Boosters(LevelStrategy):
    """L3. Apply the booster library to our top lines as prefix, suffix, and clause insert."""

    name = "boosters"
    level = 3

    def __init__(self, parents: int = 5, triggers: int = 20):
        self.parents = parents
        self.triggers = triggers
        self.lib: list[dict] | None = None
        self.used: set[str] = set()

    async def load(self, ctx) -> list[dict]:
        if self.lib is None:
            db = ctx.engine.db
            goal = ctx.objective.goal
            lib = library(db, goal, self.triggers)
            if not lib:
                ctx.log("[boosters] library empty; mining boosters from existing samples")
                await ctx.in_thread(mine, db, 3, 3, 15, 60, ctx.log)
                lib = library(db, goal, self.triggers)
            self.lib = lib
        return self.lib

    def available(self, ctx) -> bool:
        return super().available(ctx) and any(c.phrase not in self.used for c in ctx.archive.top_by_board(self.parents))

    async def run(self, ctx) -> None:
        lib = await self.load(ctx)
        if not lib:
            ctx.log("[boosters] no boosters for this goal yet")
            self.used |= {c.phrase for c in ctx.archive.top_by_board(self.parents)}
            return
        parents = [c for c in ctx.archive.top_by_board(self.parents) if c.phrase not in self.used]
        items, origin_of = [], {}
        for parent in parents:
            self.used.add(parent.phrase)
            words = parent.words
            breaks = [i for i, w in enumerate(words) if w.casefold() in CLAUSE_BREAKS][:2]
            for row in lib:
                t = row["text"]
                variants = [f"{t} {parent.phrase}", f"{parent.phrase} {t}"]
                variants += [" ".join(words[:i] + t.split() + words[i:]) for i in breaks]
                for v in variants:
                    items.append((v, self.name, parent.phrase, parent.archetype))
                    origin_of.setdefault(v, (t, parent))
        scored = await ctx.evaluate(items)
        winners = set()
        for cand in scored:
            source = origin_of.get(cand.phrase)
            if source and cand.fitness > source[1].fitness:
                winners.add(source[0])
        for row in lib:
            credit(ctx.engine.db, ctx.objective.goal, row["text"], row["text"] in winners)
        ctx.log(f"[boosters] {len(scored)} lines from {len(lib)} boosters x {len(parents)} parents; "
                f"{len(winners)} boosters beat their parent")
