"""Search strategies. Each `run(ctx)` proposes phrases and scores them through `ctx.evaluate`."""

from __future__ import annotations

import asyncio
import random
import time

from ..config import EXPENSIVE_MODELS, TIER_LEVEL, TIER_NAMES
from ..llm import LLMError
from ..objective import site_round
from . import prompts
from .archive import ARCHETYPES

FUNCTION_WORDS = [
    "the", "a", "our", "my", "this", "is", "are", "so", "and", "which", "that", "here", "means", "always",
    "only", "every", "no", "not", "yes", "we", "i", "it", "of", "in", "at", "with", "for", "one", "all",
]
CLAUSE_BREAKS = {"and", "so", "which", "that", "because", "but", "then", "hereafter", "therefore", "where"}
# Long-chain regime: at or below this many words, word-level strategies take over from chain compaction.
WORD_STAGE = 60


class Strategy:
    name = "base"
    needs_llm = False

    def available(self, ctx) -> bool:
        return (not self.needs_llm or ctx.llm is not None) and len(ctx.archive) > 0

    async def run(self, ctx) -> None:
        raise NotImplementedError


def short(model: str) -> str:
    return model.split("/")[-1]


class LLMGenerate(Strategy):
    """Generator models write whole phrases from the archetypes, seeing our scored lines (OPRO style).
    One instance and bandit arm per price tier; each batch goes to the next model in the tier, round robin."""

    needs_llm = True
    COLD_AFTER = 2  # consecutive empty batches before a model sits out
    COLD_SECONDS = 300

    def __init__(self, models: list[str] | None = None, tier: int = 0, count: int = 40):
        from ..config import GEN_MODEL

        self.models = list(models or [GEN_MODEL])
        self.tier = tier
        self.name = "llm:" + TIER_NAMES.get(tier, str(tier))
        self.count = count
        self.round = 0
        self.turn = 0
        self.pending: asyncio.Future | None = None
        self.empty_streak = {m: 0 for m in self.models}
        self.cold_until = {m: 0.0 for m in self.models}

    def unlocked(self, ctx) -> bool:
        """Pricier tiers join only once the ladder has escalated far enough."""
        return ctx.engine.level >= TIER_LEVEL.get(self.tier, 0)

    def warm(self, ctx=None) -> list[str]:
        """Models off cooldown. In the long-chain regime the expensive ones only join on a stall, within the
        per-question allowance."""
        now = time.monotonic()
        warm = [m for m in self.models if now >= self.cold_until[m]]
        if ctx is not None and ctx.engine.frugal() and not ctx.engine.expensive_ok():
            warm = [m for m in warm if m not in EXPENSIVE_MODELS]
        return warm

    def next_model(self, ctx=None) -> str | None:
        warm = self.warm(ctx)
        if not warm:
            return None
        model = warm[0]
        if ctx is not None and ctx.engine.frugal():
            pricey = [m for m in warm if m in EXPENSIVE_MODELS]
            if pricey:
                ctx.engine.expensive_used += 1
                ctx.log(f"[{self.name}] stall: one {short(pricey[0])} batch "
                        f"({ctx.engine.expensive_used} used this question)")
                return pricey[0]
        for _ in range(len(self.models)):
            candidate = self.models[self.turn % len(self.models)]
            self.turn += 1
            if candidate in warm:
                return candidate
        return model

    def available(self, ctx) -> bool:
        if ctx.llm is None or not self.warm(ctx) or not self.unlocked(ctx):
            return False
        return self.pending is None or self.pending.done() or len(ctx.archive) < 20

    def cancel(self) -> None:
        if self.pending is not None and not self.pending.done():
            self.pending.cancel()

    def pick_archetypes(self, ctx, fresh: bool = False) -> list[str]:
        names = [a for a in ARCHETYPES if a in prompts.ARCHETYPE_GUIDE]
        if ctx.focus:
            focus = [a for a in ctx.focus if a in names]
            others = [a for a in names if a not in focus]
            return focus + ctx.rng.sample(others, min(1, len(others)))
        if fresh:
            tries = {a: s["tries"] for a, s in ctx.archive.style_stats().items()}
            weights = [1.0 / (1.0 + tries.get(a, 0) / 50.0) for a in names]
        else:
            best = ctx.archetype_bests()
            weights = [1.0 + 3.0 * max(best.get(a, 0.0), 0.0) for a in names]
        chosen: list[str] = []
        while len(chosen) < 3:
            pick = ctx.rng.choices(names, weights)[0]
            if pick not in chosen:
                chosen.append(pick)
        return chosen

    async def _call(self, ctx, model: str, archetypes: list[str], variants: list[str],
                    fresh: bool = False) -> list[tuple[str, str]]:
        max_tokens = 6000
        if ctx.objective.long:
            target = ctx.engine.long_target
            user = prompts.long_chain_user(ctx, archetypes, min(self.count, max(4, 3600 // max(target, 1))), target)
            max_tokens = 9000
        else:
            user = prompts.gen_user(ctx, archetypes, self.count, variants, fresh=fresh)
        try:
            data = await ctx.llm.json(model, prompts.gen_system(ctx), user, temperature=1.0, max_tokens=max_tokens)
        except (LLMError, Exception) as error:  # one bad model reply should not stop the lab
            ctx.log(f"[{self.name} {short(model)}] {error}")
            return []
        rows = data.get("phrases") if isinstance(data, dict) else data
        out = []
        for row in rows or []:
            if isinstance(row, dict):
                text = str(row.get("text") or row.get("phrase") or "")
                tactic = str(row.get("tactic") or "other")
            else:
                text, tactic = str(row), "other"
            tactic = tactic.strip("[] ").lower()
            out.append((text, tactic if tactic in ARCHETYPES else "other"))
        return out

    async def _batch(self, ctx) -> tuple[str, list[str], list[tuple[str, str]], bool]:
        self.round += 1
        model = self.next_model(ctx)
        fresh = ctx.fresh and self.round % 2 == 1
        archetypes = self.pick_archetypes(ctx, fresh)
        if model is None:
            return "", archetypes, [], fresh
        variants = []
        if not fresh and self.round % 2 == 0 and len(ctx.archive):
            variants = [c.phrase for c in ctx.parents(3)]
        proposals = await self._call(ctx, model, archetypes, variants, fresh)
        return model, archetypes, proposals, fresh

    def prefetch(self, ctx) -> None:
        """Start writing the next batch now so oracle strategies never wait on the LLM."""
        if self.pending is None:
            self.pending = asyncio.ensure_future(self._batch(ctx))

    async def run(self, ctx) -> None:
        self.prefetch(ctx)
        task, self.pending = self.pending, None
        model, archetypes, proposals, fresh = await task
        if not model:
            return
        if proposals:
            self.empty_streak[model] = 0
        else:
            self.empty_streak[model] += 1
            if self.empty_streak[model] >= self.COLD_AFTER:
                self.cold_until[model] = time.monotonic() + self.COLD_SECONDS
                self.empty_streak[model] = 0
                ctx.log(f"[{self.name}] {short(model)}: {self.COLD_AFTER} empty batches in a row; sitting out "
                        f"{self.COLD_SECONDS // 60} minutes")
        if self.warm(ctx):
            self.prefetch(ctx)
        if not proposals:
            return
        kind = "fresh-frame" if fresh else ""
        ctx.log(f"[{self.name} {short(model)}] {len(proposals)} {kind + ' ' if kind else ''}phrases for "
                f"{', '.join(archetypes)}")
        await ctx.evaluate([(text, self.name, "", tactic) for text, tactic in proposals])


class LocalEdit(Strategy):
    """The single-edit neighbourhood of a top phrase: substitute, delete, insert, swap, move clauses."""

    name = "local_edit"

    def __init__(self, cap: int = 500):
        self.cap = cap
        self.expanded: dict[str, int] = {}

    def pick_parent(self, ctx):
        top = ctx.parents(8) if ctx.fresh else ctx.archive.top(8, min_n=1)
        weights = [1.0 / (1 + i) / (1 + self.expanded.get(c.phrase, 0)) ** 2 for i, c in enumerate(top)]
        return ctx.rng.choices(top, weights)[0]

    async def run(self, ctx) -> None:
        parent = self.pick_parent(ctx)
        self.expanded[parent.phrase] = self.expanded.get(parent.phrase, 0) + 1
        words = parent.words
        alts = await ctx.alternatives(words)
        pool = ctx.word_pool(80)
        must: list[str] = []
        optional: list[str] = []

        for i in range(len(words)):
            must.append(" ".join(words[:i] + words[i + 1 :]))
        for i in range(len(words) - 1):
            swapped = words[:]
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            must.append(" ".join(swapped))
        for i, word in enumerate(words):
            rest = words[:i] + words[i + 1 :]
            must.append(" ".join(rest + [word]))
            must.append(" ".join([word] + rest))
        breaks = [i for i, w in enumerate(words) if w.casefold() in CLAUSE_BREAKS and 0 < i < len(words) - 1]
        for i in breaks:
            must.append(" ".join(words[i + 1 :] + [words[i]] + words[:i]))
            must.append(" ".join(words[i:] + words[:i]))

        for i, word in enumerate(words):
            choices = list(dict.fromkeys(alts.get(word, []) + ctx.rng.sample(pool, min(12, len(pool)))))
            for alt in choices:
                if alt != word:
                    optional.append(" ".join(words[:i] + [alt] + words[i + 1 :]))
        for slot in range(len(words) + 1):
            for word in ctx.rng.sample(pool + FUNCTION_WORDS, min(10, len(pool) + len(FUNCTION_WORDS))):
                optional.append(" ".join(words[:slot] + [word] + words[slot:]))

        optional = [p for p in dict.fromkeys(optional) if p not in ctx.archive]
        budget = max(self.cap - len(must), 50)
        optional = await ctx.screen(optional, budget)
        ctx.log(f"[local_edit] {len(must)} structural + {len(optional)} word edits of {parent.p:.3f} {parent.units}w")
        await ctx.evaluate([(p, self.name, parent.phrase, parent.archetype) for p in must + optional])


class Compress(Strategy):
    """Ablate the best line word by word, then beam-delete while the rounded score holds."""

    name = "compress"

    def __init__(self, beam: int = 3, depth: int = 10, share: float = 0.2):
        self.beam = beam
        self.depth = depth
        self.share = share  # of the remaining budget one turn may spend; a full beam on 48 words is ~5k calls
        self.done: set[str] = set()
        self.stop_at = 0

    @staticmethod
    def worth_cutting(ctx, cand) -> bool:
        """Cutting only pays when we are level with the leader on score or longer than them.
        Shortest yes: any line longer than one word that has landed a roll over the threshold."""
        lead = ctx.leader
        if cand.units <= 1:
            return False
        if ctx.objective.shortest:
            return ctx.objective.reachable(cand.score)
        if ctx.objective.long:
            # Longer lines belong to chain compaction; below the ceiling only a line level with the leader pays.
            return cand.units <= WORD_STAGE and (ctx.objective.at_ceiling(cand.p) or lead is None
                                                 or site_round(cand.p) >= site_round(lead.probability))
        return lead is None or site_round(cand.p) >= site_round(lead.probability) or cand.units >= lead.units

    def available(self, ctx) -> bool:
        return any(c.phrase not in self.done and self.worth_cutting(ctx, c) for c in ctx.archive.top_by_board(8))

    def left(self, ctx) -> int:
        return self.stop_at - ctx.engine.oracle.calls

    async def run(self, ctx, roots: int = 3) -> None:
        self.stop_at = ctx.engine.oracle.calls + max(150, int(ctx.engine.remaining() * self.share))
        for _ in range(roots):
            candidates = [c for c in ctx.archive.top_by_board(8)
                          if c.phrase not in self.done and self.worth_cutting(ctx, c)]
            if not candidates or self.left(ctx) < 20:
                return
            if await self.compress(ctx, candidates[0]):
                return

    async def compress(self, ctx, root) -> bool:
        """Returns True if a shorter line kept the rounded score (Shortest yes: still lands rolls over the
        threshold, which is worth a gamble since the board keeps our best roll)."""
        self.done.add(root.phrase)
        await ctx.evaluate([(root.phrase, "resample", root.parent, root.archetype)], n=4)
        root = ctx.archive.items[root.phrase]
        await self.ablate(ctx, root)
        objective = ctx.objective
        target = site_round(root.p)
        level = objective.ceiling if objective.holds_ceiling(root.score, 1) else target

        def holds(cand) -> bool:
            if objective.shortest:
                return objective.reachable(cand.score)
            if objective.long:
                return objective.holds(cand.score, level, 2)
            return site_round(cand.p) >= target - 0.01 + 1e-9

        beam = [root]
        for depth in range(self.depth):
            children: list[tuple[str, str, str]] = []
            for node in beam:
                w = node.words
                for i in range(len(w)):
                    children.append((" ".join(w[:i] + w[i + 1 :]), node.phrase, node.archetype))
                for i in range(len(w) - 1):
                    children.append((" ".join(w[:i] + w[i + 2 :]), node.phrase, node.archetype))
            children = [c for c in children if c[0]]
            left = self.left(ctx)
            if left < 20:
                ctx.log(f"[compress] turn budget used at depth {depth + 1}; best so far {beam[0].units}w")
                break
            if len(children) * 2 > left:
                origin = {p: (parent, arch) for p, parent, arch in children}
                children = [(p, *origin[p]) for p in await ctx.screen(list(origin), left // 2) if p in origin]
            scored = await ctx.evaluate([(p, self.name, parent, arch) for p, parent, arch in children], n=2)
            keep = [c for c in scored if holds(c)]
            if not keep:
                if depth == 0:
                    floor = objective.threshold if objective.shortest else level if objective.long else target - 0.01
                    ctx.log(f"[compress] {root.units}w {root.p:.3f} is tight: every deletion drops below {floor:.2f}")
                break
            keep.sort(key=lambda c: (objective.score_key(c.score, c.units), c.p), reverse=True)
            beam = keep[: self.beam]
            ctx.log(f"[compress] depth {depth + 1}: {beam[0].p:.3f} {beam[0].units}w {beam[0].phrase}")
        if beam and beam[0].phrase != root.phrase:
            await ctx.evaluate([(beam[0].phrase, "resample", beam[0].parent, beam[0].archetype)], n=5)
            best = ctx.archive.items[beam[0].phrase]
            if objective.shortest:
                return objective.reachable(best.score) and best.units < root.units
            if objective.long:
                return objective.holds(best.score, level, 5) and best.units < root.units
            return site_round(best.p) >= target and best.units < root.units
        return False

    async def ablate(self, ctx, cand) -> None:
        w = cand.words
        if len(w) < 2:
            return
        deletions = [" ".join(w[:i] + w[i + 1 :]) for i in range(len(w))]
        scored = await ctx.evaluate([(p, "ablate", cand.phrase, cand.archetype) for p in deletions], n=2)
        by_phrase = {c.phrase: c for c in scored}
        ctx.archive.ablation[cand.phrase] = [
            (cand.p - by_phrase[p].p) if p in by_phrase else 0.0 for p in deletions
        ]


def mutate(words: list[str], pool: list[str], rng: random.Random, cap: int = 0) -> list[str]:
    """One random edit. Well under the length cap, inserts outnumber deletes so lines can grow."""
    w = list(words)
    growing = cap and len(w) < cap * 0.6
    delete, replace, insert = (0.2, 0.45, 0.85) if growing else (0.3, 0.6, 0.8)
    if cap and len(w) >= cap:
        insert = replace
    op = rng.random()
    if op < delete and len(w) > 2:
        del w[rng.randrange(len(w))]
    elif op < replace and pool:
        w[rng.randrange(len(w))] = rng.choice(pool)
    elif op < insert and pool:
        w.insert(rng.randrange(len(w) + 1), rng.choice(pool))
    elif len(w) > 1:
        i = rng.randrange(len(w) - 1)
        w[i], w[i + 1] = w[i + 1], w[i]
    return w
