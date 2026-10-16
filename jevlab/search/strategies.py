"""Search strategies. Each `run(ctx)` proposes phrases and scores them through `ctx.evaluate`."""

from __future__ import annotations

import asyncio
import random
import time

from ..config import EXPENSIVE_MODELS, TIER_LEVEL, TIER_NAMES
from .. import llm as llm_mod
from ..llm import LLMError
from .hardboard import HARD_LENGTH, HARD_TIMEOUT_WORDS
from ..objective import site_round
from ..rules import normalize
from . import prompts
from . import probe
from .archive import ARCHETYPES
from .routes import assign

FUNCTION_WORDS = [
    "the",
    "a",
    "our",
    "my",
    "this",
    "is",
    "are",
    "so",
    "and",
    "which",
    "that",
    "here",
    "means",
    "always",
    "only",
    "every",
    "no",
    "not",
    "yes",
    "we",
    "i",
    "it",
    "of",
    "in",
    "at",
    "with",
    "for",
    "one",
    "all",
]
CLAUSE_BREAKS = {"and", "so", "which", "that", "because", "but", "then", "hereafter", "therefore", "where"}
# Long-chain regime: at or below this many words, word-level strategies take over from chain compaction.
WORD_STAGE = 60
SYLLOGISM = "syllogism"
# Opening core-tier batches reserve one tactic slot for a deduction, until a few of those lines are scored.
SYLLOGISM_EARLY_ROUNDS = 3
SYLLOGISM_EARLY_TRIES = 12
# Once a batch is entirely past this many scored lines, one slot moves to a tactic that is not.
COVERAGE_TRIES = 8


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

    One instance and bandit arm per price tier. Before the first batch, and again when the search
    goes flat, every unlocked model writes the same five tactics. The winner then writes every tactic
    until the next probe. A stall leaves that winner when another model scored. Within a tier that
    has not been probed, models still take turns.
    """

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
        self.just_probed = False

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
        if ctx is not None:
            locked = ctx.engine.probe_lock.model
            if locked:
                if locked not in self.models or time.monotonic() < self.cold_until.get(locked, 0):
                    return None
                return locked
        warm = self.warm(ctx)
        if not warm:
            return None
        model = warm[0]
        if ctx is not None and ctx.engine.frugal():
            pricey = [m for m in warm if m in EXPENSIVE_MODELS]
            if pricey:
                ctx.engine.expensive_used += 1
                ctx.log(
                    f"[{self.name}] stall: one {short(pricey[0])} batch "
                    f"({ctx.engine.expensive_used} used this question)"
                )
                return pricey[0]
        for _ in range(len(self.models)):
            candidate = self.models[self.turn % len(self.models)]
            self.turn += 1
            if candidate in warm:
                return candidate
        return model

    def _opening_arm(self, ctx) -> bool:
        """Until a model is locked, only the cheapest unlocked tier runs the probe."""
        arms = [
            arm
            for arm in ctx.engine.strategies.values()
            if isinstance(arm, LLMGenerate) and arm.unlocked(ctx) and arm.warm(ctx)
        ]
        if not arms:
            return True
        lead = min(arms, key=lambda arm: (arm.tier, arm.name))
        return lead is self

    def available(self, ctx) -> bool:
        if ctx.llm is None or not self.unlocked(ctx):
            return False
        locked = ctx.engine.probe_lock.model
        if locked:
            ready = locked in self.models and time.monotonic() >= self.cold_until.get(locked, 0)
        else:
            ready = bool(self.warm(ctx)) and self._opening_arm(ctx)
        if not ready:
            return False
        return self.pending is None or self.pending.done() or len(ctx.archive) < 20

    def cancel(self) -> None:
        if self.pending is not None and not self.pending.done():
            self.pending.cancel()

    def reserve_syllogism(self, ctx) -> bool:
        """Highest is still looking for a high score: try a deduction before the mix settles."""
        if self.tier != 0 or ctx.objective.shortest or ctx.engine.one_word_race:
            return False
        if self.round > SYLLOGISM_EARLY_ROUNDS:
            return False
        tries = ctx.archive.style_stats().get(SYLLOGISM, {}).get("tries", 0)
        return tries < SYLLOGISM_EARLY_TRIES

    def pick_archetypes(self, ctx, fresh: bool = False) -> list[str]:
        names = [a for a in ARCHETYPES if a in prompts.ARCHETYPE_GUIDE]
        if ctx.focus:
            focus = [a for a in ctx.focus if a in names]
            others = [a for a in names if a not in focus]
            chosen = focus + ctx.rng.sample(others, min(1, len(others)))
        else:
            if fresh:
                tries = {a: s["tries"] for a, s in ctx.archive.style_stats().items()}
                weights = [1.0 / (1.0 + tries.get(a, 0) / 50.0) for a in names]
            else:
                best = ctx.archetype_bests()
                weights = [1.0 + 3.0 * max(best.get(a, 0.0), 0.0) for a in names]
            chosen = []
            while len(chosen) < 3:
                pick = ctx.rng.choices(names, weights)[0]
                if pick not in chosen:
                    chosen.append(pick)
        protected: list[str] = []
        if SYLLOGISM in names and SYLLOGISM not in chosen and self.reserve_syllogism(ctx):
            rest = [name for name in chosen if name != SYLLOGISM]
            chosen = [SYLLOGISM, *rest[:2]]
            protected = [SYLLOGISM]
        return self.cover_untried(ctx, chosen, names, protected)

    def cover_untried(self, ctx, chosen: list[str], names: list[str], protected: list[str]) -> list[str]:
        """Keep one slot on a tactic the archive has barely scored, without evicting a reserved one."""
        stats = ctx.archive.style_stats()

        def tries(name: str) -> int:
            return int(stats.get(name, {}).get("tries", 0))

        if any(tries(name) < COVERAGE_TRIES for name in chosen):
            return chosen
        needy = [name for name in names if name not in chosen and tries(name) < COVERAGE_TRIES]
        if not needy:
            return chosen
        floor = min(tries(name) for name in needy)
        pick = ctx.rng.choice([name for name in needy if tries(name) == floor])
        if len(chosen) < 3:
            return [*chosen, pick]
        held = set(protected)
        droppable = [i for i, name in enumerate(chosen) if name not in held]
        if not droppable:
            return chosen
        drop = max(droppable, key=lambda i: (tries(chosen[i]), i))
        return [name for i, name in enumerate(chosen) if i != drop] + [pick]

    async def _call(
        self, ctx, model: str, archetypes: list[str], variants: list[str], fresh: bool = False, count: int | None = None
    ) -> list[tuple[str, str]]:
        n = self.count if count is None else count
        max_tokens = 6000
        reply_timeout = None
        if ctx.objective.long:
            target = ctx.engine.long_target
            user = prompts.long_chain_user(ctx, archetypes, min(n, max(4, 3600 // max(target, 1))), target)
            max_tokens = 9000
        else:
            user = prompts.gen_user(ctx, archetypes, n, variants, fresh=fresh)
            schedule = getattr(ctx.engine, "schedule", None)
            words = schedule.deal_length() if schedule is not None and schedule.hard and schedule.phase == "deal" else 0
            if words > 0:
                # One ordinary timeout covers about 20 words. z.ai was still going at 60s on an 80-word ask.
                if words > HARD_LENGTH:
                    max_tokens = round(6000 * words / HARD_LENGTH)
                reply_timeout = llm_mod.LLM_TIMEOUT * words / HARD_TIMEOUT_WORDS
        try:
            extra = {} if reply_timeout is None else {"timeout": reply_timeout}
            data = await ctx.llm.json(
                model, prompts.gen_system(ctx), user, temperature=1.0, max_tokens=max_tokens, **extra
            )
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

    def route(self, ctx, model: str | None, archetypes: list[str]) -> tuple[str | None, list[str]]:
        """Keep a dear model on tactics it wins. Otherwise hand the batch to the next model."""
        if model is None:
            return None, archetypes
        routed = assign(model, archetypes)
        if routed is not None:
            return model, routed
        skipped = short(model)
        for _ in range(max(len(self.models) - 1, 0)):
            model = self.next_model(ctx)
            if model is None:
                break
            routed = assign(model, archetypes)
            if routed is not None:
                ctx.log(f"[{self.name}] {skipped} skipped; {short(model)} writes this batch")
                return model, routed
        ctx.log(f"[{self.name}] {skipped} skipped; no other model is warm")
        return None, archetypes

    def _arms(self, ctx) -> list[LLMGenerate]:
        arms = [arm for arm in ctx.engine.strategies.values() if isinstance(arm, LLMGenerate) and arm.unlocked(ctx)]
        if self.unlocked(ctx) and self not in arms:
            arms.append(self)
        return sorted(arms, key=lambda arm: (arm.tier, arm.name))

    def probe_pool(self, ctx) -> list[str]:
        """Unlocked models, cheapest tier first. A cold model stays in the pool so the probe can drop it."""
        found: list[str] = []
        pricey = ctx.engine.frugal() and not ctx.engine.expensive_ok()
        for arm in self._arms(ctx):
            for model in arm.models:
                if pricey and model in EXPENSIVE_MODELS:
                    continue
                if model not in found:
                    found.append(model)
        return found

    def _writer_cold(self, ctx) -> bool:
        model = ctx.engine.probe_lock.model
        if not model:
            return False
        for arm in self._arms(ctx):
            if model in arm.cold_until and time.monotonic() < arm.cold_until[model]:
                return True
        return False

    def _probe_due(self, ctx) -> bool:
        stats = ctx.archive.style_stats()
        tries = {name: int(row.get("tries", 0)) for name, row in stats.items()}
        tactics = [name for name in ARCHETYPES if name in prompts.ARCHETYPE_GUIDE]
        return probe.due(
            ctx.engine.probe_lock,
            stall=ctx.engine.stall,
            exhausted=probe.searched_out(tries, tactics, COVERAGE_TRIES),
            cold=self._writer_cold(ctx),
            force=ctx.engine.probe_force,
        )

    def _release(self, ctx, model: str) -> None:
        """A model the probe just picked gets to write, even if empty batches had sat it out."""
        for arm in self._arms(ctx):
            if model in arm.cold_until:
                arm.cold_until[model] = 0.0
                arm.empty_streak[model] = 0

    async def _probe(self, ctx) -> str | None:
        async with ctx.engine.probe_gate:
            if not self._probe_due(ctx):
                return ctx.engine.probe_lock.model
            pool = self.probe_pool(ctx)
            if not pool:
                return None
            ctx.log(f"[{self.name}] probing {', '.join(short(model) for model in pool)}")
            tactics = list(probe.PROBE_TACTICS)
            batches = await asyncio.gather(
                *(self._call(ctx, model, tactics, [], count=probe.PROBE_COUNT) for model in pool)
            )
            owners: dict[str, str] = {}
            items = []
            for model, proposals in zip(pool, batches, strict=True):
                kept = proposals
                if ctx.engine.one_word_race:
                    kept = [(text, tactic) for text, tactic in proposals if len(text.split()) == 1]
                for text, tactic in kept:
                    key = normalize(text)
                    if key in owners:
                        continue
                    owners[key] = model
                    items.append((text, self.name, "", tactic))
            scored = await ctx.evaluate(items) if items else []
            by_model: dict[str, list[float]] = {model: [] for model in pool}
            for cand in scored:
                owner = owners.get(cand.phrase) or owners.get(normalize(cand.phrase))
                if owner in by_model:
                    by_model[owner].append(cand.p)
            scores = {model: probe.mean_top(values) for model, values in by_model.items()}
            # Cold and a newly unlocked tier may keep the writer. A stall may not, once someone else scored.
            held = ctx.engine.probe_lock.model
            avoid = held if held and not ctx.engine.probe_force and not self._writer_cold(ctx) else None
            winner = probe.choose(scores, pool, avoid=avoid) or probe.choose(scores, pool) or pool[0]
            ctx.engine.probe_lock = probe.Lock(model=winner, batches=0)
            ctx.engine.probe_force = False
            self.just_probed = True
            self._release(ctx, winner)
            ranked = " ".join(f"{short(model)}={scores[model]:.3f}" for model in pool)
            if avoid and winner != avoid:
                ctx.log(f"[{self.name}] probe leaves {short(avoid)} after a stall; locks {short(winner)} ({ranked})")
            else:
                ctx.log(f"[{self.name}] probe locks {short(winner)} ({ranked})")
            return winner

    async def _batch(self, ctx) -> tuple[str, list[str], list[tuple[str, str]], bool]:
        self.round += 1
        if self._probe_due(ctx):
            winner = await self._probe(ctx)
            if self.just_probed:
                return winner or "", list(probe.PROBE_TACTICS), [], False
            if not winner or winner not in self.models:
                return "", [], [], False
        model = self.next_model(ctx)
        fresh = ctx.fresh and self.round % 2 == 1
        archetypes = self.pick_archetypes(ctx, fresh)
        model, archetypes = self.route(ctx, model, archetypes)
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

    def _prefetch_next(self, ctx) -> None:
        """A batch that is about to re-probe waits until this one is finished, so a flat run is not asked twice."""
        if self.warm(ctx) and not self._probe_due(ctx):
            self.prefetch(ctx)

    async def run(self, ctx) -> None:
        self.prefetch(ctx)
        task, self.pending = self.pending, None
        model, archetypes, proposals, fresh = await task
        probed = self.just_probed
        self.just_probed = False
        if ctx.engine.one_word_race:
            proposals = [(text, tactic) for text, tactic in proposals if len(text.split()) == 1]
        if probed:
            if model:
                self.empty_streak[model] = 0
            self._prefetch_next(ctx)
            return
        if not model:
            return
        ctx.engine.probe_lock.batches += 1
        if proposals:
            self.empty_streak[model] = 0
        else:
            self.empty_streak[model] += 1
            if self.empty_streak[model] >= self.COLD_AFTER:
                self.cold_until[model] = time.monotonic() + self.COLD_SECONDS
                self.empty_streak[model] = 0
                ctx.log(
                    f"[{self.name}] {short(model)}: {self.COLD_AFTER} empty batches in a row; sitting out "
                    f"{self.COLD_SECONDS // 60} minutes"
                )
        self._prefetch_next(ctx)
        if not proposals:
            return
        kind = "fresh-frame" if fresh else ""
        ctx.log(
            f"[{self.name} {short(model)}] {len(proposals)} {kind + ' ' if kind else ''}phrases for "
            f"{', '.join(archetypes)}"
        )
        await ctx.evaluate([(text, self.name, "", tactic) for text, tactic in proposals])


class LocalEdit(Strategy):
    """The single-edit neighbourhood of a top phrase: substitute, delete, insert, swap, move clauses."""

    name = "local_edit"

    def __init__(self, cap: int = 500):
        self.cap = cap
        self.expanded: dict[str, int] = {}

    def pick_parent(self, ctx):
        if ctx.engine.one_word_race:
            top = ctx.single_word_lines(8)
        else:
            top = []
        if not top:
            top = ctx.parents(8) if ctx.fresh else ctx.archive.top(8, min_n=1)
        top = ctx.restrict(top)
        if not top:
            return None
        weights = [1.0 / (1 + i) / (1 + self.expanded.get(c.phrase, 0)) ** 2 for i, c in enumerate(top)]
        return ctx.rng.choices(top, weights)[0]

    async def run(self, ctx) -> None:
        parent = self.pick_parent(ctx)
        if parent is None:
            return
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
        if not ctx.engine.one_word_race:
            for slot in range(len(words) + 1):
                for word in ctx.rng.sample(pool + FUNCTION_WORDS, min(10, len(pool) + len(FUNCTION_WORDS))):
                    optional.append(" ".join(words[:slot] + [word] + words[slot:]))

        optional = [p for p in dict.fromkeys(optional) if p not in ctx.archive]
        budget = max(self.cap - len(must), 50)
        optional = await ctx.screen(optional, budget)
        ctx.log(f"[local_edit] {len(must)} structural + {len(optional)} word edits of {parent.p:.3f} {parent.units}w")
        await ctx.evaluate([(p, self.name, parent.phrase, parent.archetype) for p in must + optional])


def clause_span_deletions(words: list[str]) -> list[str]:
    """Phrases left after deleting one clause span.

    An interior marker starts a span, so the marker is deleted with the clause that follows it.
    The prefix before the first marker is its own span. No interior marker means no cut: dropping
    the whole line is not a candidate. A span whose removal leaves nothing is skipped.
    """
    breaks = [i for i, word in enumerate(words) if word.casefold() in CLAUSE_BREAKS and 0 < i < len(words) - 1]
    if not breaks:
        return []
    bounds = [0, *breaks, len(words)]
    kept = []
    for start, end in zip(bounds, bounds[1:]):
        rest = words[:start] + words[end:]
        if rest:
            kept.append(" ".join(rest))
    return kept


class Compress(Strategy):
    """Ablate the best line word by word, then beam-delete while the rounded score holds.

    Each step also tries deleting one clause span. Those cuts are scored before the one- and
    two-word pile and never pass through the budget screen.
    """

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
            if ctx.engine.one_word_race:
                return True
            return ctx.objective.reachable(cand.score)
        if ctx.objective.long:
            # Longer lines belong to chain compaction; below the ceiling only a line level with the leader pays.
            return cand.units <= WORD_STAGE and (
                ctx.objective.at_ceiling(cand.p) or lead is None or site_round(cand.p) >= site_round(lead.probability)
            )
        if ctx.objective.kind == "noul":
            # Highest: length alone does not open compression. Hard visits also wait until after growth.
            return ctx.engine.highest_may_compress(cand)
        return lead is None or site_round(cand.p) >= site_round(lead.probability) or cand.units >= lead.units

    def available(self, ctx) -> bool:
        return any(c.phrase not in self.done and self.worth_cutting(ctx, c) for c in ctx.archive.top_by_board(8))

    def left(self, ctx) -> int:
        return self.stop_at - ctx.engine.oracle.calls

    async def run(self, ctx, roots: int = 3) -> None:
        self.stop_at = ctx.engine.oracle.calls + max(150, int(ctx.engine.remaining() * self.share))
        for _ in range(roots):
            candidates = [
                c for c in ctx.archive.top_by_board(8) if c.phrase not in self.done and self.worth_cutting(ctx, c)
            ]
            if not candidates or self.left(ctx) < 20:
                return
            if await self.compress(ctx, candidates[0]):
                return

    async def compress(self, ctx, root) -> bool:
        """Returns True if a shorter line kept the rounded score (Shortest yes: still lands rolls over the
        threshold, which is worth a gamble since the board keeps our best roll)."""
        if ctx.engine.one_word_race:
            self.done.add(root.phrase)
            fresh = [w for w in dict.fromkeys(root.words) if w not in ctx.archive]
            if fresh:
                await ctx.evaluate([(w, self.name, root.phrase, root.archetype) for w in fresh])
            return False
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
            spans: list[tuple[str, str, str]] = []
            for node in beam:
                w = node.words
                short: list[tuple[str, str, str]] = []
                for i in range(len(w)):
                    short.append((" ".join(w[:i] + w[i + 1 :]), node.phrase, node.archetype))
                for i in range(len(w) - 1):
                    short.append((" ".join(w[:i] + w[i + 2 :]), node.phrase, node.archetype))
                children.extend(short)
                already = {row[0] for row in short}
                for phrase in clause_span_deletions(w):
                    if phrase in already:
                        continue
                    spans.append((phrase, node.phrase, node.archetype))
                    already.add(phrase)
            children = [c for c in children if c[0]]
            left = self.left(ctx)
            if left < 20:
                ctx.log(f"[compress] turn budget used at depth {depth + 1}; best so far {beam[0].units}w")
                break
            # Span cuts are few and must not be subsampled. Score them first; the word pile
            # gets whatever calls remain. If the spans themselves do not fit, take as many as
            # n=2 allows and skip the pile.
            room = left // 2
            spans_now = spans[:room]
            scored: list = []
            if spans_now:
                scored += await ctx.evaluate([(p, self.name, parent, arch) for p, parent, arch in spans_now], n=2)
            if len(spans_now) == len(spans):
                left = self.left(ctx)
                if len(children) * 2 > left:
                    origin = {p: (parent, arch) for p, parent, arch in children}
                    picked = await ctx.screen(list(origin), left // 2)
                    children = [(p, *origin[p]) for p in picked if p in origin]
                scored += await ctx.evaluate([(p, self.name, parent, arch) for p, parent, arch in children], n=2)
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
        ctx.archive.ablation[cand.phrase] = [(cand.p - by_phrase[p].p) if p in by_phrase else 0.0 for p in deletions]


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
