"""Genetic algorithm, surrogate-guided Bayesian optimisation, and GCG-style swaps."""

from __future__ import annotations

from ..llm import LLMError
from . import prompts
from .strategies import CLAUSE_BREAKS, Strategy, mutate


def crossover(a: list[str], b: list[str], rng) -> list[str]:
    """Splice a prefix of one parent onto a suffix of the other, preferring clause boundaries."""
    def cuts(words):
        preferred = [i for i, w in enumerate(words) if w.casefold() in CLAUSE_BREAKS and 0 < i < len(words)]
        return preferred or list(range(1, max(len(words), 2)))

    i = rng.choice(cuts(a))
    j = rng.choice(cuts(b))
    child = a[:i] + b[j:]
    return child if child else a


class Genetic(Strategy):
    """MAP-Elites-seeded GA: tournament selection, crossover, mutation, occasional LLM rewrites."""

    name = "genetic"

    def __init__(self, offspring: int = 140, send: int = 90):
        self.offspring = offspring
        self.send = send
        self.generation = 0

    def available(self, ctx) -> bool:
        return len(ctx.archive) >= 6

    def population(self, ctx):
        pop = {c.phrase: c for c in ctx.archive.elite_list()}
        for c in ctx.parents(30):
            pop.setdefault(c.phrase, c)
        return list(pop.values())

    def tournament(self, pop, rng, size: int = 3):
        return max(rng.sample(pop, min(size, len(pop))), key=lambda c: c.fitness)

    async def llm_children(self, ctx, parents) -> list[tuple[str, str, str]]:
        if ctx.llm is None or not parents:
            return []
        try:
            data = await ctx.llm.json(ctx.writer_model(), prompts.gen_system(ctx),
                                      prompts.rewrite_user(ctx, [p.phrase for p in parents], 4), temperature=1.0)
        except (LLMError, Exception) as error:
            ctx.log(f"[genetic] llm rewrite failed: {error}")
            return []
        out = []
        for row in (data.get("children") if isinstance(data, dict) else data) or []:
            if not isinstance(row, dict):
                continue
            try:
                parent = parents[int(row.get("parent") or 0)]
            except (ValueError, IndexError):
                parent = parents[0]
            out.append((str(row.get("text") or ""), parent.phrase, parent.archetype))
        return out

    async def run(self, ctx) -> None:
        self.generation += 1
        rng = ctx.rng
        pop = self.population(ctx)
        pool = ctx.word_pool(60)
        children: dict[str, tuple[str, str]] = {}
        for _ in range(self.offspring * 3):
            if len(children) >= self.offspring:
                break
            a = self.tournament(pop, rng)
            if rng.random() < 0.6:
                b = self.tournament(pop, rng)
                words = crossover(a.words, b.words, rng)
                if rng.random() < 0.5:
                    words = mutate(words, pool, rng, ctx.engine.grow_cap)
            else:
                words = a.words
                for _ in range(rng.choice((1, 1, 2, 3))):
                    words = mutate(words, pool, rng, ctx.engine.grow_cap)
            phrase = " ".join(words)
            if phrase and phrase not in ctx.archive:
                children[phrase] = (a.phrase, a.archetype)
        ranked = await ctx.screen(list(children), self.send)
        batch = [(p, self.name, children[p][0], children[p][1]) for p in ranked]
        if self.generation % 3 == 0:
            batch += [(t, "llm_rewrite", par, arch) for t, par, arch in await self.llm_children(ctx, ctx.parents(5))]
        ctx.log(f"[genetic] gen {self.generation}: pop {len(pop)}, {len(children)} offspring, sending {len(batch)}")
        await ctx.evaluate(batch)


class SurrogateBO(Strategy):
    """Generate thousands of edits locally, send the top UCB picks from the phrase surrogate."""

    name = "surrogate_bo"

    def __init__(self, pool_size: int = 1500, send: int = 120, kappa: float = 1.5):
        self.pool_size = pool_size
        self.send = send
        self.kappa = kappa

    def available(self, ctx) -> bool:
        return ctx.phrase_surrogate is not None and ctx.phrase_surrogate.ready and len(ctx.archive) >= 10

    async def run(self, ctx) -> None:
        rng = ctx.rng
        parents = ctx.parents(20) + ctx.archive.elite_list()
        pool_words = ctx.word_pool(120)
        cands: dict[str, tuple[str, str]] = {}
        for _ in range(self.pool_size * 2):
            if len(cands) >= self.pool_size:
                break
            parent = rng.choice(parents)
            words = parent.words
            for _ in range(rng.choice((1, 2, 2, 3, 4))):
                words = mutate(words, pool_words, rng, ctx.engine.grow_cap)
            if rng.random() < 0.2:
                other = rng.choice(parents)
                words = crossover(words, other.words, rng)
            phrase = " ".join(words)
            if phrase and phrase not in ctx.archive:
                cands[phrase] = (parent.phrase, parent.archetype)
        phrases = ctx.valid_only(list(cands))
        scores = await ctx.in_thread(ctx.phrase_surrogate.ucb, phrases, self.kappa)
        order = sorted(range(len(phrases)), key=lambda i: -scores[i])[: self.send]
        picks = [phrases[i] for i in order]
        ctx.log(f"[surrogate_bo] pool {len(phrases)} -> {len(picks)} by UCB (rho {ctx.phrase_surrogate.rho:.2f})")
        await ctx.evaluate([(p, self.name, cands[p][0], cands[p][1]) for p in picks])


class GCGSwap(Strategy):
    """Greedy coordinate gradient on the word surrogate: HotFlip top-k swaps, verified by the oracle."""

    name = "gcg"

    def __init__(self, topk: int = 16, batch: int = 384, send: int = 48, steps: int = 3):
        self.topk = topk
        self.batch = batch
        self.send = send
        self.steps = steps

    def available(self, ctx) -> bool:
        return ctx.word_surrogate is not None and ctx.word_surrogate.ready and len(ctx.archive) >= 10

    async def run(self, ctx) -> None:
        rng = ctx.rng
        starts = ctx.parents(4)
        current = starts[rng.randrange(len(starts))]
        vocab = ctx.word_pool(400)
        for step in range(self.steps):
            words = current.words
            flips = await ctx.in_thread(ctx.word_surrogate.hotflip, words, vocab, self.topk)
            if not flips:
                return
            by_pos: dict[int, list[tuple[str, float]]] = {}
            for pos, word, gain in flips:
                by_pos.setdefault(pos, []).append((word, gain))
            cands: set[str] = set()
            for _ in range(self.batch * 2):
                if len(cands) >= self.batch:
                    break
                w = list(words)
                for _ in range(1 if rng.random() < 0.7 else 2):
                    pos = rng.choice(list(by_pos))
                    w[pos] = rng.choice(by_pos[pos])[0]
                phrase = " ".join(w)
                if phrase not in ctx.archive:
                    cands.add(phrase)
            phrases = ctx.valid_only(list(cands))
            if not phrases:
                return
            predicted = await ctx.in_thread(ctx.word_surrogate.predict, phrases)
            order = sorted(range(len(phrases)), key=lambda i: -predicted[i])[: self.send]
            scored = await ctx.evaluate([(phrases[i], self.name, current.phrase, current.archetype) for i in order])
            better = [c for c in scored if c.fitness > current.fitness]
            ctx.log(f"[gcg] step {step + 1}: {len(phrases)} swaps, {len(better)} beat parent {current.p:.3f}")
            if not better:
                return
            current = max(better, key=lambda c: c.fitness)
