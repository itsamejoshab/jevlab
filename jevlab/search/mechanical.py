"""Systematic escalation strategies: precision climbing, exhaustive sweeps, word-by-word beam building."""

from __future__ import annotations

import asyncio
import math

from ..llm import LLMError
from ..objective import site_round
from . import prompts, vocab
from .strategies import FUNCTION_WORDS, Strategy

NOISE_SD = 0.009  # oracle per-sample sd measured in calibration
# Per-roll sd by distance from the middle, pooled over every line with 5+ rolls: the noise shrinks toward the
# ends, so one fixed sd makes precision steps near the ceiling look three times less certain than they are.
NOISE_BY_LEVEL = ((0.6, 0.013), (0.7, 0.012), (0.8, 0.010), (0.9, 0.008), (0.95, 0.0055), (1.01, 0.0045))


def noise_sd(p: float) -> float:
    extremity = max(p, 1.0 - p)
    return next(sd for bound, sd in NOISE_BY_LEVEL if extremity < bound)


def board_key(ctx):
    """What the board ranks, as a key function over candidates."""
    return lambda cand: ctx.objective.score_key(cand.score, cand.units)


class LevelStrategy(Strategy):
    """Only runs while the engine sits on its escalation level."""

    level = 0

    def available(self, ctx) -> bool:
        return ctx.engine.level == self.level and len(ctx.archive) > 0


def single_edits(words: list[str], pool: list[str], rng, per_slot: int = 12) -> list[str]:
    """Deletions, adjacent swaps, substitutions and insertions around one line."""
    out: list[str] = []
    for i in range(len(words)):
        out.append(" ".join(words[:i] + words[i + 1 :]))
    for i in range(len(words) - 1):
        w = words[:]
        w[i], w[i + 1] = w[i + 1], w[i]
        out.append(" ".join(w))
    for i in range(len(words)):
        for alt in rng.sample(pool, min(per_slot, len(pool))):
            if alt != words[i]:
                out.append(" ".join(words[:i] + [alt] + words[i + 1 :]))
    for gap in range(len(words) + 1):
        for word in rng.sample(pool, min(per_slot // 2 or 1, len(pool))):
            out.append(" ".join(words[:gap] + [word] + words[gap:]))
    return [p for p in dict.fromkeys(out) if p]


def sd_of(cand) -> float:
    """A line's own spread once it has enough rolls to trust, else the pooled noise at its level."""
    pooled = noise_sd(cand.p)
    return max(cand.score.spread, pooled) if cand.score.n >= 8 else pooled


def margin(a, b, z: float = 1.0) -> float:
    return z * math.sqrt(sd_of(a) ** 2 / max(a.score.n, 1) + sd_of(b) ** 2 / max(b.score.n, 1))


async def race(ctx, cands: list, rounds: tuple[int, ...] = (2, 4, 8), keep: float = 0.5, origin: str = "resample"):
    """Successive halving on dithered means: top everyone up to the first n, keep the better half, repeat. Lines
    that tie on the rounded score are told apart by how often their rolls land on the upper step."""
    alive = list({c.phrase: c for c in cands}.values())
    for i, n in enumerate(rounds):
        if not alive:
            break
        await ctx.evaluate([(c.phrase, origin, c.parent, c.archetype) for c in alive if c.score.n < n], n=n)
        alive = sorted((ctx.archive.items[c.phrase] for c in alive), key=lambda c: -c.p)
        if i < len(rounds) - 1:
            alive = alive[: max(1, math.ceil(len(alive) * keep))]
    return alive


class PrecisionClimb(LevelStrategy):
    """L1. The oracle rounds to 0.01 but its noise dithers, so the mean of n samples resolves ~0.009/sqrt(n).
    Hill-climb on those means, accepting a step only when it clears one combined standard error."""

    name = "precision"
    level = 1

    def __init__(self, parents: int = 5, n_parent: int = 8, n_child: int = 4, send: int = 48, steps: int = 4):
        self.parents = parents
        self.n_parent = n_parent
        self.n_child = n_child
        self.send = send
        self.steps = steps

    async def run(self, ctx) -> None:
        tops = ctx.climb_roots(self.parents)
        await ctx.evaluate([(c.phrase, "resample", c.parent, c.archetype) for c in tops], n=self.n_parent)
        tops = [ctx.archive.items[c.phrase] for c in tops]
        parent = max(tops, key=lambda c: c.fitness)
        pool = ctx.word_pool(120)
        for step in range(self.steps):
            neighbours = [p for p in single_edits(parent.words, pool, ctx.rng) if p not in ctx.archive]
            picks = await ctx.screen(neighbours, self.send)
            scored = await ctx.evaluate([(p, self.name, parent.phrase, parent.archetype) for p in picks],
                                        n=self.n_child)
            if not scored:
                return
            # The best of ~50 noisy means is flattered; race the leading few before trusting any of them.
            leading = sorted(scored, key=lambda c: -c.fitness)[:6]
            best = max(await race(ctx, leading, (self.n_child, self.n_parent)), key=lambda c: c.fitness)
            need = margin(best, parent)
            if ctx.objective.shortest:
                shorter_tie = board_key(ctx)(best) > board_key(ctx)(parent)
            else:
                shorter_tie = site_round(best.p) >= site_round(parent.p) and best.units < parent.units
            if best.p - parent.p <= need and not shorter_tie:
                ctx.log(f"[precision] step {step + 1}: best neighbour {best.p:.4f} vs parent {parent.p:.4f} "
                        f"(needs +{need:.4f}); parent holds")
                return
            await ctx.evaluate([(best.phrase, "resample", best.parent, best.archetype)], n=self.n_parent)
            best = ctx.archive.items[best.phrase]
            if best.p - parent.p <= margin(best, parent) / 2 and not shorter_tie:
                ctx.log(f"[precision] step {step + 1}: {best.p:.4f} fell back after re-scoring; parent holds")
                return
            ctx.log(f"[precision] step {step + 1}: {parent.p:.4f} -> {best.p:.4f} {best.units}w {best.phrase}")
            parent = best


class Extend(Strategy):
    """Plateau mode. Grow the climb root toward the leader's length a clause at a time: pieces of our other strong
    lines, boosters, and a cheap model's supporting clauses go in at every clause boundary, and the ones that tie
    the root's rounded score are raced on dithered means. The next root is whichever line then measures best."""

    name = "extend"

    def __init__(self, send: int = 80, width: int = 40):
        self.send = send
        self.width = width
        self.written: dict[str, list[str]] = {}

    def available(self, ctx) -> bool:
        roots = ctx.climb_roots(1) if len(ctx.archive) else []
        return ctx.engine.plateau and bool(roots) and roots[0].units + 2 <= ctx.engine.grow_cap

    async def llm_clauses(self, ctx, root) -> list[str]:
        if ctx.llm is None:
            return []
        if root.phrase not in self.written:
            try:
                data = await ctx.llm.json(ctx.writer_model(), prompts.GEN_SYSTEM, prompts.clauses_user(ctx, root.phrase, 30),
                                          temperature=1.0, max_tokens=2000)
                rows = data.get("phrases") if isinstance(data, dict) else data
                self.written[root.phrase] = [str(r.get("text") if isinstance(r, dict) else r) for r in rows or []]
            except (LLMError, Exception) as error:
                ctx.log(f"[extend] clause call failed: {error}")
                self.written[root.phrase] = []
        return self.written[root.phrase]

    async def clauses(self, ctx, root) -> list[str]:
        from ..rules import RuleError, check_phrase, normalize
        from . import boosters
        from .chains import split_phrase

        raw: list[str] = await self.llm_clauses(ctx, root)
        for cand in ctx.archive.top(60):
            if cand.phrase != root.phrase:
                raw += split_phrase(cand.phrase, 8)
        if ctx.engine.yes_no:
            try:
                raw += [row["text"] for row in
                        await ctx.in_thread(boosters.library, ctx.engine.db, ctx.objective.goal, 20)]
            except Exception:
                pass
        out: dict[str, None] = {}
        for text in raw:
            try:
                clean = check_phrase(normalize(text), ctx.engine.choices, 12)
            except RuleError:
                continue
            if 2 <= len(clean.split()) and clean not in root.phrase:
                out.setdefault(clean, None)
        return list(out)

    async def run(self, ctx) -> None:
        from .chains import split_phrase

        root = ctx.climb_roots(1)[0]
        await ctx.evaluate([(root.phrase, "resample", root.parent, root.archetype)], n=8)
        root = ctx.archive.items[root.phrase]
        words = root.words
        gaps, pos = {0, len(words)}, 0
        for piece in split_phrase(root.phrase, 6):
            pos += len(piece.split())
            gaps.add(pos)
        room = ctx.engine.grow_cap - root.units
        grown = {" ".join(words[:g] + clause.split() + words[g:]) for clause in await self.clauses(ctx, root)
                 if len(clause.split()) <= room for g in gaps}
        grown = [p for p in grown if p not in ctx.archive]
        if not grown:
            ctx.log("[extend] nothing new to insert")
            return
        picks = await ctx.screen(grown, self.send)
        scored = await ctx.evaluate([(p, self.name, root.phrase, root.archetype) for p in picks])
        tied = sorted((c for c in scored if site_round(c.p) >= site_round(root.p)), key=lambda c: -c.p)
        if not tied:
            ctx.log(f"[extend] {len(scored)} clause insertions, none holds {site_round(root.p):.2f}")
            return
        raced = await race(ctx, tied[: self.width], (2, 4, 8))
        best = raced[0]
        gained = best.p - root.p
        ctx.log(f"[extend] {len(scored)} insertions, {len(tied)} hold {site_round(root.p):.2f}; best {best.p:.4f} "
                f"{best.units}w vs root {root.p:.4f} {root.units}w"
                + (f" (+{gained:.4f}, clears noise)" if gained > margin(best, root) else ""))


class Drift(Strategy):
    """Plateau mode, for lines pinned at the top rounded level where every roll is identical and dithered means
    say nothing. What still varies is the neighbourhood: across past questions, a 0.98 line whose one-word edits
    mostly hold 0.98 was far likelier to have a 0.99 edit (within-question AUC 0.84) than one whose edits fall away.
    So walk the plateau: each step tries edits of the current line, measures how well the edits that hold keep
    holding under their own edits, and moves to the steadiest. The walk's position is kept across runs."""

    name = "drift"

    def __init__(self, send: int = 40, scouts: int = 4, probe: int = 12, slack: float = 0.0, steps: int = 3):
        self.send, self.scouts, self.probe, self.slack, self.steps = send, scouts, probe, slack, steps
        self.line: str | None = None
        self.vocab: list[str] | None = None

    def available(self, ctx) -> bool:
        return ctx.engine.plateau and len(ctx.archive) > 0

    def restore(self, line: str | None) -> None:
        self.line = line or None

    def pool(self, ctx) -> list[str]:
        grow = ctx.engine.strategies.get("grow")
        learned = grow.best_words(60) if grow is not None else []
        return list(dict.fromkeys(learned + ctx.word_pool(150) + (self.vocab or [])[:300]))

    def edits(self, ctx, phrase: str, k: int) -> list[str]:
        words = phrase.split()
        per = max(2, k // max(len(words), 1) + 1)
        cands = [e for e in single_edits(words, self.pool(ctx), ctx.rng, per)
                 if e not in ctx.archive and len(e.split()) <= ctx.engine.grow_cap]
        return ctx.rng.sample(cands, min(k, len(cands)))

    def hold_rate(self, ctx, phrase: str, level: float) -> tuple[float, int]:
        """Share of the line's evaluated edits that hold the level, shrunk toward even odds so a dozen lucky
        edits can't outrank fifty honest ones."""
        kids = [c for c in ctx.archive.items.values() if c.parent == phrase]
        holds = sum(site_round(c.p) >= level for c in kids)
        return (holds + 1) / (len(kids) + 2), len(kids)

    def start(self, ctx):
        """The steadiest line at the top level, judged against lines of like length."""
        top = site_round(ctx.archive.top_by_board(1)[0].p)
        kids: dict[str, list[float]] = {}
        for c in ctx.archive.items.values():
            if c.parent:
                kids.setdefault(c.parent, []).append(site_round(c.p))
        best, best_rate = None, -1.0
        for cand in ctx.archive.items.values():
            ks = kids.get(cand.phrase, [])
            if site_round(cand.p) < top or cand.units > ctx.engine.grow_cap or len(ks) < 10:
                continue
            rate = sum(k >= top for k in ks) / len(ks)
            if rate > best_rate:
                best, best_rate = cand, rate
        return best or ctx.climb_roots(1)[0]

    async def run(self, ctx) -> None:
        if self.vocab is None:
            self.vocab = await ctx.in_thread(vocab.build, ctx, 4000)
        for _ in range(self.steps):
            if ctx.engine.remaining() < self.send + self.scouts * self.probe * 3:
                return
            await self.step(ctx)

    async def step(self, ctx) -> None:
        cur = ctx.archive.items.get(self.line) if self.line else None
        if cur is None:
            cur = self.start(ctx)
            ctx.log(f"[drift] starting from {cur.units}w {cur.phrase}")
        level = site_round(cur.p)
        scored = await ctx.evaluate([(p, self.name, cur.phrase, cur.archetype) for p in self.edits(ctx, cur.phrase, self.send)])
        above = [c for c in scored if site_round(c.p) > level]
        if above:
            raced = await race(ctx, above, (2, 4, 8))
            ctx.log(f"[drift] {len(above)} edits rolled above {level:.2f}; best mean {raced[0].p:.4f} {raced[0].phrase}")
        here, n_here = self.hold_rate(ctx, cur.phrase, level)
        holders = [c for c in scored if site_round(c.p) >= level]
        ctx.rng.shuffle(holders)
        field = holders[: self.scouts]
        for probe in (self.probe, self.probe * 2):
            for cand in field:
                await ctx.evaluate([(p, self.name, cand.phrase, cand.archetype)
                                    for p in self.edits(ctx, cand.phrase, probe)])
            field.sort(key=lambda c: -self.hold_rate(ctx, c.phrase, level)[0])
            field = field[: max(1, len(field) // 2)]
        rates = [(self.hold_rate(ctx, c.phrase, level)[0], c) for c in field]
        if rates and rates[0][0] >= here - self.slack:
            rate, nxt = rates[0]
            self.line = nxt.phrase
            ctx.log(f"[drift] {len(holders)}/{len(scored)} edits hold {level:.2f} (rate {here:.2f} over {n_here}); "
                    f"moved to {nxt.units}w line holding {rate:.2f}: {nxt.phrase}")
        else:
            self.line = cur.phrase
            ctx.log(f"[drift] {len(holders)}/{len(scored)} edits hold {level:.2f} (rate {here:.2f} over {n_here}); "
                    "no steadier neighbour, staying")


PROBE_SD = 0.02  # per-roll sd of a probe pair near mid-range (measured 0.008-0.015 on hard boards), rounded up
PROBE_TOO_HIGH = 0.92  # the walk's line shrugs the claim off; repeat it until the line lands mid-range again


class ProbeClimb(Strategy):
    """Plateau mode. Every line on the top rounded level rolls the same number, but a probe (a short claim for
    the other answer appended to the line) pulls it to mid-range, where lines nearer the next level resist more.
    On boards we did clear, 0.99 lines out-probed their 0.98 neighbours (AUC ~0.8); on the hard boards, tied 0.98
    lines spread 4-6x wider under the probe than its own roll noise. So climb on the probe value: edits of the
    current line that hold the level are probed once, the front runners re-probed, and the walk moves to the best
    when it clears the noise. The claim and the walk's position are kept across runs, and a pair already probed
    is read back from the sample cache for free."""

    name = "probe_climb"

    def __init__(self, send: int = 48, probe_k: int = 24, finalists: int = 4, n_final: int = 3, steps: int = 3,
                 patience: int = 4):
        self.send, self.probe_k, self.finalists, self.n_final = send, probe_k, finalists, n_final
        self.steps, self.patience = steps, patience
        self.probe = None
        self.saved: dict = {}
        self.line: str | None = None
        self.failed = False
        self.stuck = 0
        self.generation = 0
        self.scores: dict = {}  # phrase -> probe Score under the current claim

    def available(self, ctx) -> bool:
        return ctx.engine.plateau and len(ctx.archive) > 0 and not self.failed

    def restore(self, saved: dict | None) -> None:
        self.saved = dict(saved or {})
        self.line = self.saved.get("line") or None

    def state(self) -> dict:
        probe = self.probe
        return {"line": self.line or "", "claim": probe.claim if probe else self.saved.get("claim"),
                "base": probe.base if probe else self.saved.get("base")}

    def value(self, ctx, phrase: str) -> float | None:
        score = self.scores.get(phrase)
        return ctx.objective.p(score) if score is not None and score.n else None

    def gap(self, a: str, b: str) -> float:
        na, nb = self.scores[a].n, self.scores[b].n
        return PROBE_SD * math.sqrt(1 / max(na, 1) + 1 / max(nb, 1))

    async def measure(self, ctx, phrases: list[str], n: int) -> None:
        need = [p for p in dict.fromkeys(phrases) if p not in self.scores or self.scores[p].n < n]
        room = ctx.engine.remaining() // max(n, 1)
        pairs = {p: self.probe.pair(p) for p in need[:room]}
        if not pairs:
            return
        got = await ctx.engine.oracle.score(list(pairs.values()), n, qkey=self.probe.qkey)
        for phrase, pair in pairs.items():
            if got[pair].n:
                self.scores[phrase] = got[pair]

    def level(self, ctx) -> float:
        return site_round(ctx.climb_roots(1)[0].p)

    def tied(self, ctx, level: float) -> list:
        cap = ctx.engine.grow_cap
        return [c for c in ctx.archive.items.values() if site_round(c.p) >= level and c.units <= cap]

    async def ready(self, ctx) -> bool:
        from .chains import Probe

        if self.probe is None:
            self.probe = Probe(ctx.engine.oracle.qkey)
            if self.saved.get("claim"):
                self.probe.claim, self.probe.base, self.probe.tried = self.saved["claim"], self.saved.get("base"), True
                ctx.log(f"[probe_climb] reusing the claim from the last run: {self.probe.claim!r}")
        if self.probe.ready:
            return True
        level = self.level(ctx)
        refs = sorted(self.tied(ctx, level), key=lambda c: -c.score.n)[:8]
        ctx.log(f"[probe_climb] {len(self.tied(ctx, level))} lines tie at {level:.2f}; fitting a probe claim on {len(refs)}")
        if not await self.probe.calibrate(ctx, [c.phrase for c in refs]):
            self.failed = True
            ctx.log("[probe_climb] no claim pulls the tied lines to mid-range; leaving the plateau to drift")
            return False
        return True

    async def start(self, ctx):
        """Probe a spread of tied lines and walk from whichever resists the claim best."""
        level = self.level(ctx)
        tied = self.tied(ctx, level)
        pool = sorted(tied, key=lambda c: -c.score.n)[:200]
        picks = ctx.rng.sample(pool, min(16, len(pool)))
        await self.measure(ctx, [c.phrase for c in picks], 1)
        ranked = sorted((c for c in picks if self.value(ctx, c.phrase) is not None),
                        key=lambda c: -self.value(ctx, c.phrase))
        if not ranked:
            return None
        await self.measure(ctx, [c.phrase for c in ranked[:4]], self.n_final)
        best = max(ranked[:4], key=lambda c: self.value(ctx, c.phrase))
        ctx.log(f"[probe_climb] probed {len(picks)} of {len(tied)} lines at {level:.2f}: "
                f"{self.value(ctx, ranked[-1].phrase):.3f}..{self.value(ctx, best.phrase):.3f}; "
                f"walking from {best.units}w {best.phrase}")
        return best

    async def run(self, ctx) -> None:
        if not await self.ready(ctx):
            return
        from .engine import BudgetExhausted

        for i in range(self.steps):
            if ctx.engine.remaining() < self.send + self.probe_k + self.finalists * self.n_final:
                if i == 0:
                    raise BudgetExhausted()
                return
            await self.step(ctx)

    async def step(self, ctx) -> None:
        if self.probe.generation != self.generation:
            self.generation, self.scores = self.probe.generation, {}
        cur = ctx.archive.items.get(self.line) if self.line else None
        if cur is None or site_round(cur.p) < self.level(ctx):
            cur = await self.start(ctx)
            if cur is None:
                return
            self.line, self.stuck = cur.phrase, 0
        await self.measure(ctx, [cur.phrase], self.n_final)
        here = self.value(ctx, cur.phrase)
        if here is None:
            return
        if here >= PROBE_TOO_HIGH and await self.probe.strengthen(ctx, [cur.phrase], [cur.phrase]):
            return
        level = site_round(cur.p)
        drift = ctx.engine.strategies["drift"]
        if drift.vocab is None:
            drift.vocab = await ctx.in_thread(vocab.build, ctx, 4000)
        picks = await ctx.screen(drift.edits(ctx, cur.phrase, self.send * 4), self.send)
        scored = await ctx.evaluate([(p, self.name, cur.phrase, cur.archetype) for p in picks])
        above = [c for c in scored if site_round(c.p) > level]
        if above:
            raced = await race(ctx, above, (2, 4))
            self.line, self.stuck = raced[0].phrase, 0
            ctx.log(f"[probe_climb] {len(above)} edits rolled above {level:.2f}; walking on from "
                    f"{raced[0].p:.4f} {raced[0].units}w {raced[0].phrase}")
            return
        holders = [c for c in scored if site_round(c.p) >= level][: self.probe_k]
        await self.measure(ctx, [c.phrase for c in holders], 1)
        probed = sorted((c for c in holders if self.value(ctx, c.phrase) is not None),
                        key=lambda c: -self.value(ctx, c.phrase))
        if probed:
            await self.measure(ctx, [c.phrase for c in probed[: self.finalists]], self.n_final)
            best = max(probed[: self.finalists], key=lambda c: self.value(ctx, c.phrase))
            there = self.value(ctx, best.phrase)
            if there - here > self.gap(best.phrase, cur.phrase):
                self.line, self.stuck = best.phrase, 0
                ctx.log(f"[probe_climb] {len(holders)}/{len(scored)} edits hold {level:.2f}; probe {here:.3f} -> "
                        f"{there:.3f}, moved to {best.units}w {best.phrase}")
                return
        self.stuck += 1
        top = f"best edit probes {self.value(ctx, probed[0].phrase):.3f}" if probed else "none probed"
        if self.stuck >= self.patience:
            self.line, self.stuck = None, 0
            ctx.log(f"[probe_climb] {len(holders)}/{len(scored)} hold {level:.2f}, {top} vs {here:.3f}; "
                    f"no better neighbour in {self.patience} steps, restarting from another tied line")
        else:
            ctx.log(f"[probe_climb] {len(holders)}/{len(scored)} hold {level:.2f}, {top} vs {here:.3f}; staying")


class ScenarioJoin(Strategy):
    """Plateau mode. Our tuned lines pile up words that argue the answer, and on the hard boards every edit of
    them stalls one step short. What moved two of them past it was a sentence setting the stakes, joined to a tuned
    line: a situation where the wrong answer would be absurd (a taunt given to someone asking how to stop a
    building collapsing), then the salad. A writer model sets such scenes, sees which scenes' joins did best, and
    each scene is joined in both orders with several of our lines at the top level. Scenes lower the joins on some
    boards, so after `dry` rounds with nothing above the level it steps aside."""

    name = "scenario_join"
    needs_llm = True

    def __init__(self, count: int = 30, calls: int = 2, width: int = 25, partners: int = 6, dry: int = 2):
        self.count, self.calls, self.width, self.partners, self.dry = count, calls, width, partners, dry
        self.rated: dict[str, float] = {}
        self.misses = 0

    def available(self, ctx) -> bool:
        return ctx.engine.plateau and ctx.llm is not None and len(ctx.archive) > 0 and self.misses < self.dry

    async def scenes(self, ctx) -> list[str]:
        from ..rules import RuleError, check_phrase, normalize

        best = sorted(((v, s) for s, v in self.rated.items()), reverse=True)[:8]
        user = prompts.scenario_user(ctx, best, self.count)
        replies = await asyncio.gather(*(ctx.llm.json(ctx.writer_model(), prompts.GEN_SYSTEM, user, temperature=1.0,
                                                      max_tokens=6000) for _ in range(self.calls)),
                                       return_exceptions=True)
        out: dict[str, None] = {}
        for data in replies:
            if isinstance(data, BaseException):
                ctx.log(f"[scenario_join] writer call failed: {data}")
                continue
            rows = data.get("phrases") if isinstance(data, dict) else data
            for row in rows or []:
                text = str(row.get("text") if isinstance(row, dict) else row)
                try:
                    clean = check_phrase(normalize(text), ctx.engine.choices, 40)
                except RuleError:
                    continue
                if clean not in ctx.archive and clean not in self.rated:
                    out.setdefault(clean, None)
        return list(out)

    async def run(self, ctx) -> None:
        from .engine import BudgetExhausted

        if ctx.engine.remaining() < self.count * self.calls + self.width * self.partners * 2:
            raise BudgetExhausted()
        cap = ctx.engine.grow_cap
        level = site_round(ctx.climb_roots(1)[0].p)
        tied = sorted((c for c in ctx.archive.items.values() if site_round(c.p) >= level and c.units < cap),
                      key=lambda c: -c.score.n)[:30]
        scenes = await self.scenes(ctx)
        if not tied or not scenes:
            self.misses += 1
            return
        alone = await ctx.evaluate([(s, self.name, "", "other") for s in scenes])
        joins: dict[str, str] = {}
        for scene in sorted(alone, key=lambda c: -c.p)[: self.width]:
            for line in ctx.rng.sample(tied, min(self.partners, len(tied))):
                for joined in (f"{scene.phrase} {line.phrase}", f"{line.phrase} {scene.phrase}"):
                    if len(joined.split()) <= cap and joined not in ctx.archive:
                        joins.setdefault(joined, line.phrase)
        scored = await ctx.evaluate([(j, self.name, parent, ctx.archive.items[parent].archetype) for j, parent in joins.items()])
        by_scene: dict[str, list[float]] = {}
        for cand in scored:
            for scene in alone:
                if cand.phrase.startswith(scene.phrase + " ") or cand.phrase.endswith(" " + scene.phrase):
                    by_scene.setdefault(scene.phrase, []).append(cand.p)
        for scene, values in by_scene.items():
            self.rated[scene] = sum(values) / len(values)
        holds = sum(site_round(c.p) >= level for c in scored)
        above = [c for c in scored if site_round(c.p) > level]
        if above:
            self.misses = 0
            raced = await race(ctx, above, (2, 4, 8))
            ctx.log(f"[scenario_join] {len(above)} of {len(scored)} joins rolled above {level:.2f}; best mean "
                    f"{raced[0].p:.4f} {raced[0].units}w {raced[0].phrase}")
            return
        self.misses += 1
        top = max(self.rated.items(), key=lambda kv: kv[1]) if self.rated else ("", 0.0)
        ctx.log(f"[scenario_join] {len(scenes)} scenes, {holds}/{len(scored)} joins hold {level:.2f}, none above"
                f" ({self.misses}/{self.dry} dry); best scene joins average {top[1]:.3f}: {top[0]}")


def apply_edits(words: list[str], edits: list[tuple[str, int, str]]) -> list[str]:
    """edits: ("sub", i, word) replaces slot i; ("ins", g, word) inserts before original slot g (g may be len)."""
    subs = {i: w for kind, i, w in edits if kind == "sub"}
    ins: dict[int, list[str]] = {}
    for kind, g, w in edits:
        if kind == "ins":
            ins.setdefault(g, []).append(w)
    out: list[str] = []
    for i in range(len(words) + 1):
        out.extend(ins.get(i, []))
        if i < len(words):
            out.append(subs.get(i, words[i]))
    return out


class Sweep(LevelStrategy):
    """L2. Coordinate ascent: every slot and every gap against a large vocabulary (sized to the budget),
    then pairwise combinations of the best single edits. Stops at a line no single change improves."""

    name = "sweep"
    level = 2

    def __init__(self, share: float = 0.5, pair_top: int = 20, chunk: int = 300, cap: int = 1500,
                 give_up: float = 0.4):
        self.share = share
        self.pair_top = pair_top
        self.chunk = chunk
        self.cap = cap  # most calls one round may plan, whatever the remaining budget
        self.give_up = give_up  # share of a round's singles after which no board progress ends the sweep
        self.race_width = 80  # most tied single edits one round races
        self.vocab: list[str] | None = None
        self.progress = 0.0
        self.optimal: set[str] = set()

    def available(self, ctx) -> bool:
        return super().available(ctx) and ctx.climb_roots(1)[0].phrase not in self.optimal

    async def candidates(self, ctx, words: list[str], per: int) -> dict[tuple[int, str], list[str]]:
        """Vocabulary per position: the word surrogate's HotFlip ranking when trusted, else vocab order."""
        vocab_list = self.vocab or []
        out: dict[tuple[int, str], list[str]] = {}
        ranked_sub: dict[int, list[str]] = {}
        surrogate = ctx.word_surrogate
        if per < len(vocab_list) and surrogate is not None and surrogate.ready:
            flips = await ctx.in_thread(surrogate.hotflip, words, vocab_list[:8000], per)
            for pos, word, _gain in flips:
                ranked_sub.setdefault(pos, []).append(word)
        head = vocab_list[:per]
        for i, word in enumerate(words):
            ranked = ranked_sub.get(i, [])
            picks = list(dict.fromkeys(ranked + head))[:per]
            picks += [c for c in vocab.casings(word) if c != word]
            out[(i, "sub")] = [w for w in picks if w != word]
        if len(words) < ctx.engine.grow_cap:
            for gap in range(len(words) + 1):
                out[(gap, "ins")] = head
        return out

    async def run(self, ctx) -> None:
        if self.vocab is None:
            self.vocab = await ctx.in_thread(vocab.build, ctx)
            ctx.log(f"[sweep] vocabulary {len(self.vocab)} words")
        root = ctx.climb_roots(1)[0]
        # Highest: edits that tie the rounded score are raced on dithered means, so the root needs a sharp mean.
        fine = not ctx.objective.shortest
        await ctx.evaluate([(root.phrase, "resample", root.parent, root.archetype)], n=8 if fine else 5)
        root = ctx.archive.items[root.phrase]
        lead = ctx.leader
        extending = bool(lead) and not ctx.objective.shortest and root.units < lead.units * 0.8
        # While growing toward the leader's length, extra words are the point, so only the score counts.
        key = (lambda c: (site_round(c.p), 0)) if extending else board_key(ctx)

        def beats(c, r) -> bool:
            if key(c) > key(r):
                return True
            return fine and site_round(c.p) >= site_round(r.p) and c.score.n >= 4 and c.p - r.p > margin(c, r)

        def ties(c, r) -> bool:
            return site_round(c.p) >= site_round(r.p)
        for round_no in range(6 if extending else 3):
            words = root.words
            positions = len(words) + (len(words) + 1 if len(words) < ctx.engine.grow_cap else 0)
            spend = min(self.cap, max(200, int(ctx.engine.remaining() * self.share)))
            per = max(20, min(len(self.vocab), spend // max(positions, 1)))
            plan = await self.candidates(ctx, words, per)
            edits: dict[str, tuple[str, int, str]] = {}
            rank: dict[str, int] = {}
            for (pos, kind), options in plan.items():
                for i, word in enumerate(options):
                    phrase = " ".join(apply_edits(words, [(kind, pos, word)]))
                    if phrase != root.phrase:
                        edits.setdefault(phrase, (kind, pos, word))
                        rank.setdefault(phrase, i)
            # Each slot's most promising options first, so an early stop has already tried every position.
            phrases = sorted(ctx.valid_only(list(edits)), key=lambda p: rank.get(p, 0))
            ctx.log(f"[sweep] round {round_no + 1}: {len(words)}w line, {positions} positions x {per} words "
                    f"= {len(phrases)} single edits{' (extending toward the leader length)' if extending else ''}")
            scored = []
            for start in range(0, len(phrases), self.chunk):
                self.progress = start / max(len(phrases), 1)
                batch = phrases[start : start + self.chunk]
                scored += await ctx.evaluate([(p, self.name, root.phrase, root.archetype) for p in batch])
                done = start + len(batch)
                hopeful = (lambda c: ties(c, root)) if fine else (lambda c: key(c) > key(root))
                if done < len(phrases) and done >= len(phrases) * self.give_up \
                        and not any(hopeful(c) for c in scored):
                    self.progress = 1.0
                    self.optimal.add(root.phrase)
                    ctx.log(f"[sweep] giving up after {done}/{len(phrases)} edits: none beats "
                            f"{site_round(root.p):.2f} at {root.units}w")
                    return
            self.progress = 1.0
            singles = sorted((c for c in scored if c.phrase in edits), key=lambda c: c.fitness, reverse=True)
            if fine:
                tied = [c for c in singles if ties(c, root)][: self.race_width]
                if tied:
                    raced = await race(ctx, tied, (2, 4))
                    singles = raced + [c for c in singles if c.phrase not in {r.phrase for r in raced}]
                    ctx.log(f"[sweep] raced {len(tied)} edits that tie {site_round(root.p):.2f}; "
                            f"best mean {raced[0].p:.4f} (root {root.p:.4f})")
            top = singles[: self.pair_top]
            pairs: dict[str, str] = {}
            for a in range(len(top)):
                for b in range(a + 1, len(top)):
                    ea, eb = edits[top[a].phrase], edits[top[b].phrase]
                    if ea[:2] == eb[:2]:
                        continue
                    pairs[" ".join(apply_edits(words, [ea, eb]))] = root.phrase
            pair_scored = await ctx.evaluate([(p, self.name, root.phrase, root.archetype) for p in pairs], n=2)
            contenders = sorted(top[:5] + pair_scored, key=lambda c: c.fitness, reverse=True)[:5]
            await ctx.evaluate([(c.phrase, "resample", c.parent, c.archetype) for c in contenders],
                               n=8 if fine else 5)
            contenders = [ctx.archive.items[c.phrase] for c in contenders]
            best = max(contenders, key=lambda c: c.fitness) if contenders else None
            if best is None or not beats(best, root):
                self.optimal.add(root.phrase)
                ctx.log(f"[sweep] local optimum: no single or paired change beats "
                        f"{site_round(root.p):.2f} at {root.units}w")
                return
            ctx.log(f"[sweep] {root.p:.3f} {root.units}w -> {best.p:.3f} {best.units}w {best.phrase}")
            root = best


class SingleWord(Strategy):
    """Shortest yes: most boards are led by one word, so score single words (and pairs) directly. Each round
    sends LLM ideas, casings and pairs of our best singles, the best player words, then the next slice of the
    big vocabulary."""

    name = "single_word"
    needs_llm = False

    def __init__(self, send: int = 240):
        self.send = send
        self.round = 0
        self.vocab: list[str] | None = None
        self.cursor = 0

    def available(self, ctx) -> bool:
        return ctx.objective.shortest and len(ctx.archive) > 0

    async def llm_words(self, ctx) -> list[str]:
        from ..config import GEN_MODEL

        try:
            data = await ctx.llm.json(GEN_MODEL, prompts.GEN_SYSTEM_SHORTEST, prompts.single_words_user(ctx, 60),
                                      temperature=1.0, max_tokens=3000)
        except (LLMError, Exception) as error:
            ctx.log(f"[single_word] llm call failed: {error}")
            return []
        rows = (data.get("phrases") if isinstance(data, dict) else data) or []
        return [str(r.get("text") if isinstance(r, dict) else r).strip() for r in rows if r]

    async def run(self, ctx) -> None:
        self.round += 1
        if self.vocab is None:
            self.vocab = await ctx.in_thread(vocab.build, ctx, 8000)
        pool: list[str] = []
        if ctx.llm is not None and self.round % 2 == 1:
            pool += await self.llm_words(ctx)
        singles = sorted((c for c in ctx.archive.items.values() if c.units == 1), key=lambda c: -c.p)[:12]
        for cand in singles:
            pool += vocab.casings(cand.phrase)
        tops = [c.phrase for c in singles[:8]]
        pool += [f"{a} {b}" for a in tops for b in tops if a != b]
        pool += [f"{w} {t}" for t in tops[:4] for w in ("yes", "no", "definitely", "absolutely", "my", "our")]
        pool += [w for w, impact in ctx.top_impacts(150) if impact > 0]
        pool = ctx.valid_only([p for p in dict.fromkeys(pool) if p and p not in ctx.archive])[: self.send]
        while len(pool) < self.send and self.cursor < len(self.vocab):
            word = self.vocab[self.cursor]
            self.cursor += 1
            if word not in ctx.archive and word not in pool:
                pool.append(word)
        if not pool:
            return
        scored = await ctx.evaluate([(p, self.name, "", "wild") for p in pool])
        best = max(scored, key=lambda c: (ctx.objective.score_key(c.score, c.units), c.p), default=None)
        if best is not None:
            ctx.log(f"[single_word] {len(scored)} short lines (vocab {self.cursor}/{len(self.vocab)}); best "
                    f"{best.p:.3f} {best.units}w {best.phrase}")


class BeamBuild(LevelStrategy):
    """L4. Build lines from nothing, one word at a time; every prefix is a scoreable phrase."""

    name = "beam"
    level = 4

    def __init__(self, width: int = 16, expand: int = 20, steps_per_run: int = 6):
        self.width = width
        self.expand = expand
        self.steps_per_run = steps_per_run
        self.beam: list = []
        self.depth = 0
        self.vocab: list[str] | None = None

    async def next_words(self, ctx, prefixes: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {p: [] for p in prefixes}
        if ctx.llm is not None:
            from ..config import GEN_MODEL_ALT

            body = "\n".join(f"{i}: {p or '(start)'}" for i, p in enumerate(prefixes))
            user = (f"Question: {ctx.question['title']}\n{prompts.target_line(ctx)}\n"
                    f"For each partial phrase below, list {self.expand} single next words that would push a "
                    "reader's answer toward the target when the phrase continues. Single plain words only "
                    "(Latin letters or digits, no punctuation).\n" + body +
                    '\nReturn JSON: {"next": {"<index>": ["word", ...]}}')
            try:
                data = await ctx.llm.json(GEN_MODEL_ALT, "Reply with JSON only.", user, temperature=1.0,
                                          max_tokens=4000)
                for key, words in ((data or {}).get("next") or {}).items():
                    try:
                        prefix = prefixes[int(key)]
                    except (ValueError, IndexError):
                        continue
                    out[prefix] = [str(w).strip() for w in words or [] if str(w).strip()][: self.expand]
            except (LLMError, Exception) as error:
                ctx.log(f"[beam] next-word call failed: {error}")
        fill = ctx.word_pool(200) + (self.vocab or [])[:400]
        for prefix in prefixes:
            extra = ctx.rng.sample(fill, min(self.expand // 2, len(fill)))
            out[prefix] = list(dict.fromkeys(out[prefix] + extra + ctx.rng.sample(FUNCTION_WORDS, 3)))
        return out

    async def run(self, ctx) -> None:
        if self.vocab is None:
            self.vocab = await ctx.in_thread(vocab.build, ctx, 4000)
        cap = ctx.engine.grow_cap
        if not self.beam or self.depth >= cap:
            self.beam, self.depth = [""], 0
            ctx.log("[beam] starting a new beam from an empty line")
        for _ in range(self.steps_per_run):
            if ctx.engine.remaining() < self.width * self.expand:
                return
            prefixes = [c if isinstance(c, str) else c.phrase for c in self.beam]
            proposals = await self.next_words(ctx, prefixes)
            children = []
            for prefix, words in proposals.items():
                for word in words:
                    children.append((f"{prefix} {word}".strip(), self.name, prefix, "other"))
            scored = await ctx.evaluate(children)
            if not scored:
                return
            scored.sort(key=lambda c: c.fitness, reverse=True)
            self.beam = scored[: self.width]
            self.depth += 1
            head = self.beam[0]
            ctx.log(f"[beam] depth {self.depth}: best {head.p:.3f} {head.phrase}")
            if self.depth >= cap:
                return


class Grow(BeamBuild):
    """Plateau mode. The leader's own method, done frugally: build a fresh line one appended word at a time up
    to one under the leader's length, with a narrow beam that keeps at most two children per parent. Near the
    top, children that tie on the rounded score are raced before the beam is chosen. The beam is kept in the
    question's memory, so each day's run keeps growing the same lines."""

    name = "grow"
    level = -1

    def __init__(self, width: int = 4, expand: int = 12, steps_per_run: int = 8, per_parent: int = 2):
        super().__init__(width, expand, steps_per_run)
        self.per_parent = per_parent
        self.gains: dict[str, list[float]] = {}  # word -> [sum of score change when appended, count]

    def available(self, ctx) -> bool:
        return ctx.engine.plateau and len(ctx.archive) > 0

    def restore(self, prefixes: list[str], gains: dict | None = None) -> None:
        if prefixes:
            self.beam = list(prefixes)
            self.depth = max(len(p.split()) for p in prefixes)
        for word, (total, count) in (gains or {}).items():
            self.gains[word] = [float(total), float(count)]

    def state(self) -> list[str]:
        return [c if isinstance(c, str) else c.phrase for c in self.beam]

    def best_words(self, k: int) -> list[str]:
        """Words whose appends have raised the score most on average, shrunk toward zero when rarely seen."""
        ranked = sorted(self.gains.items(), key=lambda kv: -kv[1][0] / (kv[1][1] + 2))
        return [w for w, (total, _) in ranked[:k] if total > 0]

    def learn(self, ctx, scored: list) -> None:
        for cand in scored:
            parent = ctx.archive.items.get(cand.parent)
            base = parent.p if parent is not None else None
            if base is None or not cand.phrase.startswith(cand.parent):
                continue
            word = cand.phrase[len(cand.parent):].strip()
            entry = self.gains.setdefault(word, [0.0, 0.0])
            entry[0] += cand.p - base
            entry[1] += 1

    async def next_words(self, ctx, prefixes: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {p: [] for p in prefixes}
        if ctx.llm is not None:
            from ..config import GEN_MODEL_ALT

            site = [w for w, i in ctx.top_impacts(30) if i > 0]
            ours = self.best_words(20)
            body = "\n".join(f"{i}: {p or '(start)'}" for i, p in enumerate(prefixes))
            user = (f"Question: {ctx.question['title']}\n{prompts.target_line(ctx)}\n"
                    "We are building a line one word at a time; every appended word is scored by how strongly a "
                    "reader of the whole line would answer with the target. Blunt, decisive words beat filler.\n"
                    + (f"Words that raised the score for other players: {', '.join(site)}\n" if site else "")
                    + (f"Words that raised the score in our own appends: {', '.join(ours)}\n" if ours else "")
                    + f"For each partial line below, list {self.expand} different single next words. Plain words "
                    "only (Latin letters or digits, no punctuation). Avoid repeating words already in the line.\n"
                    + body + '\nReturn JSON: {"next": {"<index>": ["word", ...]}}')
            try:
                data = await ctx.llm.json(GEN_MODEL_ALT, "Reply with JSON only.", user, temperature=1.0,
                                          max_tokens=4000)
                for key, words in ((data or {}).get("next") or {}).items():
                    try:
                        prefix = prefixes[int(key)]
                    except (ValueError, IndexError):
                        continue
                    out[prefix] = [str(w).strip() for w in words or [] if str(w).strip()][: self.expand]
            except (LLMError, Exception) as error:
                ctx.log(f"[grow] next-word call failed: {error}")
        learned = self.best_words(40)
        fill = ctx.word_pool(200) + (self.vocab or [])[:400]
        for prefix in prefixes:
            llm = out[prefix][: self.expand // 2]
            known = ctx.rng.sample(learned, min(self.expand // 4, len(learned)))
            extra = ctx.rng.sample(fill, self.expand)
            out[prefix] = list(dict.fromkeys(llm + known + extra))[: self.expand]
        return out

    async def run(self, ctx) -> None:
        if self.vocab is None:
            self.vocab = await ctx.in_thread(vocab.build, ctx, 4000)
        cap = ctx.engine.grow_cap
        if not self.beam or self.depth >= cap:
            self.beam, self.depth = [""], 0
            ctx.log(f"[grow] starting a fresh line, to grow up to {cap} words")
        for _ in range(self.steps_per_run):
            if ctx.engine.remaining() < self.width * self.expand * 2:
                return
            prefixes = self.state()
            proposals = await self.next_words(ctx, prefixes)
            children = [(f"{prefix} {word}".strip(), self.name, prefix, "other")
                        for prefix, words in proposals.items() for word in words]
            scored = await ctx.evaluate(children)
            if not scored:
                return
            self.learn(ctx, scored)
            scored.sort(key=lambda c: c.fitness, reverse=True)
            # The best of ~50 single rolls is mostly luck; re-roll the front runners before choosing.
            raced = await race(ctx, scored[: self.width * 2], (2, 4))
            scored = raced + [c for c in scored if c.phrase not in {r.phrase for r in raced}]
            beam, per = [], {}
            for cand in scored:
                if per.get(cand.parent, 0) < self.per_parent:
                    beam.append(cand)
                    per[cand.parent] = per.get(cand.parent, 0) + 1
                if len(beam) >= self.width:
                    break
            self.beam = beam
            self.depth += 1
            head = self.beam[0]
            ctx.log(f"[grow] {head.units}w: best {head.p:.4f} (n={head.score.n}) {head.phrase}")
            if self.depth >= cap:
                return
