"""Long-chain search for Strict Highest yes/no boards.

Chains of about LONG_TARGET words are built from a library of short fragments (diverse archive lines,
boosters, player words, cheap-LLM fragments, and pieces of LLM-written chains). A ridge fit on which fragments
each scored chain holds, plus block ablation on the best chains, says which fragments carry the score; the
recipes and fragment-level recombination lean on those. Once a chain holds the ceiling it is compacted in
stages (fragment drops, then halving windows) until word-level compression can take over."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..db import DB
from ..llm import LLMError
from ..objective import CEILING, logit, site_round
from ..oracle import question_key
from ..rules import RuleError, check_phrase, normalize
from . import boosters, prompts
from .strategies import CLAUSE_BREAKS, WORD_STAGE, Strategy

FRAG_MAX = 25
SPLIT_EVERY = 12
MIN_CHAINS_TO_FIT = 20
REFIT_EVERY = 50
RECIPES = ("repeat", "stack", "stack_strong_last", "repeat_best_block", "booster_interleave", "recombine")
WINDOWS = (32, 16, 8, 4, 2)
PROBE_LOW, PROBE_HIGH, PROBE_AIM = 0.4, 0.88, 0.7
PROBE_SATURATED = 0.92  # the best chains resist the claim this well: it has stopped telling them apart
PROBE_REPEATS = (1, 2, 4, 8)
PROBE_TOP = 3  # the best probed chains are re-measured to this many samples, so one lucky roll does not lead


def observed_ceiling(db: DB) -> float:
    """The highest rounded yes/no score seen anywhere: our own samples on yes/no questions and live leaders."""
    qkeys, slugs = [], []
    for row in db.all("SELECT slug FROM questions WHERE kind = 'noul'"):
        question = db.question(row["slug"])
        if question and question.get("jev_request"):
            qkeys.append(question_key(question["jev_request"]))
            slugs.append(row["slug"])
    best = CEILING
    if qkeys:
        marks = ",".join("?" * len(qkeys))
        row = db.one(f"SELECT MAX(noul) AS hi, MIN(noul) AS lo FROM oracle_samples WHERE qkey IN ({marks})",
                     tuple(qkeys))
        if row and row["hi"] is not None:
            best = max(best, site_round(float(row["hi"])), site_round(1.0 - float(row["lo"])))
    for slug in slugs:
        for board_row in db.board(slug, "strict_chain", "highScores"):
            if board_row.get("probability") is not None:
                best = max(best, site_round(float(board_row["probability"])))
    return min(best, 1.0)


def split_phrase(phrase: str, every: int = SPLIT_EVERY) -> list[str]:
    """Cut a chain into fragments at clause breaks, or every `every` words. The pieces rejoin to the chain."""
    pieces, cur = [], []
    for word in phrase.split():
        if cur and ((word.casefold() in CLAUSE_BREAKS and len(cur) >= 4) or len(cur) >= every):
            pieces.append(" ".join(cur))
            cur = []
        cur.append(word)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def kmeans(X: np.ndarray, k: int, rng, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Cosine k-means: (labels, unit-length centroids)."""
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    k = max(1, min(k, len(X)))
    centroids = X[rng.sample(range(len(X)), k)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = np.argmax(X @ centroids.T, axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members):
                c = members.mean(axis=0)
                centroids[j] = c / max(np.linalg.norm(c), 1e-9)
    return labels, centroids


@dataclass
class Fragment:
    id: int
    text: str
    source: str  # archive | booster | impact | llm | split
    cluster: int = -1  # -1: its own group
    value: float = 0.0  # ridge estimate, logit units per copy
    late: bool = False  # the ridge prefers it in the last third
    ablation: list[float] = field(default_factory=list)  # logit drop when it was removed from a top chain
    uses: int = 0
    exhausted: bool = False

    @property
    def units(self) -> int:
        return len(self.text.split())

    @property
    def score(self) -> float:
        drop = sum(self.ablation) / len(self.ablation) if self.ablation else 0.0
        return self.value + drop


class Probe:
    """Tells apart lines that round to the same score. The oracle answers in steps of 0.01, so near the top
    every line on a level reads the same and the search has nothing to climb. A short claim for the opposite
    answer, appended to the line, pulls the pair down to mid-range where a 0.01 step is fine; lines that sit
    closer to the next level resist the claim more. Probe samples are filed under their own key so the pairs
    never enter the question's history."""

    def __init__(self, qkey: str):
        self.qkey = f"{qkey}:probe"
        self.claim: str | None = None
        self.base: str | None = None
        self.tried = False
        self.values: dict[str, float] = {}
        self.samples: dict[str, int] = {}
        self.generation = 0
        self.stuck = False  # repeating the claim no longer pulls the best chains down

    @property
    def ready(self) -> bool:
        return self.claim is not None

    def pair(self, phrase: str, claim: str | None = None) -> str:
        return f"{phrase} {claim or self.claim}"

    async def _score(self, ctx, phrases: list[str], claim: str, n: int) -> dict[str, float]:
        room = ctx.engine.remaining() // max(n, 1)
        states = {p: self.pair(p, claim) for p in phrases[:room]}
        if not states:
            return {}
        scores = await ctx.engine.oracle.score(list(states.values()), n, qkey=self.qkey)
        return {p: ctx.objective.p(scores[s]) for p, s in states.items() if scores[s].n}

    async def claims(self, ctx) -> list[str]:
        opposite = "yes" if ctx.objective.goal == "no" else "no"
        title = normalize(ctx.question["title"]).rstrip("?")
        out: list[str] = []
        if ctx.llm is not None:
            try:
                data = await ctx.llm.json(ctx.writer_model(), prompts.GEN_SYSTEM, prompts.counter_user(ctx, 8),
                                          temperature=0.9, max_tokens=1500)
                rows = data.get("phrases") if isinstance(data, dict) else data
                out = [normalize(str(r.get("text") if isinstance(r, dict) else r)) for r in rows or []]
            except (LLMError, Exception) as error:
                ctx.log(f"[probe] claim call failed: {error}")
        out += [f"{opposite} {title} {opposite}", f"{title} absolutely {opposite}"]
        return [c for c in dict.fromkeys(out) if 2 <= len(c.split()) <= 20]

    async def calibrate(self, ctx, refs: list[str]) -> bool:
        """Pick the claim, repeated as often as needed, that brings the reference lines closest to PROBE_AIM."""
        self.tried = True
        found: list[tuple[float, str, str]] = []
        for base in await self.claims(ctx):
            claim, mean = await self._fit_repeats(ctx, refs, base, PROBE_REPEATS)
            if claim is not None:
                found.append((abs(mean - PROBE_AIM), claim, base))
                if len(found) >= 3:
                    break
        if not found:
            ctx.log("[probe] no opposing claim brings the top lines to mid-range; ranking by rounded score only")
            return False
        _, self.claim, self.base = min(found)
        ctx.log(f"[probe] ties on a level are broken by appending: {self.claim!r}")
        return True

    async def _fit_repeats(self, ctx, refs: list[str], base: str, repeats) -> tuple[str | None, float]:
        mean = 1.0
        claim = base
        for k in repeats:
            claim = " ".join([base] * k)
            got = await self._score(ctx, refs, claim, 1)
            if not got:
                break
            mean = sum(got.values()) / len(got)
            if mean <= PROBE_HIGH:
                break
        return (claim, mean) if PROBE_LOW <= mean <= PROBE_HIGH else (None, mean)

    def saturated(self, top: list[str]) -> bool:
        values = [self.values[p] for p in top if p in self.values]
        return len(values) >= PROBE_TOP and min(values) >= PROBE_SATURATED

    async def strengthen(self, ctx, top: list[str], remeasure: list[str]) -> bool:
        """The best chains now shrug the claim off. Repeat it more until they land mid-range again, then
        measure the strongest chains afresh: values from different claims do not compare."""
        if self.base is None or self.stuck:
            return False
        have = len(self.claim.split()) // max(len(self.base.split()), 1)
        claim, mean = await self._fit_repeats(ctx, top, self.base, [have * m for m in (2, 3, 4, 6, 8)])
        if claim is None:
            self.stuck = True
            ctx.log(f"[probe] the best chains resist even {have * 8} copies of the claim; keeping the old one")
            return False
        self.claim = claim
        self.generation += 1
        self.values, self.samples = {}, {}
        await self.measure(ctx, remeasure)
        ctx.log(f"[probe] claim strengthened to {len(claim.split())} words (top lines now {mean:.2f}); "
                f"{len(self.values)} chains re-measured")
        return True

    async def measure(self, ctx, phrases: list[str], n: int = 1) -> dict[str, float]:
        if not self.ready or not phrases:
            return {}
        got = await self._score(ctx, phrases, self.claim, n)
        for phrase, value in got.items():
            self.values[phrase] = value
            self.samples[phrase] = max(self.samples.get(phrase, 0), n)
        return got

    def value(self, phrase: str) -> float | None:
        return self.values.get(phrase)


class ChainLibrary:
    """Fragments, the fragment make-up of every chain we built or split, and the recipes over them."""

    def __init__(self, choices: list[str], cap: int, target: int, rng):
        self.choices = choices
        self.cap = cap
        self.target = min(target, cap)
        self.rng = rng
        self.frags: list[Fragment] = []
        self.by_text: dict[str, int] = {}
        self.parts: dict[str, list[int]] = {}
        self.centroids: np.ndarray | None = None
        self.embedder = None
        self.fitted_on = 0
        self.ablated: set[str] = set()
        self.exhausted_roots: set[str] = set()
        self.absorbed: set[str] = set()
        self.unassigned: list[Fragment] = []
        self.built = False
        self.building: asyncio.Future | None = None
        self.probe: Probe | None = None

    def merit(self, cand) -> float:
        """Rounded score first; on the same level the probe decides."""
        value = self.probe.value(cand.phrase) if self.probe is not None else None
        return site_round(cand.p) + 0.009 * (value if value is not None else 0.0)

    # Fragments.

    def add(self, text: str, source: str, cluster: int = -1, min_units: int = 1) -> Fragment | None:
        try:
            text = check_phrase(normalize(text), self.choices)
        except RuleError:
            return None
        units = len(text.split())
        if units < min_units or units > FRAG_MAX:
            return None
        if text in self.by_text:
            return self.frags[self.by_text[text]]
        frag = Fragment(len(self.frags), text, source, cluster)
        self.frags.append(frag)
        self.by_text[text] = frag.id
        return frag

    def live(self, source: str | None = None) -> list[Fragment]:
        return [f for f in self.frags if not f.exhausted and (source is None or f.source == source)]

    def stackable(self) -> list[Fragment]:
        return [f for f in self.live() if f.source != "booster"]

    def top_fragments(self, k: int = 12) -> list[str]:
        scored = [f for f in self.live() if f.value or f.ablation]
        return [f.text for f in sorted(scored, key=lambda f: -f.score)[:k]]

    def weighted(self, pool: list[Fragment]) -> Fragment:
        weights = [math.exp(1.5 * max(min(f.score, 4.0), -4.0)) / (1.0 + 0.02 * f.uses) for f in pool]
        return self.rng.choices(pool, weights)[0]

    # Chains.

    def units_of(self, ids: list[int]) -> int:
        return sum(self.frags[i].units for i in ids)

    def compose(self, ids: list[int]) -> str:
        return " ".join(self.frags[i].text for i in ids)

    def register(self, ids: list[int]) -> str:
        phrase = self.compose(ids)
        self.parts.setdefault(phrase, list(ids))
        return phrase

    def finish(self, ids: list[int]) -> list[int]:
        """Trim the weakest fragments until the chain fits the cap, and top short chains back up."""
        ids = list(ids)
        while len(ids) > 1 and self.units_of(ids) > self.cap:
            weakest = min(range(len(ids) - 1), key=lambda j: self.frags[ids[j]].score)
            del ids[weakest]
        pool = self.stackable()
        for _ in range(100):
            if not pool or self.units_of(ids) >= self.target * 0.8:
                break
            frag = self.weighted(pool)
            if self.units_of(ids) + frag.units > self.cap:
                break
            ids.insert(self.rng.randrange(len(ids) + 1), frag.id)
        return ids

    def stack(self, target: int, distinct: bool = True) -> list[int]:
        pool = self.stackable()
        if not pool:
            return []
        clusters = {f.cluster for f in pool if f.cluster >= 0}
        ids: list[int] = []
        seen: set[int] = set()
        for _ in range(400):
            if self.units_of(ids) >= target * 0.9:
                break
            frag = self.weighted(pool)
            if self.units_of(ids) + frag.units > self.cap:
                continue
            if distinct and frag.cluster >= 0 and frag.cluster in seen and len(seen) < len(clusters):
                continue
            ids.append(frag.id)
            seen.add(frag.cluster)
        return ids

    def recipe(self, name: str, scored: list[tuple[list[int], float]] | None = None) -> list[int]:
        pool = self.stackable()
        if not pool:
            return []
        if name == "repeat":
            frag = self.weighted(pool)
            ids = [frag.id] * max(1, self.target // max(frag.units, 1))
        elif name == "stack":
            ids = self.stack(self.target)
            self.rng.shuffle(ids)
        elif name == "stack_strong_last":
            ids = self.stack(self.target)
            if ids:
                best = max(range(len(ids)), key=lambda j: self.frags[ids[j]].score)
                ids.append(ids.pop(best))
        elif name == "repeat_best_block":
            block = self.stack(int(self.target * 0.4))
            ranked = sorted(set(block), key=lambda i: -self.frags[i].score)[: self.rng.choice((1, 2))]
            ids = list(block)
            while ranked and self.units_of(ids) < self.target * 0.9:
                ids += ranked
        elif name == "booster_interleave":
            base = self.stack(int(self.target * 0.8))
            extra = self.live("booster")
            ids = []
            for j, frag_id in enumerate(base):
                ids.append(frag_id)
                if extra and j % 2 == 1:
                    ids.append(self.weighted(extra).id)
        elif name == "recombine":
            ids = self.recombine(scored or [])
        else:
            raise ValueError(f"unknown recipe {name!r}")
        return self.finish(ids)

    def recombine(self, scored: list[tuple[list[int], float]]) -> list[int]:
        """Fragment-level GA step: crossover of two strong chains, then swap, drop, duplicate, or move."""
        if not scored:
            return self.stack(self.target)
        top = sorted(scored, key=lambda s: -s[1])[:12]
        weights = [1.0 / (1 + i) for i in range(len(top))]
        a = self.rng.choices(top, weights)[0][0]
        b = self.rng.choices(top, weights)[0][0]
        i = self.rng.randrange(1, max(len(a), 2))
        j = self.rng.randrange(0, max(len(b), 1))
        ids = list(a[:i]) + list(b[j:])
        pool = self.stackable()
        for _ in range(self.rng.choice((1, 2, 3))):
            op = self.rng.random()
            if not ids:
                break
            k = self.rng.randrange(len(ids))
            if op < 0.35 and pool:
                ids[k] = self.weighted(pool).id
            elif op < 0.55 and len(ids) > 2:
                del ids[k]
            elif op < 0.8:
                ids.insert(k, ids[k])
            else:
                ids.insert(self.rng.randrange(len(ids) + 1), ids.pop(k))
        return ids

    def scored_chains(self, archive) -> list[tuple[list[int], float]]:
        return [(ids, self.merit(archive.items[p])) for p, ids in self.parts.items() if p in archive.items]

    # Building the library.

    async def ensure(self, ctx) -> None:
        if self.built:
            return
        if self.building is None:
            self.building = asyncio.ensure_future(self.build(ctx))
        await self.building

    async def embedder_for(self, ctx):
        engine = ctx.engine
        if engine.embedder is None and engine.train_task is not None and not engine.train_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(engine.train_task), 120)
            except (asyncio.TimeoutError, Exception):
                pass
        self.embedder = engine.embedder
        return self.embedder

    async def cluster(self, ctx, texts: list[str]) -> list[int] | None:
        embedder = await self.embedder_for(ctx)
        if embedder is None or len(texts) < 4:
            return None
        X = await ctx.in_thread(embedder.embed, texts)
        labels, self.centroids = kmeans(np.asarray(X, dtype=np.float32), min(16, max(2, len(texts) // 4)), self.rng)
        return [int(v) for v in labels]

    async def assign(self, ctx, frags: list[Fragment]) -> None:
        """Nearest-centroid cluster for fragments added after the first clustering."""
        todo = [f for f in frags if f.cluster < 0 and f.source != "booster"]
        if not todo or self.centroids is None or self.embedder is None:
            return
        X = np.asarray(await ctx.in_thread(self.embedder.embed, [f.text for f in todo]), dtype=np.float32)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        for frag, label in zip(todo, np.argmax(X @ self.centroids.T, axis=1)):
            frag.cluster = int(label)

    async def build(self, ctx) -> None:
        lines = sorted((c for c in ctx.archive.items.values() if 3 <= c.units <= FRAG_MAX),
                       key=lambda c: -c.p)[:400]
        labels = await self.cluster(ctx, [c.phrase for c in lines])
        per_cluster: Counter = Counter()
        for index, cand in enumerate(lines):
            label = labels[index] if labels is not None else -1
            if labels is None and index >= 40:
                break
            if label >= 0 and per_cluster[label] >= 3:
                continue
            if self.add(cand.phrase, "archive", label, min_units=3) is not None and label >= 0:
                per_cluster[label] += 1
        try:
            rows = await ctx.in_thread(boosters.library, ctx.engine.db, ctx.objective.goal, 20)
        except Exception:
            rows = []
        for row in rows:
            self.add(row["text"], "booster")
        words = [w for w, impact in ctx.top_impacts(60) if impact > 0]
        for k in range(0, len(words), 5):
            self.add(" ".join(words[k : k + 5]), "impact", min_units=2)
        await self.llm_fragments(ctx, 60)
        await self.absorb(ctx)
        self.built = True
        sources = Counter(f.source for f in self.frags)
        ctx.log(f"[chains] fragment library: {len(self.frags)} fragments ("
                + ", ".join(f"{n} {s}" for s, n in sources.most_common()) + f"), "
                f"{len({f.cluster for f in self.frags if f.cluster >= 0})} clusters")

    async def llm_fragments(self, ctx, count: int) -> None:
        if ctx.llm is None:
            return
        try:
            data = await ctx.llm.json(ctx.writer_model(), prompts.gen_system(ctx), prompts.fragments_user(ctx, count),
                                      temperature=1.0, max_tokens=4000)
        except (LLMError, Exception) as error:
            ctx.log(f"[chains] fragment call failed: {error}")
            return
        rows = data.get("phrases") if isinstance(data, dict) else data
        added = []
        for row in rows or []:
            text = str(row.get("text") if isinstance(row, dict) else row)
            frag = self.add(text, "llm", min_units=3)
            if frag is not None:
                added.append(frag)
        await self.assign(ctx, added)

    async def absorb(self, ctx, min_p: float = 0.5) -> None:
        """Split long lines we did not build here (LLM chains, restored lines) into fragments and record their
        make-up, so attribution and compaction treat them like our own chains."""
        self.absorb_now(ctx.archive, min_p)
        added, self.unassigned = self.unassigned, []
        await self.assign(ctx, added)

    def absorb_now(self, archive, min_p: float = 0.5, restored: int = 200) -> None:
        """absorb() without the embedding call; new fragments get their cluster on the next absorb(). Of the
        lines restored from earlier runs only the best `restored` are split, or the library drowns in pieces."""
        pool = [c for c in archive.items.values() if c.units > WORD_STAGE and c.phrase not in self.parts
                and c.phrase not in self.absorbed]
        old = sorted((c for c in pool if not c.session), key=lambda c: -c.p)
        self.absorbed.update(c.phrase for c in old[restored:])
        for cand in [c for c in pool if c.session] + old[:restored]:
            self.absorbed.add(cand.phrase)
            if cand.p < min_p:
                continue
            ids = []
            for piece in split_phrase(cand.phrase):
                frag = self.add(piece, "split")
                if frag is None:
                    ids = []
                    break
                ids.append(frag.id)
                self.unassigned.append(frag)
            if ids and self.compose(ids) == cand.phrase:
                self.parts[cand.phrase] = ids

    # Attribution.

    def fit(self, archive) -> int:
        """Ridge on logit(p): per-fragment copies, first-third and last-third presence, plus most repeats,
        distinct fragments, and length. Once enough chains carry a probe value the fit uses those alone: at the
        top level every p is the same number and would only teach noise."""
        from sklearn.linear_model import Ridge

        rows = [(archive.items[p], ids) for p, ids in self.parts.items() if p in archive.items and ids]
        total = len(rows)
        probed = [(c, ids) for c, ids in rows if self.probe is not None and self.probe.value(c.phrase) is not None]
        target = lambda c: logit(c.p, 0.005)
        if len(probed) >= MIN_CHAINS_TO_FIT:
            rows = probed
            target = lambda c: logit(self.probe.value(c.phrase), 0.005)
        if len(rows) < MIN_CHAINS_TO_FIT:
            return 0
        used = sorted({i for _, ids in rows for i in ids})
        col = {f: k for k, f in enumerate(used)}
        F = len(used)
        X = np.zeros((len(rows), 3 * F + 3), dtype=np.float64)
        y = np.zeros(len(rows))
        for r, (cand, ids) in enumerate(rows):
            total = max(self.units_of(ids), 1)
            pos = 0
            for i in ids:
                units = self.frags[i].units
                mid = (pos + units / 2) / total
                X[r, col[i]] += 1
                if mid < 1 / 3:
                    X[r, F + col[i]] = 1
                elif mid > 2 / 3:
                    X[r, 2 * F + col[i]] = 1
                pos += units
            counts = Counter(ids)
            X[r, 3 * F] = max(counts.values())
            X[r, 3 * F + 1] = len(counts)
            X[r, 3 * F + 2] = total / max(self.target, 1)
            y[r] = target(cand)
        coef = Ridge(alpha=1.0).fit(X, y).coef_
        for f, k in col.items():
            frag = self.frags[f]
            frag.value = float(coef[k] + max(coef[F + k], coef[2 * F + k], 0.0))
            frag.late = bool(coef[2 * F + k] > coef[F + k])
        self.fitted_on = total
        return len(rows)

    def dominant(self, phrase: str):
        ids = self.parts.get(phrase)
        if not ids:
            return None
        best = max(set(ids), key=lambda i: (self.frags[i].score, self.frags[i].units))
        cluster = self.frags[best].cluster
        return cluster if cluster >= 0 else ("fragment", best)

    def novel(self, archive, round_new, near: float = 0.01) -> int:
        """New chains within `near` of the best earlier chain whose leading fragment comes from a cluster none
        of the top five earlier chains lead with. Short restored lines are no reference: nearly every chain
        differs from them."""
        fresh = {c.phrase for c in round_new}
        earlier = sorted((archive.items[p] for p in self.parts if p in archive.items and p not in fresh),
                         key=lambda c: -c.p)[:5]
        if not earlier:
            return 0
        floor = site_round(earlier[0].p) - near - 1e-9
        seen = {self.dominant(c.phrase) for c in earlier}
        return sum(1 for c in round_new if c.phrase in self.parts and site_round(c.p) >= floor
                   and self.dominant(c.phrase) not in seen)

    def strong(self, round_new, quantile: float = 0.8, min_values: int = 20) -> int:
        """New chains whose probe lands in the top fifth of everything measured before them."""
        probe = self.probe
        if probe is None or not probe.ready:
            return 0
        fresh = {c.phrase for c in round_new}
        before = sorted(v for p, v in probe.values.items() if p not in fresh)
        if len(before) < min_values:
            return 0
        bar = before[int(len(before) * quantile)]
        return sum(1 for p in fresh if (probe.value(p) or 0.0) > bar)


class LongChain(Strategy):
    """Build chains from one recipe, screen them, and score the most promising."""

    def __init__(self, recipe: str, build: int = 120, send: int = 48):
        self.recipe = recipe
        self.name = f"chain:{recipe}"
        self.build = build
        self.send = send

    def available(self, ctx) -> bool:
        lib = ctx.chains
        if lib is None or not ctx.objective.long or len(ctx.archive) == 0:
            return False
        if not lib.built:
            return True
        if self.recipe == "recombine":
            return len(lib.scored_chains(ctx.archive)) >= 10
        if self.recipe == "booster_interleave":
            return bool(lib.live("booster")) and bool(lib.stackable())
        return bool(lib.stackable())

    async def run(self, ctx) -> None:
        lib = ctx.chains
        await lib.ensure(ctx)
        await lib.absorb(ctx)
        scored_chains = lib.scored_chains(ctx.archive)
        if len(scored_chains) - lib.fitted_on >= REFIT_EVERY or (not lib.fitted_on
                                                                  and len(scored_chains) >= MIN_CHAINS_TO_FIT):
            fitted = await ctx.in_thread(lib.fit, ctx.archive)
            if fitted:
                top = lib.top_fragments(3)
                ctx.log(f"[chains] attribution refit on {fitted} chains; top fragments: " + " | ".join(top))
        made: dict[str, list[int]] = {}
        for _ in range(self.build * 3):
            if len(made) >= self.build:
                break
            ids = lib.recipe(self.recipe, scored_chains)
            if not ids:
                break
            phrase = lib.compose(ids)
            if phrase and phrase not in ctx.archive and phrase not in made:
                made[phrase] = ids
        if not made:
            ctx.log(f"[{self.name}] no new chains to build")
            return
        picks = await ctx.screen(list(made), self.send)
        for phrase in picks:
            lib.register(made[phrase])
            for i in set(made[phrase]):
                lib.frags[i].uses += 1
        scored = await ctx.evaluate([(p, self.name, "", "other") for p in picks])
        if scored:
            best = max(scored, key=lambda c: c.p)
            ctx.log(f"[{self.name}] {len(made)} built, {len(scored)} scored; best {best.p:.3f} {best.units}w, "
                    f"mean {sum(c.p for c in scored) / len(scored):.3f}")


class ChainAblate(Strategy):
    """Drop each fragment of the best chains once; the score lost is that fragment's measured worth."""

    name = "chain_ablate"

    def __init__(self, roots: int = 3):
        self.roots = roots

    def pending(self, ctx) -> list:
        lib = ctx.chains
        chains = [ctx.archive.items[p] for p in lib.parts if p in ctx.archive.items and p not in lib.ablated]
        return sorted((c for c in chains if c.p >= 0.5), key=lambda c: -c.p)[: self.roots]

    def available(self, ctx) -> bool:
        lib = ctx.chains
        return lib is not None and lib.built and ctx.objective.long and bool(self.pending(ctx))

    async def run(self, ctx) -> None:
        lib = ctx.chains
        for root in self.pending(ctx):
            lib.ablated.add(root.phrase)
            await ctx.evaluate([(root.phrase, "resample", root.parent, root.archetype)], n=2)
            root = ctx.archive.items[root.phrase]
            ids = lib.parts[root.phrase]
            children: dict[str, int] = {}
            for j in range(len(ids)):
                if len(ids) > 1:
                    children.setdefault(lib.register(ids[:j] + ids[j + 1 :]), ids[j])
            scored = await ctx.evaluate([(p, "ablate", root.phrase, root.archetype) for p in children], n=2)
            base = logit(root.p, 0.005)
            level = site_round(root.p)
            probe = lib.probe
            same = [c.phrase for c in scored if site_round(c.p) >= level - 1e-9]
            probed = {}
            if probe is not None and probe.ready and same:
                await probe.measure(ctx, [root.phrase], n=PROBE_TOP)
                probed = await probe.measure(ctx, same)
            root_probe = probe.value(root.phrase) if probed else None
            drops = {p: logit(root_probe, 0.005) - logit(v, 0.005) for p, v in probed.items()} if probed else {}
            worst_kept = max(list(drops.values()) + [0.0])
            for cand in scored:
                drop = drops.get(cand.phrase)
                if drop is None:
                    # losing the level costs more than any drop that kept it
                    drop = base - logit(cand.p, 0.005) + (worst_kept if drops else 0.0)
                lib.frags[children[cand.phrase]].ablation.append(drop)
            worst = sorted(set(ids), key=lambda i: lib.frags[i].score)[:2]
            ctx.log(f"[chain_ablate] {root.p:.3f} {root.units}w: {len(scored)} fragment drops; weakest "
                    + " | ".join(lib.frags[i].text for i in worst))


class ChainCompact(Strategy):
    """Cut a line that holds the ceiling (or already wins) down while it keeps that standing: first whole
    fragments, weakest first, then contiguous windows of 32, 16, 8, 4, and 2 words."""

    name = "chain_compact"

    def __init__(self, share: float = 0.25, send: int = 24):
        self.share = share
        self.send = send

    def candidates(self, ctx) -> list:
        lib = ctx.chains
        return [c for c in ctx.engine.roots() if c.units > WORD_STAGE and c.phrase not in lib.exhausted_roots]

    def available(self, ctx) -> bool:
        return ctx.chains is not None and ctx.objective.long and bool(self.candidates(ctx))

    def level(self, ctx, root) -> float:
        obj = ctx.objective
        return obj.ceiling if obj.holds_ceiling(root.score) else site_round(root.p)

    async def confirm(self, ctx, holders: list, level: float):
        for cand in holders[:3]:
            await ctx.evaluate([(cand.phrase, "resample", cand.parent, cand.archetype)], n=5)
            cand = ctx.archive.items[cand.phrase]
            if ctx.objective.holds(cand.score, level, 5):
                return cand
        return None

    async def step(self, ctx, root, phrases: list[str], level: float, origin: str):
        scored = await ctx.evaluate([(p, origin, root.phrase, root.archetype) for p in phrases], n=2)
        holders = [c for c in scored if ctx.objective.holds(c.score, level, 2) and c.units < root.units]
        holders.sort(key=lambda c: (c.units, -c.p))
        return await self.confirm(ctx, holders, level)

    async def run(self, ctx) -> None:
        lib = ctx.chains
        root = min(self.candidates(ctx), key=lambda c: (c.units, -c.p))
        start = root
        level = self.level(ctx, root)
        stop_at = ctx.engine.oracle.calls + max(300, int(ctx.engine.remaining() * self.share))

        def left() -> bool:
            return ctx.engine.oracle.calls < stop_at

        while left() and root.phrase in lib.parts and root.units > WORD_STAGE:
            ids = lib.parts[root.phrase]
            drops: dict[str, None] = {}
            for j in sorted(range(len(ids)), key=lambda j: lib.frags[ids[j]].score):
                if len(ids) > 1:
                    drops.setdefault(lib.register(ids[:j] + ids[j + 1 :]), None)
            best = await self.step(ctx, root, list(drops)[: self.send], level, "compact_fragment")
            if best is None:
                break
            ctx.log(f"[chain_compact] fragment drop: {root.units}w -> {best.units}w at {best.p:.3f}")
            root = best
        for width in WINDOWS:
            while left() and root.units > WORD_STAGE:
                words = root.words
                starts = list(range(0, len(words) - width + 1, width))
                if starts and starts[-1] + width < len(words):
                    starts.append(len(words) - width)
                cuts = list(dict.fromkeys(" ".join(words[:i] + words[i + width :]) for i in starts))
                best = await self.step(ctx, root, cuts, level, f"compact_w{width}")
                if best is None:
                    break
                ctx.log(f"[chain_compact] window {width}: {root.units}w -> {best.units}w at {best.p:.3f}")
                root = best
        if root.phrase == start.phrase:
            lib.exhausted_roots.add(root.phrase)
            lead = ctx.leader
            if lead is None or root.units >= lead.units:
                for i in set(lib.parts.get(root.phrase, [])):
                    lib.frags[i].exhausted = True
                ctx.engine.scheduler.boost = {"chain": 1.6, "llm_gen": 1.3}
                ctx.log(f"[chain_compact] {root.units}w at {level:.2f} will not shrink; its fragments are "
                        "exhausted, building resumes with other clusters")
            else:
                ctx.log(f"[chain_compact] {root.units}w at {level:.2f} will not shrink further")
        elif root.units <= WORD_STAGE:
            ctx.log(f"[chain_compact] {start.units}w -> {root.units}w; word-level compression takes over")
