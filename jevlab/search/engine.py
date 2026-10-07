"""The lab: one question, one mode, maximize the oracle's P(goal), then shorten."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable

from .. import vault
from ..config import (
    GEMINI_CALLS,
    GEN_MODEL,
    GEN_MODEL_ALT,
    GEN_TIERS,
    LONG_PLAN_CALLS,
    LONG_TARGET,
    PLAN_MODEL,
    PLATEAU_LEADER_WORDS,
    TRIAGE_K,
)
from ..db import DB
from ..llm import LLM, LLMError
from ..modes import BOARDS, HIGH_SCORES, from_board, is_searchable
from ..objective import (
    CEILING_MIN_N,
    Leader,
    Objective,
    board_leader,
    goal_p,
    logit,
    objective_for,
    site_round,
    target_rows,
)
from ..oracle import Oracle, Score, history
from ..rules import CopyGuard, RuleError, check_phrase, check_word, normalize, option_names
from ..rules.banned import contains as phrase_banned
from ..rules.strict import MAX_WORDS
from . import global_model
from .proxy import ProxyScreen
from . import memory as lab_memory
from . import prompts
from . import triage
from ..transfer import jev_db as triage_jev_db
from .advanced import GCGSwap, Genetic, SurrogateBO
from .archive import ARCHETYPES, Archive, Candidate
from .boosters import Boosters
from .chains import PROBE_TOP, RECIPES, ChainAblate, ChainCompact, ChainLibrary, LongChain, Probe, observed_ceiling
from .mechanical import BeamBuild, Drift, Extend, Grow, PrecisionClimb, ProbeClimb, ScenarioJoin, SingleWord, Sweep
from .scheduler import Scheduler
from .strategies import FUNCTION_WORDS, WORD_STAGE, Compress, LLMGenerate, LocalEdit
from .surrogate import Embedder, PhraseSurrogate, WordSurrogate


class BudgetExhausted(Exception):
    pass


# Escalation ladder: (name, strategy that leads the level, extra scheduler boosts).
LEVELS = [
    ("normal", None, {}),
    ("averaging", "precision", {}),
    ("sweep", "sweep", {}),
    ("boosters", "boosters", {}),
    ("reframe", "beam", {"llm_gen": 1.6, "surrogate_bo": 1.2}),
]
# Long-chain regime: word-level arms wait until a line at the ceiling (or winning) is short enough for them.
WORD_ARMS = {"compress", "precision", "sweep", "local_edit", "genetic", "gcg", "surrogate_bo"}
SHORT_ONLY_ARMS = {"boosters", "single_word", "beam"}


@dataclass
class Event:
    kind: str  # log | scored | best | round | vault | plan | surrogate | status
    data: dict


class LabContext:
    """Everything a strategy may touch."""

    def __init__(self, engine: "Engine"):
        self.engine = engine
        self.rng = random.Random(engine.seed)
        self.directive = ""
        self.focus: list[str] = []
        self.banned: set[str] = set()
        self.pinned: set[str] = set()
        self.synonyms: dict[str, list[str]] = {}
        self.phrase_surrogate: PhraseSurrogate | None = None
        self.word_surrogate: WordSurrogate | None = None
        self.chains: ChainLibrary | None = None

    # Shortcuts.
    question = property(lambda self: self.engine.question)
    objective = property(lambda self: self.engine.objective)
    archive = property(lambda self: self.engine.archive)
    llm = property(lambda self: self.engine.llm)
    leader = property(lambda self: self.engine.leader)
    memory = property(lambda self: self.engine.memory)
    fresh = property(lambda self: self.engine.fresh)

    def exhausted_lines(self, k: int = 10) -> list[tuple[str, float, int]]:
        """Frames earlier runs already pushed as far as they go without winning."""
        rows = [(r[0], float(r[1]), int(r[2])) for r in self.memory.exhausted]
        seen = {r[0] for r in rows}
        for cand in self.archive.top_by_board(40):
            if not cand.session and cand.phrase not in seen:
                rows.append((cand.phrase, cand.p, cand.units))
                seen.add(cand.phrase)
        return sorted(rows, key=lambda r: -r[1])[:k]

    def single_word_lines(self, k: int | None = None) -> list[Candidate]:
        """One-word lines in board order. The only lines that can win a one-word Shortest yes race."""
        lines = [c for c in self.archive.items.values() if c.units == 1]
        lines.sort(key=lambda c: (self.objective.score_key(c.score, c.units), c.p), reverse=True)
        return lines if k is None else lines[:k]

    def parents(self, k: int) -> list[Candidate]:
        """Top lines to build on. On a fresh start half come from this run, so edits leave the old peak.
        Against a one-word Shortest yes leader, edits stay on one-word lines."""
        if self.engine.one_word_race:
            singles = self.single_word_lines(k)
            if singles:
                return singles
        if not self.fresh:
            return self.archive.top(k)
        new = self.archive.top_session(k // 2)
        out = {c.phrase: c for c in new}
        for cand in self.archive.top(k):
            if len(out) >= k:
                break
            out.setdefault(cand.phrase, cand)
        return list(out.values())

    def log(self, message: str) -> None:
        self.engine.emit("log", message=message)

    def climb_roots(self, k: int) -> list[Candidate]:
        """Lines to climb from. On Highest below the leader, the board's tie-break on length does not matter
        yet; the best dithered mean does, discounted by its noise so one lucky roll does not lead. Winning or
        on Shortest yes, board order."""
        from .mechanical import noise_sd

        if self.engine.one_word_race:
            singles = self.single_word_lines(k)
            if singles:
                return singles
        if self.objective.shortest or self.engine.winning():
            return self.archive.top_by_board(k)
        cap = self.engine.grow_cap
        pool = [c for c in self.archive.items.values() if c.units <= cap]
        return sorted(pool, key=lambda c: -(c.p - noise_sd(c.p) / c.score.n**0.5))[:k]

    def writer_model(self) -> str:
        """Model for rewrites, synonyms, and fragments; the long-chain regime keeps them on the cheap one."""
        return GEN_MODEL_ALT if self.engine.frugal() else GEN_MODEL

    async def in_thread(self, fn: Callable, *args):
        return await asyncio.to_thread(fn, *args)

    def valid_only(self, phrases: list[str]) -> list[str]:
        out = []
        for phrase in phrases:
            try:
                phrase = check_phrase(phrase, self.engine.choices, self.engine.length_cap)
            except RuleError:
                continue
            if self.engine.one_word_race and self.objective.units(phrase) != 1:
                continue
            out.append(phrase)
        return out

    def rank(self, phrases: list[str], k: int) -> list[str]:
        phrases = self.valid_only(list(dict.fromkeys(phrases)))
        if len(phrases) <= k:
            return phrases
        surrogate = self.phrase_surrogate
        if surrogate is not None and surrogate.ready:
            scores = surrogate.ucb(phrases, 1.0)
            order = sorted(range(len(phrases)), key=lambda i: -scores[i])
            exploit = [phrases[i] for i in order[: int(k * 0.8)]]
            rest = [phrases[i] for i in order[int(k * 0.8) :]]
            return exploit + self.rng.sample(rest, min(k - len(exploit), len(rest)))
        return self.rng.sample(phrases, k)

    async def screen(self, phrases: list[str], k: int) -> list[str]:
        """rank(), but before the predictor is trusted a stand-in model orders the pool instead of chance."""
        surrogate = self.phrase_surrogate
        proxy = self.engine.proxy
        if (surrogate is not None and surrogate.ready) or proxy is None or not proxy.enabled:
            return await self.in_thread(self.rank, phrases, k)
        phrases = self.valid_only(list(dict.fromkeys(phrases)))
        if len(phrases) <= k:
            return phrases
        pool = self.rng.sample(phrases, min(len(phrases), proxy.MAX_POOL))
        scores = await proxy.scores(pool)
        if len(scores) < k:
            return self.rank(phrases, k)
        order = sorted(scores, key=lambda p: -scores[p])
        exploit = order[: int(k * 0.8)]
        rest = [p for p in phrases if p not in set(exploit)]
        return exploit + self.rng.sample(rest, min(k - len(exploit), len(rest)))

    def top_impacts(self, k: int) -> list[tuple[str, float]]:
        rows = sorted(self.engine.impacts, key=lambda r: -(r.get("averageImpact") or 0))
        return [(r["word"], float(r.get("averageImpact") or 0)) for r in rows[:k]]

    def word_pool(self, k: int) -> list[str]:
        ordered: dict[str, None] = {}
        for word in self.pinned:
            ordered.setdefault(word, None)
        for word, impact in self.top_impacts(200):
            if impact > 0:
                ordered.setdefault(word, None)
        for word in self.archive.vocabulary(40):
            ordered.setdefault(word, None)
        for alts in self.synonyms.values():
            for word in alts:
                ordered.setdefault(word, None)
        out = []
        for word in ordered:
            if word.casefold() in self.banned:
                continue
            try:
                out.append(check_word(word))
            except RuleError:
                continue
        return out[:k] if out else FUNCTION_WORDS[:k]

    async def alternatives(self, words: list[str]) -> dict[str, list[str]]:
        missing = [w for w in dict.fromkeys(words) if w not in self.synonyms and w.casefold() not in FUNCTION_WORDS]
        if missing and self.llm is not None:
            try:
                data = await self.llm.json(
                    self.writer_model(), prompts.GEN_SYSTEM, prompts.synonyms_user(self, missing[:30]), temperature=0.9
                )
                alts = data.get("alts") if isinstance(data, dict) else {}
                for word, values in (alts or {}).items():
                    clean = []
                    for value in values or []:
                        try:
                            clean.append(check_word(normalize(str(value)).split()[0]))
                        except (RuleError, IndexError):
                            continue
                    self.synonyms[word] = clean
            except (LLMError, Exception) as error:
                self.log(f"[synonyms] {error}")
        for word in missing:
            self.synonyms.setdefault(word, [])
        return {w: self.synonyms.get(w, []) for w in words}

    def archetype_bests(self) -> dict[str, float]:
        best: dict[str, float] = {}
        for cand in self.archive.items.values():
            best[cand.archetype] = max(best.get(cand.archetype, 0.0), cand.p)
        return best

    async def evaluate(self, items: list[tuple[str, str, str, str]], n: int = 1) -> list[Candidate]:
        return await self.engine.evaluate(items, n)


class Engine:
    def __init__(
        self,
        db: DB,
        slug: str,
        mode: str = "strict_chain",
        budget: int = 20000,
        use_llm: bool = True,
        seed: int | None = None,
        plan_every: int = 10,
        on_event: Callable[[Event], None] | None = None,
        idle_when_done: bool = False,
        embedder: Embedder | None = None,
        max_stall: int = 0,
        min_rounds: int = 8,
        escalate: bool = True,
        max_level: int = 4,
        board: str = HIGH_SCORES,
        target: str = "",
        win_extra: int = 0,
        triage_k: int = TRIAGE_K,
        question: dict | None = None,
    ):
        """`target` aims a choice question's search at one answer: scores are P(target), the leader is the best
        row filed under that answer, and run memory and vault lines are kept per answer.
        `win_extra` > 0 is win mode: the first vaulted line that beats the leader caps the run at that many
        more oracle calls.
        `triage_k` > 0 scores that many existing lines (estimated Kev vault lines, Jev's vault and history) with
        the seeds, before any are generated.
        `question` is an in-memory question (Live Mode); when set, the snapshot is not required. A typed
        sandbox question is not filed. A site live round is filed under its slug."""
        if mode != "strict_chain":
            raise ValueError("v1 searches strict_chain only")
        if board not in BOARDS:
            raise ValueError(f"unknown board {board!r}")
        self.db = db
        self.slug = slug
        self.mode = mode
        self.board = board
        self.budget = budget
        self.budget_span = max(budget, 1)
        self.seed = seed if seed is not None else int(time.time())
        self.plan_every = plan_every
        self.on_event = on_event or (lambda event: None)
        self.use_llm = use_llm
        self.idle_when_done = idle_when_done
        self.in_command = False
        self.max_stall = max_stall
        self.win_extra = win_extra
        self.won_at: int | None = None
        self.min_rounds = min_rounds
        self.current_strategy = ""
        self.origin_wins: dict[str, int] = {}
        self.novel_by_strategy: dict[str, int] = {}
        self.round_new: list[Candidate] = []
        self.restored = 0
        self.end_reason = ""
        self.memory = lab_memory.RunMemory()
        self.fresh = False
        self.escalate = escalate
        self.max_level = max(0, min(max_level, len(LEVELS) - 1))
        self.triage_k = triage_k
        self.level = 0
        self.peak_level = 0

        if question is None:
            question = db.question(slug)
        if question is None:
            raise ValueError(f"{slug!r} is not in the snapshot; run `jevlab snapshot` first")
        question.setdefault("raw", {})
        if not question.get("jev_request"):
            raise ValueError(f"{slug!r} has no jevRequest")
        if not is_searchable(question["kind"], bool(question["raw"].get("ranked"))):
            raise ValueError(
                f"{slug!r} is a {question['kind']} question; search handles yes/no, scales, and unranked choice"
            )
        self.question = question
        self.live = bool(question["raw"].get("live"))
        self.choices = option_names(question["raw"].get("choices"))
        # Yes/no-only helpers: the proxy reads yes/no logprobs, the prior and boosters were learned on yes/no.
        self.yes_no = question["kind"] == "noul"
        if target and target not in self.choices:
            raise ValueError(f"{target!r} is not an answer of {slug!r}; answers: {', '.join(self.choices) or 'none'}")
        self.target = target
        self.memory_board = f"{board}@{target}" if target else board
        self.objective = objective_for(question, board=board)
        self.me = db.me()
        if target:
            rows = target_rows({b: db.board(slug, mode, b) for b in (board, "champions")}, board, target)
        else:
            rows = db.board(slug, mode, board)
        self.leader: Leader | None = board_leader(rows, self.me, board=board)
        self.our_best: Leader | None = board_leader(
            [r for r in rows if self.me and r.get("userId") == self.me], self.me, include_ours=True, board=board
        )
        self.long = not self.objective.shortest and self.objective.kind == "noul" and LONG_TARGET > 0
        self.long_target = LONG_TARGET
        # Plateau mode (decided once history is loaded): the leader holds the ceiling with a long line we cannot
        # match with the ordinary length cap, and earlier runs already exhausted the ordinary search.
        self.plateau = False
        self.plateau_candidate = (
            not self.long
            and not self.objective.shortest
            and self.objective.kind == "noul"
            and self.leader is not None
            and self.leader.units >= PLATEAU_LEADER_WORDS
            and site_round(self.leader.probability) >= observed_ceiling(db) - 1e-9
        )
        self.expensive_used = 0
        self.plans = 0
        self._roots: tuple[int, list[Candidate]] = (-1, [])
        if self.objective.shortest:
            # Any length stays valid so restored long lines can be compressed; only growth is capped.
            # A one-word leader is beaten only by another one-word line, so nothing is grown past that.
            self.length_cap = MAX_WORDS
            self.grow_cap = 1 if self.one_word_race else (max(8, self.leader.units + 2) if self.leader else 8)
        elif self.long:
            if self.leader:
                self.objective.par_units = self.leader.units
            self.objective.long_units = LONG_TARGET
            self.objective.ceiling = observed_ceiling(db)
            self.length_cap = self.grow_cap = max(LONG_TARGET, MAX_WORDS)
        else:
            if self.leader:
                self.objective.par_units = self.leader.units
            self.length_cap = min(MAX_WORDS, max(30, self.leader.units + 10 if self.leader else 30))
            self.grow_cap = self.length_cap
        self.impacts = db.word_impacts(slug, mode)
        self.archive = Archive(self.objective)
        self.guard = CopyGuard()
        self.ctx = LabContext(self)
        self.oracle: Oracle | None = None
        self.llm: LLM | None = None
        self.proxy: ProxyScreen | None = None
        generators = [LLMGenerate(models, tier) for tier, models in GEN_TIERS.items() if models]
        chain_arms = [LongChain(r) for r in RECIPES] + [ChainAblate(), ChainCompact()] if self.long else []
        self.strategies = {
            s.name: s
            for s in (
                *generators,
                *chain_arms,
                LocalEdit(),
                Compress(),
                Genetic(),
                SurrogateBO(),
                GCGSwap(),
                PrecisionClimb(),
                Sweep(),
                *([Boosters()] if self.yes_no else []),
                BeamBuild(),
                SingleWord(),
                Extend(),
                Grow(),
                Drift(),
                ProbeClimb(),
                ScenarioJoin(),
            )
        }
        priors = {g.name: 1.1 for g in generators}
        priors.update(
            {
                "local_edit": 1.0,
                "compress": 0.8,
                "genetic": 1.0,
                "surrogate_bo": 0.9,
                "gcg": 0.8,
                "precision": 1.0,
                "sweep": 1.0,
                "boosters": 1.0,
                "beam": 1.0,
                "single_word": 1.0,
                "extend": 1.2,
                "grow": 1.3,
                "drift": 1.3,
                "probe_climb": 1.3,
                "scenario_join": 1.3,
            }
        )
        if self.objective.shortest:
            short_leader = self.leader is not None and self.leader.units <= 2
            priors.update({"compress": 1.6, "single_word": 2.5 if short_leader else 1.2, "boosters": 0.5})
            if self.one_word_race:
                priors["single_word"] = 4.0
        if self.long:
            priors.update({s.name: 1.3 for s in chain_arms})
            priors.update({"chain_ablate": 1.2, "chain_compact": 2.0, "compress": 1.6})
            self.ctx.chains = ChainLibrary(self.choices, self.length_cap, LONG_TARGET, self.ctx.rng)
        self.scheduler = Scheduler(list(self.strategies), priors=priors)
        self.paused = asyncio.Event()
        self.paused.set()
        self.stopping = False
        self.commands: asyncio.Queue = asyncio.Queue()
        self.rounds = 0
        self.stall = 0
        self.last_trained = 0
        self.training = False
        self.train_task: asyncio.Future | None = None
        self.prior_task: asyncio.Future | None = None
        self.embedder: Embedder | None = embedder
        self.vaulted: set[str] = set()
        self.status = "idle"

    # Events.

    def emit(self, kind: str, **data) -> None:
        self.on_event(Event(kind, data))

    # Scoring.

    async def evaluate(self, items: list[tuple[str, str, str, str]], n: int = 1) -> list[Candidate]:
        await self.paused.wait()
        if self.stopping:
            raise BudgetExhausted()
        clean: dict[str, tuple[str, str, str]] = {}
        for text, origin, parent, archetype in items:
            try:
                phrase = check_phrase(normalize(text), self.choices, self.length_cap)
            except RuleError:
                continue
            if self.one_word_race and origin != "manual" and self.objective.units(phrase) != 1:
                continue
            if phrase not in self.archive.items:
                words = {w.casefold() for w in phrase.split()}
                if words & self.ctx.banned or phrase_banned(phrase):
                    continue
                if self.ctx.pinned and not {p.casefold() for p in self.ctx.pinned} <= words:
                    continue
                if not self.guard.allowed(phrase):
                    continue
            clean.setdefault(phrase, (origin, parent, archetype))
        if not clean:
            return []
        remaining = self.budget - self.oracle.calls
        if self.in_command:
            remaining = max(remaining, len(clean) * n)
        if remaining <= 0:
            raise BudgetExhausted()
        phrases = list(clean)
        fresh = [p for p in phrases if p not in self.archive.items]
        if len(fresh) > remaining:
            keep = set(fresh[:remaining])
            phrases = [p for p in phrases if p in self.archive.items or p in keep]
        scores = await self.oracle.score(phrases, n)
        out = []
        persist = []
        dropped = 0
        for phrase in phrases:
            if not scores[phrase].n:
                dropped += 1  # the oracle gave up on every try (network or 5xx); no samples, mean is NaN
                continue
            origin, parent, archetype = clean[phrase]
            before = phrase in self.archive.items
            cand, improved = self.archive.add(phrase, scores[phrase], origin, parent, archetype)
            out.append(cand)
            if not before:
                persist.append((self.oracle.qkey, phrase, origin, parent, time.time(), cand.archetype))
                self.round_new.append(cand)
            if improved:
                self.origin_wins[origin] = self.origin_wins.get(origin, 0) + 1
                self.emit("best", phrase=phrase, p=cand.p, units=cand.units, n=cand.score.n, origin=origin)
        if dropped:
            self.emit("log", message=f"oracle failed on {dropped} phrase(s) after retries; skipped")
        if persist:
            self.db.executemany(
                "INSERT OR IGNORE INTO lab_candidates (qkey, state, origin, parent, created, archetype) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                persist,
            )
        self.emit(
            "scored",
            count=len(phrases),
            calls=self.oracle.calls,
            strategy=self.current_strategy,
            items=[(c.phrase, c.p, c.origin, c.score.n) for c in out],
        )
        return out

    # Setup.

    async def start(self) -> None:
        self.oracle = Oracle(self.db, self.question["jev_request"], target=self.target)
        if self.one_word_race:
            self.emit(
                "log",
                message="[shortest] leader is already 1 word; only a one-word line can win, so multi-word tries are skipped",
            )
        self.oracle.on_error = lambda message: self.emit("log", message=message)
        if self.ctx.chains is not None:
            self.ctx.chains.probe = Probe(self.oracle.qkey)
        if self.use_llm:
            try:
                self.llm = LLM()
                self.llm.on_error = lambda message: self.emit("log", message=message)
                if self.yes_no:
                    self.proxy = ProxyScreen(self.llm, self.question, self.objective.goal)
            except LLMError as error:
                self.emit("log", message=f"LLM disabled: {error}")
        self.memory = lab_memory.load(self.db, self.oracle.qkey, self.memory_board)
        self.restore_memory()
        await self.seed_archive()
        self.maybe_train(force=True)
        # Restored history is shared across boards, so on Shortest yes only this board's own runs count.
        tried_before = self.memory.runs > 0 or (self.restored >= 500 and not self.objective.shortest)
        self.fresh = tried_before and not self.winning()
        if self.plateau and self.winning():
            self.plateau = False
            self.scheduler.boost = {}
            self.emit("log", message="[plateau] our line already wins; ordinary search")
        if self.fresh:
            self.scheduler.boost = {"llm_gen": 1.6, "surrogate_bo": 1.2}
            if self.plateau:
                self.scheduler.boost = {"extend": 1.4, "sweep": 1.3, "drift": 1.3}
            peak = self.archive.top_by_board(1)[0]
            self.emit(
                "log",
                message=(
                    f"fresh start: run {self.memory.runs + 1}, earlier work peaked at {peak.p:.3f}/{peak.units}w "
                    f"without a win; steering away from {len(self.ctx.exhausted_lines())} exhausted lines "
                    "and toward the least-tried tactics"
                ),
            )
            start = min(self.memory.level, self.max_level) if self.escalate and not self.long else 0
            if self.plateau and self.escalate:
                start = max(1, start)
            if start:
                why = f"last run stalled at L{self.memory.level}; resuming there"
                if self.plateau and self.memory.level < start:
                    why = "plateau: straight to precision climbing"
                self.set_level(start, why)
        if self.llm is not None:
            for strategy in self.strategies.values():
                if isinstance(strategy, LLMGenerate) and strategy.unlocked(self.ctx):
                    strategy.prefetch(self.ctx)

    def restore_memory(self) -> None:
        mem = self.memory
        if not mem.runs:
            return
        self.scheduler.restore(mem.arms)
        self.strategies["compress"].done |= set(mem.compressed)
        self.ctx.synonyms.update(mem.synonyms)
        self.strategies["grow"].restore(mem.grow, mem.grow_gains)
        self.strategies["drift"].restore(mem.drift)
        self.strategies["probe_climb"].restore(mem.probe)

    def remaining(self) -> int:
        return max(0, self.budget - (self.oracle.calls if self.oracle else 0))

    def keeps_searching(self) -> bool:
        """A live round keeps going after the call budget. The stall ladder still has to be able to climb."""
        return self.live and self.idle_when_done

    def refill_search(self) -> bool:
        """Give a live search another budget window. False when this run is supposed to stop."""
        if not self.keeps_searching() or not self.oracle or self.oracle.calls < self.budget:
            return False
        self.budget += self.budget_span
        self.emit(
            "log",
            message=(
                f"call budget spent; continuing with {self.budget_span:,} more "
                f"(stall {self.stall}/{self.patience()}, L{self.level})"
            ),
        )
        return True

    def patience(self) -> int:
        """Flat rounds before escalating. Mechanical levels do far more per round, so they get less."""
        base = self.max_stall or 12
        return base if self.level == 0 else max(3, base // 2)

    def set_level(self, level: int, reason: str) -> None:
        self.level = level
        self.peak_level = max(self.peak_level, level)
        self.stall = 0
        name, lead, boosts = LEVELS[level]
        self.scheduler.level_boost = dict(boosts)
        if lead:
            self.scheduler.level_boost[lead] = 4.0
        if name == "reframe" and not self.winning():
            for cand in self.archive.top_by_board(10):
                if all(row[0] != cand.phrase for row in self.memory.exhausted):
                    self.memory.exhausted.append([cand.phrase, round(cand.p, 3), cand.units])
            for cand in self.archive.items.values():
                cand.session = False
            self.fresh = True
        self.emit("level", level=level, name=name, reason=reason)
        self.emit("log", message=f"[ladder] L{level} {name}: {reason}")
        joined = [
            s
            for s in self.strategies.values()
            if isinstance(s, LLMGenerate) and s.unlocked(self.ctx) and s.pending is None
        ]
        if joined and self.llm is not None:
            for strategy in joined:
                strategy.prefetch(self.ctx)
            self.emit(
                "log",
                message="[ladder] models join: "
                + "; ".join(f"{s.name} ({', '.join(m.split('/')[-1] for m in s.models)})" for s in joined),
            )

    @property
    def one_word_race(self) -> bool:
        """Shortest yes whose leader is already one word. Only another one-word line can place ahead."""
        return self.objective.shortest and self.leader is not None and self.leader.units == 1

    def winning(self) -> bool:
        if not len(self.archive):
            return False
        # The board's first line may round to the ceiling on a few lucky rolls; a confirmed win can sit below it.
        return any(self.objective.wins(c.score, c.units, self.leader) for c in self.archive.top_by_board(50))

    def unbeatable(self) -> Candidate | None:
        """A one-word line whose every roll rounds to 1.00 tops either board, so no further search can beat it."""
        for cand in self.archive.top_by_board(5):
            samples = self.objective.goal_samples(cand.score)
            if cand.units == 1 and len(samples) >= CEILING_MIN_N and all(site_round(s) >= 1.0 for s in samples):
                return cand
        return None

    # Long-chain regime.

    def roots(self) -> list[Candidate]:
        """Lines worth compacting: every roll at the ceiling, or a confirmed win. Cached per oracle call count."""
        calls = self.oracle.calls if self.oracle else 0
        if self._roots[0] != calls:
            obj = self.objective
            self._roots = (
                calls,
                [
                    c
                    for c in self.archive.items.values()
                    if obj.holds_ceiling(c.score)
                    or (c.score.n >= CEILING_MIN_N and obj.wins(c.score, c.units, self.leader))
                ],
            )
        return self._roots[1]

    def phase(self) -> str:
        return "compact" if self.roots() else "build"

    def word_stage(self) -> bool:
        return any(c.units <= WORD_STAGE for c in self.roots())

    def allowed(self, name: str) -> bool:
        if self.one_word_race and name == "boosters":
            return False
        # Edits of a longer line stay longer, and a longer line cannot win this race.
        if self.one_word_race and name in {"precision", "sweep", "local_edit", "genetic", "gcg", "surrogate_bo"}:
            if not any(c.units == 1 for c in self.archive.items.values()):
                return False
        if name in ("extend", "grow", "drift", "probe_climb", "scenario_join"):
            return self.plateau
        if name == "beam" and self.plateau:
            return False
        if not self.long:
            return True
        if name in SHORT_ONLY_ARMS:
            return False
        if name in WORD_ARMS:
            return self.word_stage()
        return True

    def frugal(self) -> bool:
        """Long-chain and plateau runs keep the expensive writer to a few stall batches per question."""
        return self.long or self.plateau

    def expensive_ok(self) -> bool:
        return self.frugal() and self.stall >= 4 and self.expensive_used < GEMINI_CALLS

    def long_lines(self) -> list[Candidate]:
        return [c for c in self.archive.items.values() if c.units > WORD_STAGE]

    def long_mark(self) -> tuple:
        """Build: the top rounded level among long chains, then the best probe on that level. Compact: the best
        root's rounded score, then fewer words."""
        roots = self.roots()
        if roots:
            best = max(roots, key=lambda c: (site_round(c.p), -c.units))
            return (1, site_round(best.p), -best.units)
        lines = self.long_lines()
        level = max((site_round(c.p) for c in lines), default=0.0)
        probe = self.ctx.chains.probe if self.ctx.chains is not None else None
        values = [probe.value(c.phrase) for c in lines if site_round(c.p) >= level - 1e-9] if probe else []
        return (0, level, max((v for v in values if v is not None), default=0.0))

    @staticmethod
    def long_gain(before: tuple, after: tuple, probe_noise: float = 0.015) -> float:
        if after <= before:
            return 0.0
        if after[0] > before[0]:
            return 1.0
        if after[1] > before[1] + 1e-9:
            return logit(after[1]) - logit(before[1]) if before[1] > 0 else 1.0
        if after[0] == 1:
            return 0.05 * (after[2] - before[2])
        if after[2] > before[2] + probe_noise:
            return 2.0 * (logit(after[2], 0.005) - logit(max(before[2], 0.005), 0.005))
        return 0.0

    async def probe_round(self) -> None:
        """Build phase: calibrate the probe once three long chains share the top level, then measure every new
        long chain on that level and re-measure the best few."""
        lib = self.ctx.chains
        if lib is None or lib.probe is None or self.roots():
            return
        probe = lib.probe
        lines = self.long_lines()
        if not lines:
            return
        level = max(site_round(c.p) for c in lines)
        top = sorted((c for c in lines if site_round(c.p) >= level - 1e-9), key=lambda c: (-c.score.n, -c.p))
        if not probe.ready:
            if probe.tried or len(top) < 3:
                return
            if not await probe.calibrate(self.ctx, [c.phrase for c in top[:3]]):
                return
            todo = [c.phrase for c in top[:60]]
        else:
            todo = [c.phrase for c in self.round_new if c.units > WORD_STAGE and site_round(c.p) >= level - 1e-9]
        await probe.measure(self.ctx, [p for p in todo if probe.value(p) is None])

        def leaders() -> list[str]:
            return sorted((c.phrase for c in top if probe.value(c.phrase) is not None), key=lambda p: -probe.value(p))[
                :PROBE_TOP
            ]

        await probe.measure(self.ctx, [p for p in leaders() if probe.samples.get(p, 0) < PROBE_TOP], n=PROBE_TOP)
        if probe.saturated(leaders()):
            ranked = sorted((c.phrase for c in top if probe.value(c.phrase) is not None), key=lambda p: -probe.value(p))
            if await probe.strengthen(self.ctx, leaders(), ranked[:60]):
                await probe.measure(self.ctx, leaders(), n=PROBE_TOP)
        # The probe's favourites get real samples too: one roll cannot say how often a chain reaches the ceiling.
        shallow = [self.archive.items[p] for p in leaders() if self.archive.items[p].score.n < CEILING_MIN_N]
        if shallow:
            await self.evaluate([(c.phrase, "resample", c.parent, c.archetype) for c in shallow], n=CEILING_MIN_N)
        best = max(leaders(), key=lambda p: probe.value(p), default=None)
        if best is not None:
            cand = self.archive.items[best]
            self.emit(
                "log",
                message=f"[probe] level {level:.2f}: best {probe.value(best):.3f} "
                f"({cand.units}w, {len(probe.values)} measured)",
            )

    def remember(self) -> None:
        """Fold this run into the question's memory."""
        if not self.oracle or not self.rounds:
            return
        mem = self.memory
        won = self.winning()
        best = self.archive.top_by_board(1)[0] if len(self.archive) else None
        mem.runs += 1
        mem.wins += int(won)
        mem.level = 0 if won else self.peak_level
        if best and self.objective.key(best.p, best.units) >= self.objective.key(mem.best_p, mem.best_units):
            mem.best_p, mem.best_units = best.p, best.units
        mem.styles = self.archive.style_stats()
        mem.arms = self.scheduler.state()
        mem.compressed = sorted(self.strategies["compress"].done)[-500:]
        mem.synonyms = dict(list(self.ctx.synonyms.items())[-400:])
        if self.plateau:
            grow = self.strategies["grow"]
            mem.grow = grow.state()
            mem.grow_gains = dict(sorted(grow.gains.items(), key=lambda kv: -kv[1][1])[:3000])
            mem.drift = self.strategies["drift"].line or ""
            mem.probe = self.strategies["probe_climb"].state()
        if not won:
            seen = {row[0] for row in mem.exhausted}
            for cand in self.archive.top_by_board(10):
                if cand.phrase not in seen:
                    mem.exhausted.append([cand.phrase, round(cand.p, 3), cand.units])
            mem.exhausted = sorted(mem.exhausted, key=lambda r: -r[1])[:30]
        mem.history.append(
            {
                "at": time.time(),
                "calls": self.oracle.calls,
                "rounds": self.rounds,
                "best_p": round(best.p, 3) if best else 0.0,
                "units": best.units if best else 0,
                "win": won,
                "fresh": self.fresh,
                "reason": self.end_reason,
            }
        )
        mem.history = mem.history[-20:]
        lab_memory.save(self.db, self.oracle.qkey, self.slug, mem, self.memory_board)

    def enter_plateau(self, reason: str) -> None:
        lead = self.leader
        self.plateau = True
        self.length_cap = self.grow_cap = min(max(self.length_cap, lead.units - 1), max(MAX_WORDS, 300))
        self.scheduler.boost = {"extend": 1.4, "sweep": 1.3, "drift": 1.3}
        self.emit(
            "log",
            message=(
                f"[plateau] {reason}: the leader holds {lead.probability:.2f} with {lead.units} words. Lines that tie "
                f"on the rounded score are ranked by a probe (an opposing claim appended) and climbed on it, and "
                f"lines may grow to {self.length_cap} words."
            ),
        )

    async def seed_archive(self) -> None:
        past = history(self.db, self.question["jev_request"], self.target)
        if self.plateau_candidate and (self.memory.runs > 0 or len(past) >= 500):
            self.enter_plateau(f"{len(past)} lines and {self.memory.runs} earlier runs on this board")
        origins = {
            r["state"]: dict(r)
            for r in self.db.all(
                "SELECT state, origin, parent, archetype FROM lab_candidates WHERE qkey = ?", (self.oracle.qkey,)
            )
        }
        for state, samples in past.items():
            try:
                phrase = check_phrase(state, self.choices, self.length_cap)
            except RuleError:
                continue
            meta = origins.get(state) or {}
            self.archive.add(
                phrase,
                Score(phrase, samples),
                meta.get("origin") or "history",
                meta.get("parent") or "",
                meta.get("archetype") or "",
                session=False,
            )
        self.restored = len(self.archive)
        site = self.db.all("SELECT DISTINCT state FROM site_scores WHERE slug = ?", (self.slug,))
        seeds = [(r["state"], "site", "", "other") for r in site]
        title = self.question["title"]
        seeds.append((title, "seed", "", "direct"))
        if self.yes_no:
            seeds.append((self.objective.goal, "seed", "", "direct"))
        else:
            seeds += [
                (c.get("description") or "", "seed", "", "direct")
                for c in self.question["raw"].get("choices") or []
                if isinstance(c, dict) and (not self.target or c.get("option") == self.target)
            ]
        if self.objective.shortest:
            seeds += self.prefix_seeds()
        triaged = self.triage_items()
        seeds += [(i.phrase, f"triage:{i.source}", "", "other") for i in triaged]
        have = {phrase for phrase, *_rest in seeds}
        for row in self.db.all("SELECT state FROM oracle_refusals WHERE qkey = ?", (self.oracle.qkey,)):
            phrase = row["state"]
            if phrase not in past and phrase not in have:
                seeds.append((phrase, "refused", "", ""))
                have.add(phrase)
        self.emit(
            "log",
            message=f"archive restored {len(self.archive)} phrases; scoring {len(seeds)} site/seed lines"
            + (f" ({len(triaged)} existing lines by triage)" if triaged else ""),
        )
        await self.evaluate(seeds)
        for item in triaged:
            cand = self.archive.items.get(item.phrase)
            if cand is not None and item.kev_entry:
                triage.record(
                    self.slug, self.objective, self.leader, item, cand.score, self.question["title"], self.target
                )
        self.emit("status", status="seeded")

    def triage_items(self) -> list["triage.TriageItem"]:
        """Existing lines worth scoring before any are generated (mirror editions: Jev's best and estimates)."""
        if self.triage_k <= 0:
            return []
        jev = triage_jev_db()
        try:
            return triage.rank_candidates(
                self.db,
                self.slug,
                self.objective,
                self.leader,
                self.triage_k,
                self.target,
                jev,
                set(self.archive.items),
                self.length_cap,
            )
        except Exception as error:
            self.emit("log", message=f"triage failed: {error!r}")
            return []
        finally:
            if jev is not None:
                jev.conn.close()

    def prefix_seeds(self, k: int = 30) -> list[tuple[str, str, str, str]]:
        """Shortest yes: every prefix of our best Highest lines. The site scores each prefix while a chain is
        built, so a strong line often already tips to yes after its first few words."""
        highest = Objective(self.objective.goal, self.objective.kind)
        tops = sorted(self.archive.items.values(), key=lambda c: (highest.key(c.p, c.units), c.p), reverse=True)[:k]
        out: dict[str, tuple[str, str, str, str]] = {}
        for cand in tops:
            words = cand.words
            if self.one_word_race:
                # Each word of a strong line can win on its own; a longer prefix cannot.
                pieces = words
            else:
                pieces = [" ".join(words[:n]) for n in range(1, len(words))]
            for piece in pieces:
                if piece and piece not in self.archive.items:
                    out.setdefault(piece, (piece, "prefix", cand.phrase, cand.archetype))
        return list(out.values())

    # Surrogates.

    def surrogate_data(self) -> dict[str, float]:
        data = {c.phrase: c.p for c in self.archive.items.values()}
        if self.target:
            return data  # site scores are Jev's top-answer probability, not P(target)
        for r in self.db.all("SELECT state, probability FROM site_scores WHERE slug = ?", (self.slug,)):
            data.setdefault(r["state"], float(r["probability"]))
        return data

    def maybe_train(self, force: bool = False) -> None:
        """Retrain in the background once the archive has grown enough; strategies keep using the old model."""
        size = len(self.archive)
        if self.training or (
            not force and size < max(60, int(self.last_trained * 1.25)) and size - self.last_trained < 300
        ):
            return
        self.training = True
        self.train_task = asyncio.ensure_future(self._train(size))

    async def _train(self, size: int) -> None:
        try:
            if self.embedder is None:
                self.embedder = await asyncio.to_thread(Embedder)
            if self.ctx.phrase_surrogate is None:
                self.ctx.phrase_surrogate = PhraseSurrogate(
                    self.embedder, None, (self.question["title"], self.objective.goal)
                )
                self.ctx.word_surrogate = WordSurrogate(self.embedder)
                if self.long:
                    self.ctx.phrase_surrogate.length_scale = float(self.long_target)
                    self.ctx.word_surrogate.length_scale = float(self.long_target)
                if self.yes_no:
                    self.prior_task = asyncio.ensure_future(self._attach_prior())
            data = self.surrogate_data()
            rho_p = await asyncio.to_thread(self.ctx.phrase_surrogate.fit, data, self.rounds)
            rho_w = await asyncio.to_thread(self.ctx.word_surrogate.fit, data, self.rounds)
            self.last_trained = size
            self.emit("surrogate", phrase_rho=rho_p, word_rho=rho_w, n=len(data), embedder=self.embedder.kind)
        except Exception as error:
            self.emit("log", message=f"surrogate training failed: {error!r}")
        finally:
            self.training = False

    async def _attach_prior(self) -> None:
        """Load (retraining on new phrases if due) the cross-question prior without holding up the search."""
        try:
            prior = await asyncio.to_thread(global_model.ensure, self.embedder, lambda m: self.emit("log", message=m))
        except Exception as error:
            self.emit("log", message=f"cross-question prior unavailable: {error!r}")
            return
        if prior is not None and self.ctx.phrase_surrogate is not None:
            self.ctx.phrase_surrogate.prior = prior
            self.emit(
                "log", message=f"cross-question prior ready (held-out rho {prior.rho:.2f}, {prior.trained_on} phrases)"
            )

    # Planner.

    async def plan(self) -> None:
        if self.llm is None or len(self.archive) < 5:
            return
        self.plans += 1
        stats = {
            "calls": self.oracle.calls,
            "size": len(self.archive),
            "arms": {
                k: {"share": v, "rate": self.scheduler.arms[k].rate()} for k, v in self.scheduler.shares().items()
            },
            "archetypes": self.ctx.archetype_bests(),
        }
        try:
            data = await self.llm.json(
                PLAN_MODEL,
                prompts.plan_system(self.ctx),
                prompts.plan_user(self.ctx, stats),
                temperature=0.6,
                max_tokens=3000,
            )
        except (LLMError, Exception) as error:
            self.emit("log", message=f"[planner] {error}")
            return
        if not isinstance(data, dict):
            return
        self.ctx.directive = str(data.get("directive") or "")[:400]
        self.ctx.focus = [f for f in data.get("focus") or [] if f in ARCHETYPES]
        protected = {p.casefold() for p in self.ctx.pinned} | set(FUNCTION_WORDS)
        for cand in self.archive.top_by_board(5):
            protected |= {w.casefold() for w in cand.words}
        applied = []
        for word in (data.get("ban") or [])[:8]:
            if isinstance(word, str) and word.casefold() not in protected:
                self.ctx.banned.add(word.casefold())
                applied.append(word)
        data["ban"] = applied
        mode = str(data.get("mode") or "")
        self.scheduler.boost = {
            "explore": {"llm_gen": 2.0, "surrogate_bo": 1.3},
            "exploit": {"local_edit": 1.6, "genetic": 1.6, "gcg": 1.4},
            "compress": {"compress": 3.0},
        }.get(mode, {})
        self.emit(
            "plan",
            directive=self.ctx.directive,
            focus=self.ctx.focus,
            mode=mode,
            ban=list(data.get("ban") or []),
            pin=list(data.get("pin") or []),
        )
        seeds = [(str(p), "planner", "", "other") for p in data.get("seed_phrases") or []]
        if seeds:
            await self.evaluate(seeds)

    # Promotion and vault.

    async def promote(self) -> None:
        """Re-score the leaders so a lucky single sample does not stay on top."""
        need = [c for c in self.archive.top(5) + self.archive.top_by_board(5) if c.score.n < 5]
        if self.long:
            # A long chain that rolled the ceiling once must be confirmed before it can root compaction.
            need += [c for c in self.round_new if c.score.n < 5 and self.objective.at_ceiling(c.p)][:10]
        need = list({c.phrase: c for c in need}.values())
        if need:
            await self.evaluate([(c.phrase, "resample", c.parent, c.archetype) for c in need], n=5)
        unsure = [
            c
            for c in self.archive.top_by_board(5)
            if c.score.n < 8
            and self.objective.at_ceiling(c.p)
            and not self.objective.wins(c.score, c.units, self.leader)
        ]
        if unsure:
            await self.evaluate([(c.phrase, "resample", c.parent, c.archetype) for c in unsure], n=8)
        for cand in self.archive.top_by_board(5):
            if self.worth_vaulting(cand):
                self.save(cand, auto=True)

    def worth_vaulting(self, cand: Candidate) -> bool:
        """A confirmed line that beats the leader, or a live line that clears the finish line for next time."""
        if cand.score.n < 5 or cand.phrase in self.vaulted or phrase_banned(cand.phrase):
            return False
        if self.objective.wins(cand.score, cand.units, self.leader):
            return True
        floor = self.question.get("yes_threshold")
        if not (self.live and self.question.get("revision_id") and floor is not None):
            return False
        try:
            return self.objective.p_lcb(cand.score) >= float(floor) - 1e-9
        except (TypeError, ValueError):
            return False

    def save(self, cand: Candidate, auto: bool = False) -> None:
        objective = self.objective
        lcb = objective.p_lcb(cand.score)
        beats = objective.wins(cand.score, cand.units, self.leader)
        extra = {}
        gamble = objective.gamble(cand.score, cand.units, self.leader)
        if gamble:
            extra = {
                "gamble": True,
                "p_reach": round(max(objective.goal_samples(cand.score)), 4),
                "hits": f"{objective.hits(cand.score)}/{cand.score.n}",
            }
        if not self.live or self.question.get("revision_id"):
            vault.save(
                self.slug,
                self.mode,
                cand.phrase,
                p_mean=cand.p,
                p_lcb=lcb,
                spread=cand.score.spread,
                n=cand.score.n,
                units=cand.units,
                leader=self.leader,
                beats=beats,
                title=self.question["title"],
                origin=cand.origin,
                note="auto" if auto else "manual",
                board=self.board,
                extra=extra,
                target=self.target,
            )
        self.vaulted.add(cand.phrase)
        self.emit(
            "vault",
            phrase=cand.phrase,
            p=cand.p,
            lcb=lcb,
            units=cand.units,
            beats=beats,
            auto=auto,
            gamble=extra.get("hits", ""),
        )
        if beats and self.win_extra and self.won_at is None:
            self.won_at = self.oracle.calls
            cap = self.oracle.calls + self.win_extra
            if cap < self.budget:
                self.budget = cap
                self.emit(
                    "log",
                    message=f"[win mode] took the lead at {self.won_at:,} calls; "
                    f"{self.win_extra:,} more to improve it, stopping at {cap:,}",
                )

    # Commands from the TUI (run between rounds).

    def submit(self, coro_factory: Callable[[], "asyncio.Future"]) -> None:
        self.commands.put_nowait(coro_factory)

    async def inject(self, text: str, n: int = 3) -> list[Candidate]:
        return await self.evaluate([(text, "manual", "", "other")], n=n)

    def ban(self, word: str) -> None:
        self.ctx.banned.add(word.casefold())
        self.ctx.pinned.discard(word)

    def pin(self, word: str) -> None:
        self.ctx.pinned.add(word)
        self.ctx.banned.discard(word.casefold())

    def unpin(self, word: str) -> None:
        self.ctx.pinned.discard(word)
        self.ctx.banned.discard(word.casefold())

    def force(self, name: str | None) -> None:
        self.scheduler.forced = name if name in self.strategies else None

    async def compress_phrase(self, phrase: str) -> None:
        cand = self.archive.items.get(phrase)
        if cand is not None and cand.units > 1:
            compress: Compress = self.strategies["compress"]
            compress.stop_at = self.oracle.calls + 6000  # asked for by hand: room for a full beam on a long line
            await compress.compress(self.ctx, cand)

    async def resample(self, phrase: str, n: int = 10) -> None:
        cand = self.archive.items.get(phrase)
        if cand is not None:
            await self.evaluate([(phrase, "resample", cand.parent, cand.archetype)], n=n)

    def pause(self) -> None:
        self.paused.clear()
        self.emit("status", status="paused")

    def resume(self) -> None:
        self.paused.set()
        self.emit("status", status="running")

    def stop(self) -> None:
        self.stopping = True
        self.paused.set()

    # Main loop.

    async def run_commands(self) -> None:
        """Manual commands from the TUI may exceed the budget; the user asked for them."""
        while not self.commands.empty():
            factory = self.commands.get_nowait()
            self.in_command = True
            try:
                await factory()
            except Exception as error:
                self.emit("log", message=f"command failed: {error!r}")
            finally:
                self.in_command = False

    def budget_reason(self) -> str:
        if self.won_at is not None and self.win_extra and self.budget == self.won_at + self.win_extra:
            return f"win mode: led at {self.won_at:,} calls, stopped {self.win_extra:,} later"
        return "budget"

    async def run(self) -> None:
        await self.start()
        self.status = "running"
        self.emit("status", status="running")
        try:
            idle_noted = False
            solved: Candidate | None = None
            while not self.stopping:
                await self.run_commands()
                if solved is None and (solved := self.unbeatable()) is not None:
                    self.emit(
                        "log",
                        message=f"[done] '{solved.phrase}' holds 1.00 on all {solved.score.n} rolls "
                        "with one word; nothing can beat it",
                    )
                    if solved.phrase not in self.vaulted and self.objective.wins(solved.score, 1, self.leader):
                        self.save(solved, auto=True)
                if solved is not None:
                    if not self.idle_when_done:
                        self.end_reason = "unbeatable: 1.00 with one word"
                        break
                    if not idle_noted:
                        self.emit("status", status="unbeatable")
                        idle_noted = True
                    await asyncio.sleep(0.5)
                    continue
                if self.oracle.calls >= self.budget:
                    if self.refill_search():
                        idle_noted = False
                    elif not self.idle_when_done:
                        self.end_reason = self.budget_reason()
                        break
                    else:
                        if not idle_noted:
                            self.emit("status", status="budget spent")
                            self.emit(
                                "log", message="budget spent; commands still work, `budget <n>` to keep searching"
                            )
                            idle_noted = True
                        await asyncio.sleep(0.5)
                        continue
                if idle_noted:
                    idle_noted = False
                    self.emit("status", status="running")
                await self.paused.wait()
                if self.stopping:
                    break
                available = {
                    name for name, s in self.strategies.items() if self.allowed(name) and s.available(self.ctx)
                }
                if not available:
                    self.emit("log", message="no strategy available; add a phrase or enable the LLM")
                    await asyncio.sleep(2)
                    continue
                name = self.scheduler.choose(available)
                if self.plateau and not self.scheduler.forced:
                    # A plateau walk pays off only when it lands, too late for the bandit, so it gets fixed
                    # shares. Scenario joins take one until they run dry and probe climbing the other while it
                    # has a claim; a fresh grown line only reached ~0.92 by 49 words on the hard boards, so grow
                    # is left to the bandit.
                    fixed = {1: "scenario_join", 2: "probe_climb"}.get(self.rounds % 3)
                    if fixed not in available:
                        fixed = {1: "drift", 2: "grow"}.get(self.rounds % 3)
                    if fixed in available:
                        name = fixed
                mark_before = self.long_mark() if self.long else None
                best_before = self.archive.best().fitness if len(self.archive) else float("-inf")
                leader_before = self.archive.best().phrase if len(self.archive) else ""
                elites_before = dict(self.archive.elites)
                calls_before = self.oracle.calls
                top_before = self.archive.top_by_board(5)
                self.round_new = []
                self.current_strategy = name
                self.emit("round", round=self.rounds + 1, strategy=name)
                try:
                    await self.strategies[name].run(self.ctx)
                    await self.promote()
                    if self.long:
                        await self.probe_round()
                except BudgetExhausted:
                    if self.stopping or not self.idle_when_done:
                        self.end_reason = self.end_reason or ("stopped" if self.stopping else self.budget_reason())
                        break
                except Exception as error:
                    self.emit("log", message=f"[{name}] failed: {error!r}")
                self.rounds += 1
                best_after = self.archive.best().fitness if len(self.archive) else best_before
                gain = best_after - best_before if best_before != float("-inf") else 0.0
                if self.long:
                    mark_after = self.long_mark()
                    gain = self.long_gain(mark_before, mark_after)
                    if mark_after[0] > mark_before[0]:
                        self.emit(
                            "log",
                            message=f"[chains] a line holds {self.objective.ceiling:.2f}; "
                            "compacting it while every roll stays there",
                        )
                new_elites = sum(1 for cell, p in self.archive.elites.items() if elites_before.get(cell) != p)
                novel = self.count_novel(top_before)
                if novel:
                    self.novel_by_strategy[name] = self.novel_by_strategy.get(name, 0) + novel
                    self.emit("log", message=f"[novelty] {novel} strong lines unlike the current best ({name})")
                self.scheduler.update(name, gain, self.oracle.calls - calls_before, new_elites, novel)
                if self.long:
                    progressed = gain > 1e-9
                else:
                    progressed = gain > 1e-9 and self.archive.best().phrase != leader_before
                self.stall = 0 if progressed else self.stall + 1
                if progressed and self.level > 0:
                    self.set_level(0, f"new best from {name} at L{self.level}; back to normal search")
                if self.scheduler.forced == "compress":
                    self.scheduler.forced = None
                self.maybe_train()
                if (
                    self.stall >= self.patience()
                    and self.rounds >= self.min_rounds
                    and self.long
                    and not self.word_stage()
                ):
                    flat = f"{self.stall} rounds without a better chain"
                    if self.max_stall:
                        self.end_reason = f"plateau in {self.phase()} ({flat})"
                        self.emit("log", message=f"moving on: {self.end_reason}")
                        break
                    self.stall = 0
                    self.scheduler.boost = {"chain": 1.5, "llm_gen": 1.3}
                    self.emit("log", message=f"[chains] {flat}; favouring new chains and writers")
                elif self.stall >= self.patience() and self.rounds >= self.min_rounds:
                    flat = f"{self.stall} rounds without a better line"
                    room = self.remaining() > 0 or self.keeps_searching()
                    if self.escalate and room and self.level < self.max_level:
                        self.refill_search()
                        self.set_level(self.level + 1, f"plateau at L{self.level} ({flat})")
                        continue
                    if self.escalate and room and not self.max_stall and self.max_level:
                        self.refill_search()
                        self.set_level(1, f"ladder exhausted ({flat}); cycling from L1 until the budget ends")
                        continue
                    if self.max_stall:
                        where = f" after L{self.level}" if self.escalate else ""
                        self.end_reason = f"plateau{where} ({flat})"
                        self.emit("log", message=f"moving on: {self.end_reason}")
                        break
                if self.frugal():
                    plan_due = self.stall == 4 and self.plans < LONG_PLAN_CALLS
                else:
                    plan_due = self.rounds % self.plan_every == 0 or self.stall == 4
                if plan_due and self.remaining() > 0:
                    try:
                        await self.plan()
                    except BudgetExhausted:
                        pass
        finally:
            if self.stopping and not self.end_reason:
                self.end_reason = "stopped"
            self.status = "stopped"
            self.emit("status", status="stopped")
            await self.close()

    @staticmethod
    def content_words(cand: Candidate) -> set[str]:
        return {w.casefold() for w in cand.words} - set(FUNCTION_WORDS)

    def count_novel(self, top_before: list[Candidate], near: float = 0.02, max_overlap: float = 0.34) -> int:
        """New lines within `near` of the best rounded score that share little vocabulary with the top 5."""
        if not top_before or not self.round_new:
            return 0
        if self.long and self.ctx.chains is not None:
            lib = self.ctx.chains
            if lib.built:
                lib.absorb_now(self.archive)
            # At the plateau nearly every chain is within a step of the best; only the top level counts.
            return min(lib.novel(self.archive, self.round_new, near=0.0), 3) + min(lib.strong(self.round_new), 5)
        floor = site_round(top_before[0].p) - near - 1e-9
        refs = [self.content_words(c) for c in top_before]
        count = 0
        for cand in self.round_new:
            if site_round(cand.p) < floor:
                continue
            words = self.content_words(cand)
            if not words:
                continue
            overlap = max((len(words & ref) / len(words | ref) for ref in refs if ref), default=0.0)
            if overlap < max_overlap:
                count += 1
        return count

    async def close(self) -> None:
        try:
            self.remember()
        except Exception as error:
            self.emit("log", message=f"could not save run memory: {error!r}")
        for strategy in self.strategies.values():
            if isinstance(strategy, LLMGenerate):
                strategy.cancel()
        if self.prior_task is not None and not self.prior_task.done():
            self.prior_task.cancel()  # a retrain already running finishes in its thread and saves
        if self.train_task is not None and not self.train_task.done():
            try:
                await asyncio.wait_for(self.train_task, 30)
            except (asyncio.TimeoutError, Exception):
                pass
        if self.oracle:
            await self.oracle.close()
        if self.llm:
            await self.llm.close()

    # Summary for headless runs.

    def summary(self, k: int = 10) -> list[str]:
        lines = []
        lead = f"{self.leader.probability:.2f}/{self.leader.units}w ({self.leader.name})" if self.leader else "none"
        goal = f"answer={self.target}" if self.target else f"goal={self.objective.goal}"
        lines.append(
            f"{self.question['title']} [{from_board(self.board).label}] {goal} leader={lead} "
            f"calls={self.oracle.calls if self.oracle else 0} archive={len(self.archive)}"
        )
        for cand in self.archive.top_by_board(k):
            win = "    "
            if self.objective.wins(cand.score, cand.units, self.leader):
                win = "BET " if self.objective.gamble(cand.score, cand.units, self.leader) else "WIN "
            lines.append(
                f"{win}{cand.p:.3f}±{cand.score.spread:.3f} n={cand.score.n} {cand.units:2}w "
                f"[{cand.origin}] {cand.phrase}"
            )
        return lines


__all__ = ["BudgetExhausted", "Engine", "Event", "goal_p", "site_round"]
