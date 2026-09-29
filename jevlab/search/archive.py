"""Every scored phrase for one question, plus a MAP-Elites grid for diversity."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..objective import Objective
from ..oracle import Score

ARCHETYPES = ("rename", "speaker", "scene", "eager", "direct", "wild", "other")
LENGTH_BUCKETS = (1, 3, 5, 8, 12, 17, 24, 33, 45, 60, 90, 130, 180, 240, 300)


def length_bucket(units: int) -> int:
    for index, top in enumerate(LENGTH_BUCKETS):
        if units <= top:
            return index
    return len(LENGTH_BUCKETS) - 1


@dataclass
class Candidate:
    phrase: str
    score: Score
    origin: str
    parent: str = ""
    archetype: str = "other"
    created: float = field(default_factory=time.time)
    p: float = 0.0
    fitness: float = 0.0
    units: int = 0
    session: bool = True  # found in this run, not restored from an earlier one

    @property
    def words(self) -> list[str]:
        return self.phrase.split()


class Archive:
    def __init__(self, objective: Objective):
        self.objective = objective
        self.items: dict[str, Candidate] = {}
        self.elites: dict[tuple[int, str], str] = {}
        self.ablation: dict[str, list[float]] = {}
        self.best_history: list[tuple[float, float]] = []

    def __contains__(self, phrase: str) -> bool:
        return phrase in self.items

    def __len__(self) -> int:
        return len(self.items)

    def add(self, phrase: str, score: Score, origin: str, parent: str = "", archetype: str = "",
            session: bool = True) -> tuple[Candidate, bool]:
        """Insert or refresh. Returns (candidate, new_best_fitness)."""
        previous_best = self.best().fitness if self.items else float("-inf")
        cand = self.items.get(phrase)
        if cand is None:
            cand = Candidate(phrase, score, origin, parent, archetype or "other", session=session)
            self.items[phrase] = cand
        else:
            cand.score = score
            if archetype and cand.archetype == "other":
                cand.archetype = archetype
        cand.units = self.objective.units(phrase)
        cand.p = self.objective.p(score)
        cand.fitness = self.objective.fitness(score, phrase)
        cell = (length_bucket(cand.units), cand.archetype)
        holder = self.items.get(self.elites.get(cell, ""))
        if holder is None or cand.fitness > holder.fitness:
            self.elites[cell] = phrase
        improved = cand.fitness > previous_best + 1e-9
        if improved or not self.best_history:
            self.best_history.append((time.time(), self.best().p))
        return cand, improved

    def best(self) -> Candidate:
        return max(self.items.values(), key=lambda c: c.fitness)

    def top(self, k: int = 10, min_n: int = 0) -> list[Candidate]:
        pool = [c for c in self.items.values() if c.score.n >= min_n]
        return sorted(pool, key=lambda c: c.fitness, reverse=True)[:k]

    def top_by_board(self, k: int = 10) -> list[Candidate]:
        """Board order (see Objective.score_key); mean breaks remaining ties."""
        return sorted(
            self.items.values(),
            key=lambda c: (self.objective.score_key(c.score, c.units), c.p),
            reverse=True,
        )[:k]

    def top_session(self, k: int = 10) -> list[Candidate]:
        """Best lines found in this run only."""
        pool = [c for c in self.items.values() if c.session]
        return sorted(pool, key=lambda c: c.fitness, reverse=True)[:k]

    def style_stats(self) -> dict[str, dict]:
        """Per archetype: best p reached and how many lines were tried."""
        out: dict[str, dict] = {}
        for cand in self.items.values():
            row = out.setdefault(cand.archetype, {"best": 0.0, "tries": 0})
            row["best"] = max(row["best"], cand.p)
            row["tries"] += 1
        return out

    def elite_list(self) -> list[Candidate]:
        return [self.items[p] for p in self.elites.values() if p in self.items]

    def vocabulary(self, k: int = 40) -> list[str]:
        """Words from the best lines, best first."""
        seen: dict[str, None] = {}
        for cand in self.top(k):
            for word in cand.words:
                seen.setdefault(word, None)
        return list(seen)
