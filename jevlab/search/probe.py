"""Pick one generator model for a question, then keep it until the search goes flat.

The route bake-off scored the same tactics with every model. The winning model changed
with the question, and five tactics (rename, scene, syllogism, direct, wild) were enough
to see which one. A fresh board stays with that model; a board whose tactics are already
covered asks again after a shorter flat stretch, because one model stops covering them.
"""

from __future__ import annotations

from dataclasses import dataclass

# One short batch, the same five tactics, for every model in the pool.
PROBE_TACTICS = ("rename", "scene", "syllogism", "direct", "wild")
PROBE_COUNT = 8
# A cheaper model this close to the best mean keeps the question. The pool is listed cheapest first.
WORTH_IT = 0.02
# Generator batches the lock must write, and flat search rounds, before another probe.
HOLD_FRESH = 8
STALL_FRESH = 6
HOLD_EXHAUSTED = 4
STALL_EXHAUSTED = 4


@dataclass
class Lock:
    model: str | None = None
    batches: int = 0


def mean_top(values: list[float], k: int = 3) -> float:
    """Mean of the best `k` scores. Fewer than `k` lines uses whatever came back."""
    if not values:
        return 0.0
    top = sorted(values, reverse=True)[:k]
    return sum(top) / len(top)


def choose(scores: dict[str, float], pool: list[str]) -> str | None:
    """Highest mean. Within ``WORTH_IT`` of it, the earlier pool entry wins. Nobody scoring loses."""
    if not pool:
        return None
    best = max((scores.get(model, 0.0) for model in pool), default=0.0)
    if best <= 0:
        return None
    for model in pool:
        if best - scores.get(model, 0.0) <= WORTH_IT:
            return model
    return None


def searched_out(tries: dict[str, int], tactics: list[str], bar: int) -> bool:
    """True once half the tactics already have `bar` scored lines."""
    if not tactics:
        return False
    covered = sum(1 for name in tactics if int(tries.get(name, 0)) >= bar)
    return covered * 2 >= len(tactics)


def due(lock: Lock, *, stall: int, exhausted: bool, cold: bool, force: bool) -> bool:
    """Ask again at the start, when the writer is cold, when a tier unlocks, or after a flat hold."""
    if lock.model is None or cold or force:
        return True
    hold = HOLD_EXHAUSTED if exhausted else HOLD_FRESH
    stall_bar = STALL_EXHAUSTED if exhausted else STALL_FRESH
    return lock.batches >= hold and stall >= stall_bar
