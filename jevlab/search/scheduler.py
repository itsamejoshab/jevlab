"""Bandit over strategies: budget flows to whatever has been improving recently."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Arm:
    name: str
    prior: float = 1.0
    pulls: int = 0
    calls: int = 0
    reward: float = 0.0
    recent: list[float] = field(default_factory=list)

    def rate(self) -> float:
        """Discounted reward per 100 oracle calls over the recent window."""
        if not self.recent:
            return 0.0
        return sum(self.recent) / max(len(self.recent), 1)


def factor(boosts: dict[str, float], name: str) -> float:
    """A boost keyed `llm_gen` applies to every generator model arm (`llm:<model>`), `chain` to every
    long-chain recipe arm (`chain:<recipe>`)."""
    if name in boosts:
        return boosts[name]
    if name.startswith("llm:"):
        return boosts.get("llm_gen", 1.0)
    if name.startswith("chain:"):
        return boosts.get("chain", 1.0)
    return 1.0


class Scheduler:
    def __init__(self, names: list[str], priors: dict[str, float] | None = None, seed: int = 0):
        self.arms = {name: Arm(name, (priors or {}).get(name, 1.0)) for name in names}
        self.forced: str | None = None
        self.boost: dict[str, float] = {}  # planner mode
        self.level_boost: dict[str, float] = {}  # escalation level
        self.rng = random.Random(seed)
        self.total = 0

    def choose(self, available: set[str] | None = None) -> str:
        names = [n for n in self.arms if available is None or n in available]
        if self.forced and self.forced in names:
            return self.forced
        for name in names:
            if self.level_boost.get(name, 1.0) >= 4.0 and self.arms[name].pulls == 0:
                return name
        for name in names:
            if self.arms[name].pulls == 0:
                return name
        self.total += 1
        best, best_value = names[0], -math.inf
        for name in names:
            arm = self.arms[name]
            exploit = arm.rate()
            explore = math.sqrt(2 * math.log(self.total + 1) / arm.pulls)
            value = (exploit + 0.3 * explore) * arm.prior * factor(self.boost, name) * factor(self.level_boost, name)
            value *= 1 + 0.05 * self.rng.random()
            if value > best_value:
                best, best_value = name, value
        return best

    def update(self, name: str, gain: float, calls: int, new_elites: int, novel: int = 0) -> None:
        """gain: fitness improvement of the global best; new_elites: MAP-Elites cells filled or improved;
        novel: strong lines far from the current best (pays for exploration that has not won yet)."""
        arm = self.arms[name]
        arm.pulls += 1
        arm.calls += calls
        value = (max(gain, 0) * 10 + 0.2 * new_elites + 0.5 * min(novel, 10)) / max(calls / 100, 0.25)
        arm.reward += value
        arm.recent.append(value)
        del arm.recent[:-8]

    def state(self) -> dict[str, dict]:
        return {name: {"pulls": a.pulls, "reward": a.reward, "recent": a.recent}
                for name, a in self.arms.items()}

    def restore(self, state: dict[str, dict]) -> None:
        for name, saved in state.items():
            arm = self.arms.get(name)
            if arm is None:
                continue
            arm.pulls = int(saved.get("pulls") or 0)
            arm.reward = float(saved.get("reward") or 0.0)
            arm.recent = [float(v) for v in saved.get("recent") or []][-8:]
        self.total = sum(a.pulls for a in self.arms.values())

    def shares(self) -> dict[str, float]:
        total = sum(a.calls for a in self.arms.values()) or 1
        return {name: arm.calls / total for name, arm in self.arms.items()}
