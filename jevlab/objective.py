"""What "better" means for a board, and whether we beat the live leader.

Oracle values are the model's P(yes) (or the choice/score equivalent). The site
shows P(goal): 1 - P(yes) when the goal is "no". Highest ranks by rounded
probability, then fewer units. Shortest yes ranks lines that reach the yes
threshold by fewer units, then rounded probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .modes import HIGH_SCORES, SHORTEST_YES
from .oracle import Score

# Fallback when a question does not name a yes line. Most boards publish 0.5; we sit just over it.
SHORTEST_THRESHOLD = 0.51
SHORTEST_MAX_UNITS = 60
# Shortest yes keeps our best-ever roll, so a line shorter than the leader that clears the threshold on at
# least one of this many samples is worth publishing and re-rolling until it lands.
GAMBLE_MIN_N = 5
# The site returns yes/no probabilities between 0.01 and 0.99; engines raise this if they ever see higher.
CEILING = 0.99
CEILING_MIN_N = 5


def board_threshold(question: dict | None) -> float:
    """The yes line this board publishes. A missing or unusable value stays at 0.51."""
    if not question:
        return SHORTEST_THRESHOLD
    raw = question.get("yes_threshold")
    if raw is None:
        raw = question.get("yesThreshold")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return SHORTEST_THRESHOLD
    if not math.isfinite(value) or value <= 0 or value > 1:
        return SHORTEST_THRESHOLD
    return value


def objective_for(question: dict | None, *, unit: str = "word", board: str = HIGH_SCORES) -> Objective:
    """Objective for a question. Shortest yes qualifies at that question's yes line."""
    question = question or {}
    return Objective(
        question.get("goal") or "yes",
        question.get("kind") or "noul",
        unit=unit,
        board=board,
        threshold=board_threshold(question) if board == SHORTEST_YES else SHORTEST_THRESHOLD,
    )


def goal_p(value: float, goal: str | None, kind: str = "noul") -> float:
    if kind == "noul" and (goal or "yes") == "no":
        return 1.0 - value
    return value


def logit(p: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def site_round(p: float) -> float:
    return round(p + 1e-9, 2)


@dataclass(frozen=True)
class Leader:
    probability: float
    units: int
    name: str = ""
    ours: bool = False


@dataclass
class Objective:
    goal: str
    kind: str = "noul"
    unit: str = "word"
    par_units: int | None = None  # leader's length; shorter lines only matter for ties at or above it
    board: str = HIGH_SCORES
    threshold: float = SHORTEST_THRESHOLD
    long_units: int = 0  # long-chain regime: the build length; 0 is the ordinary search
    ceiling: float = CEILING

    @property
    def shortest(self) -> bool:
        return self.board == SHORTEST_YES

    @property
    def long(self) -> bool:
        return self.long_units > 0 and not self.shortest

    def at_ceiling(self, p: float) -> bool:
        return site_round(p) >= self.ceiling - 1e-9

    def holds_ceiling(self, score: Score, min_n: int = CEILING_MIN_N) -> bool:
        """Every roll rounds to the ceiling. The site rounds each roll, so one lower roll can lose the tie."""
        samples = self.goal_samples(score)
        return len(samples) >= min_n and all(self.at_ceiling(s) for s in samples)

    def holds(self, score: Score, level: float, min_n: int = 2) -> bool:
        """A cut keeps its root's standing: at the ceiling every roll must stay there, below it the mean must
        keep the rounded score."""
        if level >= self.ceiling - 1e-9:
            return self.holds_ceiling(score, min_n)
        return score.n >= min_n and site_round(self.p(score)) >= level - 1e-9

    def p(self, score: Score) -> float:
        return goal_p(score.mean, self.goal, self.kind)

    def p_lcb(self, score: Score, z: float = 1.0) -> float:
        if self.kind == "noul" and self.goal == "no":
            flipped = Score(score.state, [1 - s for s in score.samples])
            return flipped.lcb(z)
        return score.lcb(z)

    def units(self, phrase: str) -> int:
        if self.unit == "char":
            import unicodedata

            return len(unicodedata.normalize("NFC", phrase).strip())
        return len(phrase.split())

    def qualifies(self, p: float) -> bool:
        return p >= self.threshold - 1e-9

    def goal_samples(self, score: Score) -> list[float]:
        return [goal_p(s, self.goal, self.kind) for s in score.samples]

    def hits(self, score: Score) -> int:
        """Shortest yes: samples that cleared the threshold on their own."""
        return sum(1 for s in self.goal_samples(score) if self.qualifies(s))

    def reachable(self, score: Score) -> bool:
        """Shortest yes: at least one roll cleared the threshold, so re-rolling on the site can land it."""
        return self.shortest and self.hits(score) > 0

    def gamble(self, score: Score, units: int, leader: Leader | None) -> bool:
        """A Shortest-yes line that is shorter than the leader and landed on at least one of GAMBLE_MIN_N+
        samples, but whose lower bound does not qualify."""
        return (self.shortest and score.n >= GAMBLE_MIN_N and self.hits(score) > 0
                and not self.qualifies(self.p_lcb(score)) and (leader is None or units < leader.units))

    def win_p(self, score: Score, units: int, leader: Leader | None) -> float:
        """The probability a win is judged on: the lower bound, or for a Shortest-yes gamble its best roll."""
        if self.gamble(score, units, leader):
            return max(self.goal_samples(score))
        return self.p_lcb(score)

    def wins(self, score: Score, units: int, leader: Leader | None) -> bool:
        return self.beats(self.win_p(score, units, leader), units, leader)

    def entry_p(self, entry: dict, leader: Leader | None) -> float:
        """win_p for a saved vault entry, re-judged against the current leader."""
        if self.shortest and entry.get("gamble") and (leader is None or entry["units"] < leader.units):
            return float(entry.get("p_reach") or entry["p_lcb"])
        return float(entry["p_lcb"])

    def fitness(self, score: Score, phrase: str) -> float:
        """Search signal: logit keeps a gradient near 99%. Below the leader's length a higher rounded score
        wins regardless of length, so the per-unit penalty only applies from one under the leader's length.
        Shortest yes: lines that never cleared the threshold climb toward it (always <= 0). Lines that did
        get 0.2 per word saved, more than any tiebreak (< 0.2), so fewer words always rank first; at the
        same length a line whose mean qualifies beats one that only landed some rolls."""
        units = self.units(phrase)
        p = self.p(score)
        if self.shortest:
            gap = logit(p) - logit(self.threshold)
            base = 2.0 + 0.2 * max(SHORTEST_MAX_UNITS - units, 0)
            if self.qualifies(p):
                return base + 0.1 + 0.09 * min(max(gap / 4, 0.0), 1.0)
            if self.reachable(score):
                return base + 0.09 * self.hits(score) / max(score.n, 1)
            return min(gap, 0.0) - 0.01 * units
        if self.long:
            # Below the ceiling length is free; at the ceiling only fewer words can still win.
            if self.at_ceiling(p):
                return logit(self.ceiling) + 1.0 + 0.01 * max(self.long_units - units, 0)
            return logit(p) - 0.0002 * units
        if self.par_units is not None:
            units = max(0, units - self.par_units + 1)
        return logit(p) - 0.004 * units

    def key(self, p: float, units: int) -> tuple:
        """Board order. Highest: rounded probability, then fewer units. Shortest yes: qualifying lines first,
        then fewer units, then rounded probability."""
        if self.shortest:
            if not self.qualifies(p):
                return (0, 0, site_round(p))
            return (1, -units, site_round(p))
        return (site_round(p), -units)

    def score_key(self, score: Score, units: int) -> tuple:
        """key() for a scored line. Shortest yes: a line that landed any roll ranks with the qualifiers by
        length, just behind a same-length line whose mean qualifies."""
        p = self.p(score)
        if self.shortest:
            if self.qualifies(p):
                return (1, -units, 1, site_round(p))
            if self.reachable(score):
                return (1, -units, 0, site_round(p))
            return (0, 0, 0, site_round(p))
        return self.key(p, units)

    def leader_key(self, leader: Leader) -> tuple:
        """A board row already counted on the site, so on Shortest yes it qualifies whatever it shows."""
        if self.shortest:
            return (1, -leader.units, site_round(leader.probability))
        return self.key(leader.probability, leader.units)

    def beats(self, p: float, units: int, leader: Leader | None) -> bool:
        if leader is None:
            return self.qualifies(p) if self.shortest else p > 0
        return self.key(p, units) > self.leader_key(leader)


def target_rows(boards: dict, board: str, target: str) -> list[dict]:
    """Rows filed under one answer of a choice question (the site files a row under Jev's picked answer): the
    board's rows for it, plus on Highest its champion, the answer's best row."""
    rows = [r for r in boards.get(board) or [] if r.get("choice") == target]
    if board == HIGH_SCORES:
        rows += [r for r in boards.get("champions") or [] if r.get("choice") == target]
    return rows


def board_leader(rows: list[dict], me: str = "", include_ours: bool = False, unit_key: str = "wordCount",
                 board: str = HIGH_SCORES) -> Leader | None:
    """Top of a board. Our own row is skipped unless include_ours."""
    shortest = board == SHORTEST_YES
    best: Leader | None = None

    def rank(leader: Leader) -> tuple:
        if shortest:
            return (-leader.units, site_round(leader.probability))
        return (site_round(leader.probability), -leader.units)

    for row in rows or []:
        ours = bool(me) and row.get("userId") == me
        if ours and not include_ours:
            continue
        if row.get("probability") is None:
            continue
        cand = Leader(float(row["probability"]), int(row.get(unit_key) or 0), row.get("name") or "", ours)
        if best is None or rank(cand) > rank(best):
            best = cand
    return best
