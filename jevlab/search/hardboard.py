"""Hard Highest yes/no visits: deal the catalog, grow the close frames, then compress.

A hard board has a leader who is not us, showing at least 0.90. The visit does not climb the
ladder and does not start a line from nothing. Easy boards, Shortest yes, choice questions,
score questions, and the long-chain regime stay on the ladder they already use.

The schedule is the seam: a leader, the saved generator counts, the lines already scored,
and the configured model list decide the phases and which frames they touch.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..objective import Leader, site_round
from .archive import ARCHETYPES

# Catalog frames a generator line can be labeled with. "other" is a miss, not a frame.
CATALOG = tuple(name for name in ARCHETYPES if name != "other")
COVERAGE = 8
KEEP = 0.05
HARD_LINE = 0.90
MIN_BATCHES = 4
FRAMES_PER_BATCH = 3
# Growth edits an existing line. The blank-start beam is not one of these.
GROWTH_JOBS = ("sweep", "surrogate_bo", "extend")
EDIT_JOBS = GROWTH_JOBS + ("local_edit", "genetic", "gcg", "precision")


@dataclass
class ScoredLine:
    archetype: str
    origin: str
    p: float
    units: int
    phrase: str = ""


@dataclass(frozen=True)
class DealBatch:
    model: str
    frames: tuple[str, ...]


def generator_origin(origin: str) -> bool:
    """A line written by a generator batch. Edits that inherit a frame label are not."""
    return str(origin).startswith("llm:")


def credit_generator_count(counts: dict, origin: str, archetype: str) -> bool:
    """Count a scored generator line toward the frame it was labeled with.

    Inherited edits and a label outside the catalog do not move the count. The assigned
    frame is not credited when the line comes back labeled as something else.
    """
    if not generator_origin(origin) or archetype not in CATALOG:
        return False
    counts[archetype] = int(counts.get(archetype, 0) or 0) + 1
    return True


class HardSchedule:
    """One visit's phases for a hard Highest yes/no board, or an inactive schedule."""

    def __init__(
        self,
        *,
        kind: str,
        shortest: bool,
        long: bool,
        leader: Leader | None,
        we_lead: bool,
        counts: dict,
        lines: list[ScoredLine],
        models: list[str],
    ):
        self.kind = kind
        self.shortest = shortest
        self.long = long
        self.leader = leader
        self.we_lead = we_lead
        self.counts = counts
        self.lines = lines
        self.models = list(models)
        self.hard = self._is_hard()
        self.phase = "off"
        self.batches: list[DealBatch] = []
        self.growth_frames: list[str] = []
        self.deal_ran = False
        self.growth_rounds = 0

    def _is_hard(self) -> bool:
        if self.shortest or self.long or self.kind != "noul":
            return False
        lead = self.leader
        if lead is None or lead.ours or self.we_lead:
            return False
        return site_round(lead.probability) >= HARD_LINE

    def unfinished(self) -> list[str]:
        """Frames under the coverage bar, fewest generator lines first, catalog order on ties."""
        ranked = []
        for index, name in enumerate(CATALOG):
            count = int(self.counts.get(name, 0) or 0)
            if count < COVERAGE:
                ranked.append((count, index, name))
        ranked.sort()
        return [name for _, _, name in ranked]

    def plan_batches(self) -> list[DealBatch]:
        """Disjoint batches assigned up front. Empty when the schedule is off or nothing is unfinished."""
        if not self.hard or not self.models:
            return []
        queue = self.unfinished()
        if not queue:
            return []
        batches = []
        slots = max(MIN_BATCHES, len(self.models))
        for slot in range(slots):
            if not queue:
                break
            frames = tuple(queue[:FRAMES_PER_BATCH])
            del queue[: len(frames)]
            batches.append(DealBatch(self.models[slot % len(self.models)], frames))
        return batches

    def begin(self) -> None:
        self.growth_rounds = 0
        self.deal_ran = False
        self.growth_frames = []
        if not self.hard:
            self.phase = "off"
            self.batches = []
            return
        self.batches = self.plan_batches()
        if self.batches:
            self.phase = "deal"
            return
        self.phase = self._after_deal(False)

    def take_batch(self) -> DealBatch | None:
        if not self.batches:
            return None
        return self.batches.pop(0)

    def finish_deal(self) -> None:
        self.deal_ran = True
        self.batches = []
        self.phase = self._after_deal(True)

    def finish_growth(self) -> None:
        if self.phase == "grow":
            self.phase = "compress"

    def note_growth_round(self) -> None:
        if self.phase == "grow":
            self.growth_rounds += 1

    def _after_deal(self, deal_ran: bool) -> str:
        kept = self.kept_frames()
        if kept:
            self.growth_frames = kept
            return "grow"
        # The catalog pass is the visit. Do not grow or compress while frames are still unwritten.
        if deal_ran and self.unfinished():
            self.growth_frames = []
            return "done"
        if not self.unfinished():
            closest = self.closest_frame()
            if closest is not None:
                self.growth_frames = [closest]
                return "grow"
        self.growth_frames = []
        return "done"

    def displayed(self, frame: str) -> float | None:
        """Best displayed probability on this frame, whatever tool wrote the line."""
        shown = [site_round(line.p) for line in self.lines if line.archetype == frame]
        if not shown:
            return None
        return max(shown)

    def kept_frames(self) -> list[str]:
        lead = self.leader
        if lead is None:
            return []
        target = site_round(lead.probability)
        kept = []
        for name in CATALOG:
            shown = self.displayed(name)
            if shown is not None and abs(shown - target) <= KEEP + 1e-9:
                kept.append(name)
        return kept

    def closest_frame(self) -> str | None:
        lead = self.leader
        if lead is None:
            return None
        target = site_round(lead.probability)
        best: tuple[float, int, str] | None = None
        for index, name in enumerate(CATALOG):
            shown = self.displayed(name)
            if shown is None:
                continue
            rank = (abs(shown - target), index, name)
            if best is None or rank < best:
                best = rank
        return None if best is None else best[2]

    def jobs(self) -> set[str]:
        """Strategies the visit may run in this phase. The deal is not a bandit choice."""
        if self.phase == "grow":
            return set(EDIT_JOBS)
        if self.phase == "compress":
            return set(EDIT_JOBS) | {"compress"}
        return set()

    @classmethod
    def from_engine(cls, engine) -> "HardSchedule":
        from ..config import GEN_MODELS

        lines = [
            ScoredLine(cand.archetype, cand.origin, cand.p, cand.units, cand.phrase)
            for cand in engine.archive.items.values()
        ]
        counts = engine.memory.generator_counts
        if not isinstance(counts, dict):
            counts = {}
            engine.memory.generator_counts = counts
        return cls(
            kind=engine.objective.kind,
            shortest=engine.objective.shortest,
            long=engine.long,
            leader=engine.leader,
            we_lead=engine.we_lead(),
            counts=counts,
            lines=lines,
            models=list(GEN_MODELS),
        )
