"""Per-question standing from the local snapshot plus the vault, for the pickers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import vault
from .vault import site_p
from .db import DB
from .modes import HIGH_SCORES, SHORTEST_YES, STRICT, is_searchable
from .objective import Leader, Objective, board_leader, objective_for, site_round, target_rows
from .oracle import Score, question_key
from .rules import RuleError, check_phrase, option_names, rejected
from .rules.banned import contains as phrase_banned
from .search.memory import RunMemory, by_slug, by_target

TARGET_SEP = "::"


def split_key(key: str) -> tuple[str, str]:
    """(slug, answer) from a Standing.key."""
    slug, _, target = key.partition(TARGET_SEP)
    return slug, target


@dataclass
class Standing:
    slug: str
    title: str
    kind: str
    goal: str
    leader: Leader | None  # best other player
    ours: Leader | None  # our board row
    best_entry: dict | None  # best vault entry not yet published/dropped
    we_lead: bool
    entry_beats: bool  # best vault entry's lower bound beats the current leader
    memory: RunMemory | None = None  # earlier search runs
    long_shot: dict | None = None  # scored line whose mean, but not lower bound, beats the leader
    board: str = HIGH_SCORES
    ranked: bool = False
    target: str = ""  # one answer of a choice question; "" is the whole question

    @property
    def key(self) -> str:
        """Picker id: the slug, or slug::answer for an answer row."""
        return f"{self.slug}{TARGET_SEP}{self.target}" if self.target else self.slug

    @property
    def label(self) -> str:
        return f"{self.title} -> {self.target}" if self.target else self.title

    @property
    def searchable(self) -> bool:
        return is_searchable(self.kind, self.ranked)

    @property
    def unwinnable(self) -> bool:
        """Leader already has the top rounded score in one word; we could only tie, and ties go to them.
        The same holds on both boards: one word at 1.00 cannot be beaten on length or probability."""
        return (
            bool(self.leader)
            and not self.we_lead
            and site_round(self.leader.probability) >= 1.0
            and self.leader.units <= 1
        )

    @property
    def should_search(self) -> bool:
        return self.searchable and not self.we_lead and not self.unwinnable

    @property
    def target_p(self) -> float:
        return self.leader.probability if self.leader else 0.0

    @property
    def search_order(self) -> tuple:
        """Highest: weakest leader first; among equal scores, the longest (easiest to undercut).
        Shortest yes: longest leader first, then the weakest score."""
        units = self.leader.units if self.leader else 0
        if self.board == SHORTEST_YES:
            return (-units, round(self.target_p, 2), self.slug)
        return (round(self.target_p, 2), -units, self.slug)


def has_rejected_word(phrase: str) -> bool:
    return any(rejected.is_rejected(w) for w in phrase.split())


def long_shot(
    db: DB,
    slug: str,
    mode: str,
    objective: Objective,
    leader: Leader | None,
    ours: Leader | None,
    tried: set[str],
    min_n: int = 2,
) -> dict | None:
    """Best scored line whose mean beats the leader on the board while its lower bound does not."""
    question = db.question(slug)
    if not question or not question.get("jev_request") or leader is None:
        return None
    choices = list((question.get("raw") or {}).get("choices") or [])
    shots = []
    for r in db.all(
        "SELECT state, n, total, sumsq FROM phrase_stats WHERE qkey = ?",
        (question_key(question["jev_request"]),),
    ):
        if r["n"] < min_n or r["state"] in tried:
            continue
        state = r["state"]
        score = Score.moments(state, int(r["n"]), float(r["total"]), float(r["sumsq"]))
        p, lcb, units = objective.p(score), objective.p_lcb(score), len(state.split())
        if not objective.beats(p, units, leader) or objective.beats(lcb, units, leader):
            continue
        if ours and objective.leader_key(ours) >= objective.key(p, units):
            continue
        shots.append(((objective.key(p, units), p), state, score, p, lcb, units))
    for _, state, score, p, lcb, units in sorted(shots, key=lambda s: s[0], reverse=True):
        try:
            check_phrase(state, choices)
        except RuleError:
            continue
        if phrase_banned(state):
            continue
        return {
            "slug": slug,
            "mode": mode,
            "board": objective.board,
            "phrase": state,
            "p_mean": round(p, 4),
            "p_lcb": round(lcb, 4),
            "spread": round(score.spread, 4),
            "n": score.n,
            "units": units,
            "status": "long shot",
            "long_shot": True,
        }
    return None


def standings(
    db: DB, mode: str = STRICT, long_shots: bool = False, board: str = HIGH_SCORES, targets: bool = False
) -> list[Standing]:
    """One row per question; with `targets`, also one row per answer of each searchable choice question."""
    me = db.me()
    entries: dict[tuple[str, str], list[dict]] = {}
    spent: dict[str, set[str]] = {}
    for entry in vault.all_entries(board):
        if entry["mode"] != mode:
            continue
        key = (entry["slug"], entry.get("target") or "")
        if (
            entry.get("status") in ("candidate", "queued", "failed")
            and not has_rejected_word(entry["phrase"])
            and not phrase_banned(entry["phrase"])
        ):
            entries.setdefault(key, []).append(entry)
        if entry.get("status") in ("published", "failed", "rejected", "dropped"):
            spent.setdefault(entry["slug"], set()).add(entry["phrase"])
    memories = by_slug(db, board) if mode == STRICT else {}
    target_memories = by_target(db, board) if mode == STRICT and targets else {}
    out = []
    for row in db.all(
        "SELECT slug, title, kind, goal, yes_threshold, json_extract(raw, '$.ranked') AS ranked, "
        "json_extract(raw, '$.choices') AS choices FROM questions ORDER BY slug"
    ):
        boards = {b: db.board(row["slug"], mode, b) for b in (board, "champions")}
        ranked = bool(row["ranked"])
        aims = [""]
        if targets and row["kind"] == "choice" and is_searchable(row["kind"], ranked):
            aims += option_names(json.loads(row["choices"] or "[]"))
        for target in aims:
            rows = target_rows(boards, board, target) if target else boards[board]
            memory = target_memories.get((row["slug"], target)) if target else memories.get(row["slug"])
            out.append(
                standing(
                    db,
                    row,
                    rows,
                    entries.get((row["slug"], target), []),
                    spent,
                    memory,
                    me,
                    mode,
                    board,
                    ranked,
                    long_shots and not target,
                    target,
                )
            )
    return out


def standing(
    db: DB,
    row,
    rows: list[dict],
    pool: list[dict],
    spent: dict[str, set[str]],
    memory,
    me: str,
    mode: str,
    board: str,
    ranked: bool,
    long_shots: bool,
    target: str,
) -> Standing:
    leader = board_leader(rows, me, board=board)
    ours = board_leader([r for r in rows if me and r.get("userId") == me], me, include_ours=True, board=board)
    objective = objective_for(dict(row), board=board)
    we_lead = bool(ours) and (leader is None or objective.leader_key(ours) > objective.leader_key(leader))

    # Winning lines come first, so a saved long shot never hides a safe winner. On Shortest yes a gamble
    # (fewer words, some rolls over the threshold) is judged on its best roll, so it outranks longer lines.
    def offer_p(entry: dict) -> float:
        """A recorded site score replaces the local estimate for the next submit."""
        known = site_p(entry)
        return known if known is not None else objective.entry_p(entry, leader)

    pool.sort(
        key=lambda e: (
            not objective.beats(offer_p(e), e["units"], leader),
            tuple(-x for x in objective.key(offer_p(e), e["units"])),
            -offer_p(e),
        )
    )
    best = pool[0] if pool else None
    beats = False
    if best:
        p_win = offer_p(best)
        beats = objective.beats(p_win, best["units"], leader)
        if ours and objective.leader_key(ours) >= objective.key(p_win, best["units"]):
            beats = False
    shot = None
    if long_shots and mode == STRICT and not we_lead and is_searchable(row["kind"] or "noul", ranked):
        shot = long_shot(db, row["slug"], mode, objective, leader, ours, set(spent.get(row["slug"], ())))
    return Standing(
        row["slug"],
        row["title"] or row["slug"],
        row["kind"] or "",
        row["goal"] or "yes",
        leader,
        ours,
        best,
        we_lead,
        beats,
        memory,
        shot,
        board,
        ranked,
        target,
    )
