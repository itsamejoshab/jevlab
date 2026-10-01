"""Cross-post Strict vault lines to the Golf boards.

Golf scores the same text the same way, but takes the whole phrase in one turn and counts characters. So every
Strict line we have scored (from either Strict board, any status but dropped) is re-ranked by characters against
the Golf leader in the snapshot, and the best one per question and board is queued into the golf vault. Lines
aimed at one answer of a choice question are cross-posted per answer, against the Golf rows filed under it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import vault
from .db import DB
from .modes import GOLF, GOLF_HIGHEST, GOLF_SHORTEST, STRICT, GameMode
from .objective import Leader, board_leader, objective_for, target_rows


@dataclass
class Pick:
    slug: str
    mode: GameMode
    entry: dict
    units: int
    leader: Leader | None
    ours: Leader | None
    target: str = ""


def strict_pool(slugs: list[str] | None = None) -> dict[tuple[str, str], list[dict]]:
    """Strict vault lines per (question, answer); answer is "" for whole-question lines."""
    pool: dict[tuple[str, str], dict[str, dict]] = {}
    for entry in vault.all_entries():
        if entry["mode"] != STRICT or entry.get("status") == "dropped":
            continue
        if slugs and entry["slug"] not in slugs:
            continue
        seen = pool.setdefault((entry["slug"], entry.get("target") or ""), {})
        # The same phrase can sit on both Strict boards; keep the copy with more oracle samples.
        prior = seen.get(entry["phrase"])
        if prior is None or entry.get("n", 0) > prior.get("n", 0):
            seen[entry["phrase"]] = entry
    return {key: list(by_phrase.values()) for key, by_phrase in pool.items()}


def picks(db: DB, modes: tuple[GameMode, ...] = (GOLF_HIGHEST, GOLF_SHORTEST),
          slugs: list[str] | None = None) -> list[Pick]:
    """Best Strict line per question (or answer) and Golf board whose lower bound beats the Golf leader."""
    me = db.me()
    out = []
    for (slug, target), entries in sorted(strict_pool(slugs).items()):
        question = db.question(slug)
        if not question:
            continue
        for mode in modes:
            if target:
                rows = target_rows({b: db.board(slug, GOLF, b) for b in (mode.board, "champions")}, mode.board,
                                   target)
            else:
                rows = db.board(slug, GOLF, mode.board)
            leader = board_leader(rows, me, board=mode.board)
            ours = board_leader([r for r in rows if me and r.get("userId") == me], me, include_ours=True,
                                board=mode.board)
            objective = objective_for(question, unit="char", board=mode.board)
            ranked = []
            for entry in entries:
                units = objective.units(entry["phrase"])
                p = float(entry["p_lcb"])
                if not objective.beats(p, units, leader):
                    continue
                if ours and objective.leader_key(ours) >= objective.key(float(entry["p_mean"]), units):
                    continue
                ranked.append((objective.key(p, units), float(entry["p_mean"]), units, entry))
            if not ranked:
                continue
            _, _, units, entry = max(ranked, key=lambda r: (r[0], r[1]))
            out.append(Pick(slug, mode, entry, units, leader, ours, target))
    return out


def queue(db: DB, found: list[Pick]) -> None:
    for pick in found:
        e = pick.entry
        question = db.question(pick.slug) or {}
        vault.save(pick.slug, GOLF, e["phrase"], p_mean=e["p_mean"], p_lcb=e["p_lcb"], spread=e.get("spread", 0.0),
                   n=e.get("n", 0), units=pick.units, leader=pick.leader, beats=True,
                   title=e.get("title") or question.get("title") or "", origin="crosspost", status="queued",
                   note=f"from strict {e['board']}", board=pick.mode.board, target=pick.target,
                   estimated_from=e.get("estimated_from") or "", jev_p=e.get("jev_p"))
