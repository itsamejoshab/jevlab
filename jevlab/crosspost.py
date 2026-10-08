"""Cross-post Strict vault lines to the Golf boards.

Golf scores the same text the same way, but takes the whole phrase in one turn and counts characters. So every
Strict line we have scored (from either Strict board, any status but dropped) is re-ranked by characters against
the Golf leader in the snapshot, and the best one per question and board is queued into the golf vault. Lines
aimed at one answer of a choice question are cross-posted per answer, against the Golf rows filed under it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import vault
from .vault import site_p
from .db import DB
from .modes import (
    CASUAL,
    CASUAL_HIGHEST,
    CASUAL_SHORTEST,
    GOLF,
    GOLF_HIGHEST,
    GOLF_SHORTEST,
    HIGH_SCORES,
    STRICT,
    GameMode,
)
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


def picks(
    db: DB, modes: tuple[GameMode, ...] = (GOLF_HIGHEST, GOLF_SHORTEST), slugs: list[str] | None = None
) -> list[Pick]:
    """Best Strict line per question (or answer) and Golf board whose lower bound beats the Golf leader."""
    me = db.me()
    out = []
    for (slug, target), entries in sorted(strict_pool(slugs).items()):
        question = db.question(slug)
        if not question:
            continue
        for mode in modes:
            known = {
                (entry["slug"], entry.get("target") or "", entry["phrase"]): site_p(entry)
                for entry in vault.all_entries(mode.board)
                if entry["mode"] == GOLF and site_p(entry) is not None
            }
            if target:
                rows = target_rows({b: db.board(slug, GOLF, b) for b in (mode.board, "champions")}, mode.board, target)
            else:
                rows = db.board(slug, GOLF, mode.board)
            leader = board_leader(rows, me, board=mode.board)
            ours = board_leader(
                [r for r in rows if me and r.get("userId") == me], me, include_ours=True, board=mode.board
            )
            objective = objective_for(question, unit="char", board=mode.board)
            ranked = []
            for entry in entries:
                units = objective.units(entry["phrase"])
                recorded = known.get((slug, target, entry["phrase"]))
                p = recorded if recorded is not None else float(entry["p_lcb"])
                mean = recorded if recorded is not None else float(entry["p_mean"])
                if not objective.beats(p, units, leader):
                    continue
                if ours and objective.leader_key(ours) >= objective.key(mean, units):
                    continue
                ranked.append((objective.key(p, units), mean, units, entry))
            if not ranked:
                continue
            _, _, units, entry = max(ranked, key=lambda r: (r[0], r[1]))
            out.append(Pick(slug, mode, entry, units, leader, ours, target))
    return out


def casual_picks(db: DB, slugs: list[str] | None = None) -> list[Pick]:
    """The Strict Highest answer, on each Casual board that answer would take.

    Casual ranks by words, the same way Strict Highest does. A leader already at one or two words wins a
    tie against a longer line at the same score, so a crowded board usually yields nothing. An empty board
    is a take, temporary or not. Shortest yes still requires the line to clear that question's yes line.
    """
    me = db.me()
    out = []
    for (slug, target), entries in sorted(strict_pool(slugs).items()):
        question = db.question(slug)
        if not question:
            continue
        highest = objective_for(question, unit="word", board=HIGH_SCORES)
        ranked_src = []
        for entry in entries:
            recorded = site_p(entry)
            p = recorded if recorded is not None else float(entry["p_lcb"])
            mean = recorded if recorded is not None else float(entry["p_mean"])
            units = highest.units(entry["phrase"])
            ranked_src.append((highest.key(p, units), mean, units, entry))
        _, _, units, entry = max(ranked_src, key=lambda row: (row[0], row[1]))
        for mode in (CASUAL_HIGHEST, CASUAL_SHORTEST):
            known = {
                (item["slug"], item.get("target") or "", item["phrase"]): site_p(item)
                for item in vault.all_entries(mode.board)
                if item["mode"] == CASUAL and site_p(item) is not None
            }
            if target:
                rows = target_rows(
                    {b: db.board(slug, CASUAL, b) for b in (mode.board, "champions")}, mode.board, target
                )
            else:
                rows = db.board(slug, CASUAL, mode.board)
            leader = board_leader(rows, me, board=mode.board)
            ours = board_leader(
                [r for r in rows if me and r.get("userId") == me], me, include_ours=True, board=mode.board
            )
            objective = objective_for(question, unit="word", board=mode.board)
            recorded = known.get((slug, target, entry["phrase"]))
            p = recorded if recorded is not None else float(entry["p_lcb"])
            mean = recorded if recorded is not None else float(entry["p_mean"])
            if not objective.beats(p, units, leader):
                continue
            if ours and objective.leader_key(ours) >= objective.key(mean, units):
                continue
            out.append(Pick(slug, mode, entry, units, leader, ours, target))
    return out


def queue(db: DB, found: list[Pick]) -> None:
    for pick in found:
        e = pick.entry
        question = db.question(pick.slug) or {}
        vault.save(
            pick.slug,
            pick.mode.play_mode,
            e["phrase"],
            p_mean=e["p_mean"],
            p_lcb=e["p_lcb"],
            spread=e.get("spread", 0.0),
            n=e.get("n", 0),
            units=pick.units,
            leader=pick.leader,
            beats=True,
            title=e.get("title") or question.get("title") or "",
            origin="crosspost",
            status="queued",
            note=f"from strict {e['board']}",
            board=pick.mode.board,
            target=pick.target,
            estimated_from=e.get("estimated_from") or "",
            jev_p=e.get("jev_p"),
        )
