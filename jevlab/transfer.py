"""Borrow Jev's vault for a mirror edition (Kev, Laya, or Clef): every Jev line becomes a line estimated from Jev.

A mirror is the same game with its own model, question revisions, and boards. Instead of searching from scratch,
Jev's winners are copied into this edition's vault with Jev's scores as stand-ins (`estimated_from="jev"`, n=0)
and judged against this edition's leaders. Publishing one gets a real score from the site; triage
(search/triage.py) measures the most promising ones with this edition's oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import vault
from .config import EDITION, JEV_DB_PATH, JEV_VAULT
from .db import DB
from .modes import GOLF, HIGH_SCORES, STRICT, from_board
from .objective import Leader, board_leader, objective_for, target_rows
from .rules import RuleError, check_phrase, option_clash, option_names, rejected

SKIP_STATUSES = ("rejected", "dropped")


@dataclass
class Imported:
    entry: dict  # the Kev vault entry after saving
    leader: Leader | None
    beats: bool
    queued: bool = False


def jev_db() -> DB | None:
    """Jev's database, read-only; None when it does not exist."""
    return DB(JEV_DB_PATH, readonly=True) if JEV_DB_PATH.exists() else None


def kev_leader(db: DB, slug: str, mode: str, board: str, target: str) -> Leader | None:
    if target:
        rows = target_rows({b: db.board(slug, mode, b) for b in (board, "champions")}, board, target)
    else:
        rows = db.board(slug, mode, board)
    return board_leader(rows, db.me(), board=board)


def kev_ours(db: DB, slug: str, mode: str, board: str, target: str) -> Leader | None:
    me = db.me()
    if not me:
        return None
    rows = db.board(slug, mode, board)
    if target:
        rows = [r for r in rows if r.get("choice") == target]
    return board_leader([r for r in rows if r.get("userId") == me], me, include_ours=True, board=board)


def fits_kev(phrase: str, mode: str, choices: list) -> str | None:
    """The phrase as Kev's rules take it, or None when they refuse it."""
    if mode == GOLF:
        return None if option_clash(phrase, choices) else phrase.strip()
    if any(rejected.is_rejected(w) for w in phrase.split()):
        return None
    try:
        return check_phrase(phrase, choices)
    except RuleError:
        return None


def import_from_jev(db: DB, slugs: list[str] | None = None, board: str | None = None, mode: str | None = None,
                    queue: bool = False, log=print) -> list[Imported]:
    """Copy Jev vault lines into this edition's vault as estimates. With `queue`, queue the best estimated line
    on each board whose Jev estimate beats the leader (unless that board already has something queued)."""
    if EDITION == "jev":
        raise RuntimeError(
            "import-jev runs under Kev, Laya, or Clef: `jevlab --edition clef vault import-jev`"
        )
    if db.latest_snapshot() is None:
        raise RuntimeError(f"no {EDITION} snapshot yet: run `jevlab --edition {EDITION} snapshot` first")
    jev = jev_db()
    jev_goals = {r["slug"]: r["goal"] for r in jev.all("SELECT slug, goal FROM questions")} if jev else {}
    questions: dict[str, dict | None] = {}
    out: list[Imported] = []
    skipped = {"not on this edition": 0, "goal differs": 0, "rules": 0, "answer missing": 0}
    for e in vault.all_entries(board, root=JEV_VAULT):
        if e.get("status") in SKIP_STATUSES or e["mode"] not in (STRICT, GOLF):
            continue
        if (slugs and e["slug"] not in slugs) or (mode and e["mode"] != mode):
            continue
        slug, target, entry_board = e["slug"], e.get("target") or "", e["board"]
        if slug not in questions:
            questions[slug] = db.question(slug)
        question = questions[slug]
        if not question:
            skipped["not on this edition"] += 1
            continue
        goal = question.get("goal") or "yes"
        if jev_goals.get(slug, goal) != goal:
            skipped["goal differs"] += 1
            continue
        choices = list((question.get("raw") or {}).get("choices") or [])
        if target and target not in option_names(choices):
            skipped["answer missing"] += 1
            continue
        phrase = fits_kev(e["phrase"], e["mode"], choices)
        if not phrase:
            skipped["rules"] += 1
            continue
        game_mode = from_board(entry_board, e["mode"])
        objective = objective_for(question, unit=game_mode.unit, board=entry_board)
        units = objective.units(phrase)
        leader = kev_leader(db, slug, e["mode"], entry_board, target)
        extra = None
        if entry_board != HIGH_SCORES:
            extra = {k: e.get(k) for k in ("gamble", "p_reach", "hits") if k in e}
        probe = {"units": units, "p_lcb": e["p_lcb"], "gamble": e.get("gamble"), "p_reach": e.get("p_reach")}
        beats = objective.beats(objective.entry_p(probe, leader), units, leader)
        saved = vault.save(slug, e["mode"], phrase, p_mean=e["p_mean"], p_lcb=e["p_lcb"],
                           spread=e.get("spread", 0.0), n=0, units=units, leader=leader, beats=beats,
                           title=question.get("title") or "", origin="jev", note=f"jev {e.get('status')}",
                           board=entry_board, extra=extra, target=target, estimated_from="jev", jev_p=e["p_mean"])
        out.append(Imported(saved | {"slug": slug, "mode": e["mode"], "board": entry_board, "target": target},
                            leader, bool(saved.get("beats"))))
    if jev:
        jev.conn.close()
    log(f"imported {len(out)} Jev line(s) into the {EDITION} vault; "
        f"{sum(1 for i in out if i.beats)} beat a {EDITION} leader"
        + "".join(f"; {n} skipped ({why})" for why, n in skipped.items() if n))
    if queue:
        queue_best(db, out, log)
    return out


def queue_best(db: DB, found: list[Imported], log=print) -> None:
    """Per board, queue the best estimated winner that is still a candidate."""
    boards: dict[tuple[str, str, str, str], list[Imported]] = {}
    for item in found:
        e = item.entry
        boards.setdefault((e["slug"], e["mode"], e["board"], e["target"]), []).append(item)
    for (slug, mode, board, target), items in sorted(boards.items()):
        current = vault.load(slug, mode, board, target)["entries"]
        if any(x.get("status") == "queued" for x in current):
            continue
        pool = [i for i in items if i.beats and i.entry.get("status") == "candidate" and i.entry.get("estimated_from")]
        if not pool:
            continue
        game_mode = from_board(board, mode)
        question = db.question(slug) or {}
        objective = objective_for(question, unit=game_mode.unit, board=board)
        ours = kev_ours(db, slug, mode, board, target)
        pool = [i for i in pool
                if not ours or objective.leader_key(ours) < objective.key(i.entry["p_mean"], i.entry["units"])]
        if not pool:
            continue
        pick = max(pool, key=lambda i: (objective.key(objective.entry_p(i.entry, i.leader), i.entry["units"]),
                                        i.entry["p_lcb"]))
        vault.set_status(slug, mode, pick.entry["phrase"], "queued", board, target=target, detail="estimated from jev")
        pick.queued = True
        lead = f"{pick.leader.probability:.2f}/{pick.leader.units}{game_mode.unit_abbr}" if pick.leader else "empty"
        aimed = f" -> {target}" if target else ""
        log(f"queued {game_mode.name:<13} jev {pick.entry['p_mean']:.3f}/{pick.entry['units']}"
            f"{game_mode.unit_abbr} vs {EDITION} {lead:<9} {slug}{aimed}: {pick.entry['phrase']}")
