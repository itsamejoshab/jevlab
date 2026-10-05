"""Copy another edition's best measured Strict lines in as estimates.

Each edition has its own model and boards. `import_from` stores the source's lower bound as a stand-in
(`estimated_from`, n=0) and does not queue or publish. A later publish on this edition is what the site scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import vault
from .config import EDITION, EDITIONS, JEV_DATA, JEV_DB_PATH, JEV_VAULT
from .db import DB
from .modes import GOLF, HIGH_SCORES, STRICT, from_board
from .objective import Leader, board_leader, objective_for, target_rows
from .rules import RuleError, check_phrase, option_clash, option_names, rejected

SKIP_STATUSES = ("rejected", "dropped", "failed")


@dataclass
class Imported:
    entry: dict  # the Kev vault entry after saving
    leader: Leader | None
    beats: bool
    queued: bool = False


def edition_root(edition: str) -> Path:
    return JEV_DATA if edition == "jev" else JEV_DATA / edition


def edition_db_path(edition: str) -> Path:
    return JEV_DB_PATH if edition == "jev" else edition_root(edition) / "jev.db"


def edition_vault(edition: str) -> Path:
    return JEV_VAULT if edition == "jev" else edition_root(edition) / "vault"


def jev_db() -> DB | None:
    """Jev's database, read-only; None when it does not exist."""
    return DB(JEV_DB_PATH, readonly=True) if JEV_DB_PATH.exists() else None


def _source_db(edition: str) -> DB | None:
    path = edition_db_path(edition)
    return DB(path, readonly=True) if path.exists() else None


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


def _measured(entry: dict) -> bool:
    """The source edition scored this line itself. Estimates and dead lines are not crowns to borrow."""
    return not entry.get("estimated_from") and int(entry.get("n") or 0) > 0 and entry.get("status") not in SKIP_STATUSES


def import_from(
    db: DB,
    source: str,
    slugs: list[str] | None = None,
    board: str | None = None,
    mode: str | None = None,
    log=print,
    source_vault: Path | None = None,
    source_db: DB | None = None,
) -> list[Imported]:
    """Copy one best measured Strict line per question, board, and choice answer from `source`.

    The line is stored here as an estimate (`n=0`). Nothing is queued. Golf lines are not copied.
    `source_vault` and `source_db` override the edition's files (tests)."""
    if source not in EDITIONS:
        raise RuntimeError(f"unknown edition {source!r}; pick one of {', '.join(EDITIONS)}")
    if source == EDITION:
        others = [name for name in EDITIONS if name != EDITION]
        raise RuntimeError(f"already {EDITION}; import from one of {', '.join(others)}")
    if mode not in (None, STRICT):
        raise RuntimeError("import copies Strict lines only; Golf is posted when a Strict publish lands")
    if db.latest_snapshot() is None:
        raise RuntimeError(f"no {EDITION} snapshot yet: run `jevlab --edition {EDITION} snapshot` first")
    opened = source_db is None
    src = source_db if source_db is not None else _source_db(source)
    goals = {r["slug"]: r["goal"] for r in src.all("SELECT slug, goal FROM questions")} if src else {}
    root = source_vault if source_vault is not None else edition_vault(source)
    questions: dict[str, dict | None] = {}
    pools: dict[tuple[str, str, str], list[dict]] = {}
    for entry in vault.all_entries(board, root=root):
        if entry["mode"] != STRICT or not _measured(entry):
            continue
        if slugs and entry["slug"] not in slugs:
            continue
        key = (entry["slug"], entry["board"], entry.get("target") or "")
        pools.setdefault(key, []).append(entry)
    out: list[Imported] = []
    skipped = {"not on this edition": 0, "goal differs": 0, "rules": 0, "answer missing": 0, "ranked": 0}
    for (slug, entry_board, target), entries in sorted(pools.items()):
        if slug not in questions:
            questions[slug] = db.question(slug)
        question = questions[slug]
        if not question:
            skipped["not on this edition"] += len(entries)
            continue
        goal = question.get("goal") or "yes"
        if goals.get(slug, goal) != goal:
            skipped["goal differs"] += len(entries)
            continue
        raw = question.get("raw") or {}
        if question.get("kind") == "choice" and bool(raw.get("ranked")):
            skipped["ranked"] += len(entries)
            continue
        choices = list(raw.get("choices") or [])
        if target and target not in option_names(choices):
            skipped["answer missing"] += len(entries)
            continue
        game_mode = from_board(entry_board, STRICT)
        objective = objective_for(question, unit=game_mode.unit, board=entry_board)
        leader = kev_leader(db, slug, STRICT, entry_board, target)
        ranked_entries = []
        for entry in entries:
            phrase = fits_kev(entry["phrase"], STRICT, choices)
            if not phrase:
                skipped["rules"] += 1
                continue
            units = objective.units(phrase)
            ranked_entries.append((objective.key(float(entry["p_lcb"]), units), phrase, units, entry))
        if not ranked_entries:
            continue
        current = vault.load(slug, STRICT, entry_board, target)["entries"]
        copied = False
        for _, phrase, units, entry in sorted(ranked_entries, key=lambda row: row[0], reverse=True):
            prior = next((item for item in current if item["phrase"] == phrase), None)
            if prior and not prior.get("estimated_from"):
                continue
            if prior and prior.get("estimated_from") and float(prior["p_lcb"]) >= float(entry["p_lcb"]):
                copied = True
                break
            extra = None
            if entry_board != HIGH_SCORES:
                extra = {k: entry.get(k) for k in ("gamble", "p_reach", "hits") if k in entry}
            probe = {
                "units": units,
                "p_lcb": entry["p_lcb"],
                "gamble": entry.get("gamble"),
                "p_reach": entry.get("p_reach"),
            }
            beats = objective.beats(objective.entry_p(probe, leader), units, leader)
            saved = vault.save(
                slug,
                STRICT,
                phrase,
                p_mean=entry["p_mean"],
                p_lcb=entry["p_lcb"],
                spread=entry.get("spread", 0.0),
                n=0,
                units=units,
                leader=leader,
                beats=beats,
                title=question.get("title") or "",
                origin=source,
                note=f"{source} {entry.get('status')}",
                board=entry_board,
                extra=extra,
                target=target,
                estimated_from=source,
                jev_p=entry["p_mean"],
            )
            if not saved.get("estimated_from"):
                continue
            out.append(
                Imported(
                    saved | {"slug": slug, "mode": STRICT, "board": entry_board, "target": target},
                    leader,
                    bool(saved.get("beats")),
                )
            )
            copied = True
            break
        if not copied and ranked_entries:
            skipped.setdefault("already measured", 0)
            skipped["already measured"] += 1
    if opened and src is not None:
        src.conn.close()
    log(
        f"imported {len(out)} {source} line(s) into the {EDITION} vault; "
        f"{sum(1 for i in out if i.beats)} beat a {EDITION} leader"
        + "".join(f"; {n} skipped ({why})" for why, n in skipped.items() if n)
    )
    return out


def import_from_jev(
    db: DB,
    slugs: list[str] | None = None,
    board: str | None = None,
    mode: str | None = None,
    queue: bool = False,
    log=print,
) -> list[Imported]:
    """Copy Jev's best measured Strict lines into this edition. `queue` is ignored; publish does the sending."""
    del queue
    return import_from(db, "jev", slugs=slugs, board=board, mode=mode, log=log)


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
        pool = [
            i
            for i in pool
            if not ours or objective.leader_key(ours) < objective.key(i.entry["p_mean"], i.entry["units"])
        ]
        if not pool:
            continue
        pick = max(
            pool,
            key=lambda i: (objective.key(objective.entry_p(i.entry, i.leader), i.entry["units"]), i.entry["p_lcb"]),
        )
        vault.set_status(slug, mode, pick.entry["phrase"], "queued", board, target=target, detail="estimated from jev")
        pick.queued = True
        lead = f"{pick.leader.probability:.2f}/{pick.leader.units}{game_mode.unit_abbr}" if pick.leader else "empty"
        aimed = f" -> {target}" if target else ""
        log(
            f"queued {game_mode.name:<13} jev {pick.entry['p_mean']:.3f}/{pick.entry['units']}"
            f"{game_mode.unit_abbr} vs {EDITION} {lead:<9} {slug}{aimed}: {pick.entry['phrase']}"
        )
