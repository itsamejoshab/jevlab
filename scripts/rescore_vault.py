"""Replace every Jev vault score with the current checkpoint.

A vault phrase is topped up with samples from the current model. Once that set is
complete, samples from older checkpoints are removed and the vault numbers are
rewritten from the current samples alone. Run the script again to finish a stopped
pass; a phrase already on this checkpoint is left as it is.

    uv run python scripts/rescore_vault.py [slug ...]
    uv run python scripts/rescore_vault.py --dry-run
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT = "1.14"
BATCH = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _current(model: str | None) -> bool:
    return CHECKPOINT in (model or "")


def counts(db, qkey: str, request_hash: str, state: str) -> tuple[int, int]:
    """(current samples, older samples) for one phrase on one question."""
    row = db.one(
        """SELECT
             SUM(CASE WHEN model LIKE '%1.14%' AND noul IS NOT NULL THEN 1 ELSE 0 END) AS cur,
             SUM(CASE WHEN model IS NULL OR model NOT LIKE '%1.14%' THEN 1 ELSE 0 END) AS old
           FROM oracle_samples
           WHERE state = ? AND (qkey = ? OR request_hash = ?)""",
        (state, qkey, request_hash),
    )
    if row is None:
        return 0, 0
    return int(row["cur"] or 0), int(row["old"] or 0)


def keep_current(db, qkey: str, request_hash: str, state: str) -> None:
    """Drop older samples for one phrase and rebuild its phrase summary from what remains."""
    with db.lock:
        db.conn.execute("BEGIN")
        try:
            db.conn.execute(
                """DELETE FROM oracle_samples
                   WHERE state = ? AND (qkey = ? OR request_hash = ?)
                     AND (model IS NULL OR model NOT LIKE '%1.14%')""",
                (state, qkey, request_hash),
            )
            row = db.conn.execute(
                """SELECT COUNT(*) AS n, COALESCE(SUM(noul), 0) AS total, COALESCE(SUM(noul * noul), 0) AS sumsq
                   FROM oracle_samples
                   WHERE state = ? AND noul IS NOT NULL AND model LIKE '%1.14%'
                     AND (qkey = ? OR request_hash = ?)""",
                (state, qkey, request_hash),
            ).fetchone()
            n, total, sumsq = int(row["n"]), float(row["total"]), float(row["sumsq"])
            if n:
                db.conn.execute(
                    """INSERT INTO phrase_stats (qkey, state, n, total, sumsq) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(qkey, state) DO UPDATE SET
                         n = excluded.n, total = excluded.total, sumsq = excluded.sumsq""",
                    (qkey, state, n, total, sumsq),
                )
            else:
                removed = db.conn.execute(
                    "DELETE FROM phrase_stats WHERE qkey = ? AND state = ?", (qkey, state)
                ).rowcount
                if removed:
                    db.conn.execute("UPDATE lab_meta SET n = MAX(n - 1, 0) WHERE key = 'phrases'")
            db.conn.execute("COMMIT")
        except Exception:
            db.conn.execute("ROLLBACK")
            raise


def current_rows(db, qkey: str, request_hash: str, state: str) -> list:
    return db.all(
        """SELECT noul, answer FROM oracle_samples
           WHERE state = ? AND model LIKE '%1.14%' AND (qkey = ? OR request_hash = ?)""",
        (state, qkey, request_hash),
    )


def values_of(rows: list, target: str) -> list[float]:
    from jevlab.oracle import target_value

    out = []
    for row in rows:
        if target:
            value = target_value(row["answer"], target)
        else:
            value = row["noul"]
        if value is not None:
            out.append(float(value))
    return out


def apply_entry(entry: dict, score, objective, leader, board: str) -> None:
    """Write one vault line from a current-checkpoint score. Status stays as it is."""
    from jevlab.modes import HIGH_SCORES

    p = objective.p(score)
    already = entry.get("checkpoint") == CHECKPOINT
    entry["p_mean"] = round(p, 4)
    entry["p_lcb"] = round(objective.p_lcb(score), 4)
    entry["spread"] = round(score.spread, 4)
    entry["n"] = score.n
    entry["updated_at"] = _now()
    entry["checkpoint"] = CHECKPOINT
    entry["leader"] = (
        {"p": leader.probability, "units": leader.units, "name": leader.name} if leader else None
    )
    if "jev_p" in entry:
        entry["jev_p"] = round(p, 4)
    if entry.get("status") != "published":
        entry["beats"] = objective.wins(score, entry["units"], leader)
    if board != HIGH_SCORES:
        if objective.gamble(score, entry["units"], leader):
            entry["gamble"] = True
            entry["p_reach"] = round(max(objective.goal_samples(score)), 4)
            entry["hits"] = f"{objective.hits(score)}/{score.n}"
        else:
            entry["gamble"] = False
            entry["p_reach"] = None
            entry["hits"] = None
    if not already:
        entry.setdefault("history", []).append(
            {"status": "rescored", "at": entry["updated_at"], "checkpoint": CHECKPOINT, "n": score.n, "p_mean": entry["p_mean"]}
        )


def sort_entries(entries: list[dict], board: str) -> None:
    from jevlab.modes import HIGH_SCORES

    if board == HIGH_SCORES:
        entries.sort(key=lambda e: (-round(e["p_mean"], 2), e["units"], -e["p_mean"]))
    else:
        entries.sort(key=lambda e: (e["units"], -round(e["p_mean"], 2), -e["p_mean"]))


def load_books(root: Path, slugs: set[str] | None) -> dict[str, list[dict]]:
    books: dict[str, list[dict]] = {}
    if not root.exists():
        return books
    for path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        slug = data.get("slug") or path.parent.name
        if slugs is not None and slug not in slugs:
            continue
        data.setdefault("slug", slug)
        data.setdefault("target", "")
        books.setdefault(slug, []).append(data)
    return books


def phrase_need(files: list[dict]) -> dict[str, int]:
    need: dict[str, int] = {}
    for data in files:
        for entry in data.get("entries") or []:
            phrase = entry.get("phrase") or ""
            if not phrase:
                continue
            n = max(int(entry.get("n") or 1), 1)
            need[phrase] = max(need.get(phrase, 1), n)
    return need


def stamped(files: list[dict], phrase: str) -> bool:
    found = False
    for data in files:
        for entry in data.get("entries") or []:
            if entry.get("phrase") != phrase:
                continue
            found = True
            if entry.get("checkpoint") != CHECKPOINT:
                return False
    return found


def rewrite(files: list[dict], phrase: str, rows: list, question: dict, leaders: dict) -> int:
    from jevlab.modes import HIGH_SCORES, from_board
    from jevlab.objective import objective_for
    from jevlab.oracle import Score

    written = 0
    for data in files:
        board = data.get("board") or HIGH_SCORES
        target = data.get("target") or ""
        mode = data.get("mode") or ""
        samples = values_of(rows, target)
        if not samples:
            continue
        score = Score(phrase, samples)
        game = from_board(board, mode)
        objective = objective_for(question, unit=game.unit, board=board)
        leader = leaders.get((mode, board))
        changed = False
        for entry in data.get("entries") or []:
            if entry.get("phrase") != phrase:
                continue
            apply_entry(entry, score, objective, leader, board)
            changed = True
            written += 1
        if changed:
            sort_entries(data["entries"], board)
            data["_dirty"] = True
    return written


def save_dirty(files: list[dict]) -> None:
    from jevlab import vault

    for data in files:
        if not data.pop("_dirty", False):
            continue
        vault._write(data)


def leaders_for(db, slug: str, files: list[dict]) -> dict:
    from jevlab.modes import HIGH_SCORES
    from jevlab.objective import board_leader

    found = {}
    for data in files:
        mode = data.get("mode") or ""
        board = data.get("board") or HIGH_SCORES
        key = (mode, board)
        if key in found:
            continue
        found[key] = board_leader(db.board(slug, mode, board), db.me(), board=board)
    return found


def is_refusal(error: BaseException) -> bool:
    from jevlab.oracle import OracleError

    return isinstance(error, OracleError) and "refused to answer" in str(error)


async def top_up(oracle, jobs: list[tuple[str, int]]) -> None:
    from jevlab.llm import key_denied

    tasks = [oracle._one(state) for state, need in jobs for _ in range(need)]
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    denied = [r for r in results if isinstance(r, Exception) and key_denied(str(r))]
    if denied:
        raise denied[0]
    fatal = [r for r in results if isinstance(r, Exception) and not is_refusal(r)]
    if fatal:
        raise fatal[0]


async def rescore(db, books: dict[str, list[dict]], dry_run: bool, log=print) -> int:
    from jevlab.oracle import Oracle, build_request, question_key, request_hash

    calls = 0
    updated = 0
    short = 0
    done = 0
    phrases = sum(len(phrase_need(files)) for files in books.values())
    log(f"{phrases} phrases across {len(books)} questions", flush=True)
    for slug, files in books.items():
        question = db.question(slug)
        if not question or not question.get("jev_request"):
            log(f"{slug}: not in the snapshot", flush=True)
            continue
        request = question["jev_request"]
        qkey = question_key(request)
        need = phrase_need(files)
        pending = []
        for phrase, n in need.items():
            key = request_hash(build_request(request, phrase))
            cur, old = counts(db, qkey, key, phrase)
            if cur >= n and old == 0 and stamped(files, phrase):
                done += 1
                continue
            pending.append((phrase, n, key, max(n - cur, 0)))
        if not pending:
            continue
        owing = sum(item[3] for item in pending)
        log(f"{slug}: {len(pending)} phrases, {owing} new samples", flush=True)
        if dry_run:
            calls += owing
            continue
        leaders = leaders_for(db, slug, files)
        oracle = Oracle(db, request)
        oracle.on_error = lambda message: log(message)
        try:
            for start in range(0, len(pending), BATCH):
                chunk = pending[start : start + BATCH]
                await top_up(oracle, [(phrase, extra) for phrase, _n, _key, extra in chunk if extra])
                for phrase, n, key, _extra in chunk:
                    cur, _old = counts(db, qkey, key, phrase)
                    if cur < n:
                        short += 1
                        log(f"  short {cur}/{n} {phrase[:60]}", flush=True)
                        continue
                    keep_current(db, qkey, key, phrase)
                    written = rewrite(files, phrase, current_rows(db, qkey, key, phrase), question, leaders)
                    updated += written
                save_dirty(files)
                log(
                    f"  {slug} {min(start + BATCH, len(pending))}/{len(pending)}  calls {oracle.calls}  ${oracle.cost:.4f}",
                    flush=True,
                )
        finally:
            await oracle.close()
            calls += oracle.calls
    log(
        f"done  already current {done}  entries written {updated}  still short {short}  calls {calls}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from jevlab.config import activate

    activate("jev")
    from jevlab.config import VAULT
    from jevlab.db import DB

    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in args
    slugs = {item for item in args if item != "--dry-run"}
    books = load_books(VAULT, slugs or None)
    if slugs:
        missing = slugs - set(books)
        for slug in sorted(missing):
            print(f"{slug}: no vault file", flush=True)
    if not books:
        print("no vault entries")
        return 1
    return asyncio.run(rescore(DB(), books, dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
