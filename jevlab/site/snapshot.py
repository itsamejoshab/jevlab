"""Pull the whole live site into SQLite so the lab can run offline."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..config import SNAPSHOTS
from ..db import DB
from .client import MODES, SiteClient, SiteError, decode_jev_request

BOARD_KEYS = ("highScores", "shortestYes", "byLength", "champions", "winner")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def take_snapshot(
    db: DB,
    client: SiteClient | None = None,
    slugs: list[str] | None = None,
    modes: tuple[str, ...] = MODES,
    workers: int = 16,
    log=print,
) -> int:
    client = client or SiteClient()
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = SNAPSHOTS / stamp
    raw_dir.mkdir(parents=True, exist_ok=True)

    menu = client.menu()
    try:
        viewer = client.viewer()
    except SiteError as error:
        log(f"viewer failed ({error}); continuing signed out")
        viewer = {}
    me = ((viewer.get("player") or {}).get("user") or {}).get("id") or ""
    (raw_dir / "menu.json").write_text(json.dumps(menu, indent=1))
    (raw_dir / "viewer.json").write_text(json.dumps(viewer, indent=1))

    cursor = db.execute(
        "INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES (?, ?, ?, ?)",
        (now_iso(), client.edition, me, str(raw_dir)),
    )
    snapshot_id = cursor.lastrowid

    questions = [q for q in menu.get("questions") or [] if not slugs or q["slug"] in slugs]
    previous_goals = {row["slug"]: row["goal"] or "yes" for row in db.all("SELECT slug, goal FROM questions")}
    db.executemany(
        """INSERT OR REPLACE INTO questions
        (slug, snapshot_id, revision_id, title, instructions, kind, goal, baseline,
         yes_threshold, model_version, jev_request, raw)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                q["slug"], snapshot_id, q.get("revisionId"), q.get("title"), q.get("instructions"),
                q.get("kind"), q.get("goal"), q.get("baselineProbability"), q.get("yesThreshold"),
                q.get("modelVersion"), json.dumps(decode_jev_request(q.get("jevRequest"))), json.dumps(q),
            )
            for q in questions
        ],
    )
    log(f"snapshot {snapshot_id}: {len(questions)} questions x {len(modes)} modes (me={me or 'anonymous'})")
    from ..publish import realign_vault_goals

    realign_vault_goals(db, previous_goals, {q["slug"] for q in questions}, log)

    jobs = [(q, mode) for q in questions for mode in modes]
    log(f"fetching {len(jobs)} questions, {workers} at a time")

    def fetch(job):
        question, mode = job
        rev = question["revisionId"]
        out = {"slug": question["slug"], "mode": mode}
        for key, fn in (
            ("boards", lambda: client.leaderboards(rev, mode)),
            ("words", lambda: client.words(rev, mode)),
            ("attempt", lambda: client.attempt(rev, mode) if me else None),
        ):
            for tries in range(4):
                try:
                    out[key] = fn()
                    break
                except SiteError as error:
                    out[key + "_error"] = str(error)
                    # A read burst can trip the site's limiter. Wait that out instead of dropping the board.
                    pause = 0.8 * (tries + 1) if error.rate_limited or error.server_down else 0.2
                    time.sleep(pause)
                except Exception as error:  # network hiccups
                    out[key + "_error"] = repr(error)
                    time.sleep(0.2)
        return out

    results = []
    with ThreadPoolExecutor(workers) as pool:
        for index, result in enumerate(pool.map(fetch, jobs), start=1):
            results.append(result)
            if index % 40 == 0:
                log(f"  {index}/{len(jobs)} boards")
    (raw_dir / "boards.json").write_text(json.dumps(results))

    board_rows, word_rows, attempt_rows, score_rows = [], [], [], []
    seen = now_iso()
    for result in results:
        slug, mode = result["slug"], result["mode"]
        boards = (result.get("boards") or {}).get("leaderboards") or {}
        for key in BOARD_KEYS:
            if key in boards:
                board_rows.append((slug, mode, key, snapshot_id, json.dumps(boards[key])))
            rows = boards.get(key)
            for row in rows if isinstance(rows, list) else []:
                if me and row.get("userId") == me and row.get("phrase"):
                    score_rows.append((slug, mode, row["phrase"], row["probability"], f"board:{key}", seen))
        if result.get("words") is not None:
            word_rows.append((slug, mode, snapshot_id, json.dumps(result["words"])))
        attempt = result.get("attempt")
        if attempt:
            attempt_rows.append((slug, mode, snapshot_id, json.dumps(attempt)))
            for turn in attempt.get("turns") or []:
                if turn.get("cumulativeState") and turn.get("probability") is not None:
                    score_rows.append(
                        (slug, mode, turn["cumulativeState"], turn["probability"], "turn", turn.get("completedAt") or seen)
                    )
    db.executemany("INSERT OR REPLACE INTO boards VALUES (?, ?, ?, ?, ?)", board_rows)
    db.executemany("INSERT OR REPLACE INTO word_impacts VALUES (?, ?, ?, ?)", word_rows)
    db.executemany("INSERT OR REPLACE INTO attempts VALUES (?, ?, ?, ?)", attempt_rows)
    db.executemany("INSERT OR REPLACE INTO site_scores VALUES (?, ?, ?, ?, ?, ?)", score_rows)
    errors = sum(1 for r in results for k in r if k.endswith("_error") and k[:-6] not in r)
    log(
        f"snapshot {snapshot_id} done in {time.monotonic() - started:.0f}s: "
        f"{len(board_rows)} boards, {len(word_rows)} word lists, {len(attempt_rows)} attempts, "
        f"{len(score_rows)} site scores, {errors} failed fetches -> {raw_dir}"
    )
    return snapshot_id

