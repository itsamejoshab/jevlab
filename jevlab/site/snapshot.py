"""Pull the whole live site into SQLite so the lab can run offline."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..config import SNAPSHOTS
from ..db import DB
from .client import MODES, SiteClient, SiteError, decode_jev_request

BOARD_KEYS = ("highScores", "shortestYes", "byLength", "champions", "winner")
BAR_WIDTH = 16
BAR_FILL = "█"
BAR_EMPTY = "░"
_RULE_MARKS = ("SYNC ", "✓ ", "✗ ", "■ ", "↩ ", "▸ ")
_ELAPSED = r"(?:\d+h \d+m|\d+m \d+s|\d+s)"
_PROGRESS = re.compile(
    rf"^  (?P<who>.+?)  (?P<bar>[{re.escape(BAR_FILL)}{re.escape(BAR_EMPTY)}]{{{BAR_WIDTH}}})  "
    rf"(?P<pct>\d+)%  (?P<count>\d+/\d+)  (?P<elapsed>{_ELAPSED})"
    rf"(?:  (?P<failed>\d+) failed)?$"
)
_MODE_LABEL = {
    "strict_chain": "Strict",
    "golf": "Golf",
    "word_chain": "Casual",
    "emoji": "Emoji",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def edition_label(edition: str) -> str:
    return edition.capitalize()


def meter(pct: int, width: int = BAR_WIDTH) -> str:
    """A one-line bar. Any progress shows at least one cell; only 100% is full."""
    pct = max(0, min(100, int(pct)))
    filled = (pct * width + 50) // 100
    if pct > 0 and filled == 0:
        filled = 1
    if pct < 100 and filled == width:
        filled = width - 1
    return BAR_FILL * filled + BAR_EMPTY * (width - filled)


def log_rule(title: str) -> str:
    """A section line. The text after the mark is what the log panel keeps in its border."""
    if title.startswith("snapshot · "):
        return "SYNC " + title.removeprefix("snapshot · ")
    if " · done · " in title:
        mark = "✓"
    elif title.endswith(" · failed"):
        mark = "✗"
    elif title.startswith("stopped"):
        mark = "■"
    elif title.startswith("back to "):
        mark = "↩"
    else:
        mark = "▸"
    return f"{mark} {title}"


def rule_title(message: str) -> str | None:
    for mark in _RULE_MARKS:
        if message.startswith(mark):
            return message[len(mark) :]
    return None


def parse_progress(message: str) -> dict[str, str] | None:
    found = _PROGRESS.match(message)
    return found.groupdict() if found else None


def is_progress(message: str) -> bool:
    return parse_progress(message) is not None


def format_elapsed(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def section_mark(edition: str, section: str = "") -> str:
    name = edition_label(edition)
    return f"{section} · {name}" if section else name


def mode_labels(modes: tuple[str, ...]) -> str:
    return ", ".join(_MODE_LABEL.get(mode, mode) for mode in modes)


def progress_line(edition: str, index: int, total: int, elapsed: float, failed: int = 0, section: str = "") -> str:
    """One scan tick that still names its edition after the section header has scrolled away."""
    pct = (100 * index) // total if total else 0
    parts = [
        section_mark(edition, section),
        meter(pct),
        f"{pct}%",
        f"{index}/{total}",
        format_elapsed(elapsed),
    ]
    if failed:
        parts.append(f"{failed} failed")
    return "  " + "  ".join(parts)


def _fetch_errors(results: list[dict]) -> int:
    return sum(1 for result in results for key in result if key.endswith("_error") and key[:-6] not in result)


# Rows that belong to a question the menu used to list. A full refresh drops these with the question.
_QUESTION_TABLES = ("questions", "boards", "word_impacts", "attempts", "site_scores", "defense_words")


def _drop_absent(db: DB, live_slugs: set[str], log) -> None:
    """Remove local questions the live menu no longer lists, and the rows stored for them."""
    gone = sorted({row["slug"] for row in db.all("SELECT slug FROM questions")} - live_slugs)
    if not gone:
        return
    rows = [(slug,) for slug in gone]
    for table in _QUESTION_TABLES:
        db.executemany(f"DELETE FROM {table} WHERE slug = ?", rows)
    log(f"  removed  {len(gone)} question(s) no longer on the menu")


def fill_jev_requests(client: SiteClient, questions: list[dict], workers: int, log) -> list[dict]:
    """The menu lists questions without `jevRequest`. Opening one still returns that body."""
    missing = [q for q in questions if not q.get("jevRequest") and q.get("revisionId")]
    if not missing:
        return questions
    found: dict[str, dict] = {}

    def fetch(question: dict):
        revision = question["revisionId"]
        try:
            full = client.question(revision)
        except SiteError as error:
            log(f"  fail  {question['slug']}  {error}")
            return revision, None
        except Exception as error:
            log(f"  fail  {question['slug']}  {error!r}")
            return revision, None
        return revision, full if isinstance(full, dict) else None

    with ThreadPoolExecutor(workers) as pool:
        for revision, full in pool.map(fetch, missing):
            if full and full.get("jevRequest"):
                found[revision] = full
    filled = []
    for question in questions:
        full = found.get(question.get("revisionId") or "")
        if not full:
            filled.append(question)
            continue
        merged = dict(question)
        for key in ("jevRequest", "instructions", "modelVersion", "askedBy"):
            if full.get(key) is not None:
                merged[key] = full[key]
        filled.append(merged)
    log(f"  requests  {len(found)}/{len(missing)}")
    return filled


def take_snapshot(
    db: DB,
    client: SiteClient | None = None,
    slugs: list[str] | None = None,
    modes: tuple[str, ...] = MODES,
    workers: int = 16,
    log=print,
    section: str = "",
) -> int:
    client = client or SiteClient()
    started = time.monotonic()
    mark = section_mark(client.edition, section)
    if not section:
        log(log_rule(mark))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = SNAPSHOTS / stamp
    raw_dir.mkdir(parents=True, exist_ok=True)

    menu = client.menu()
    try:
        viewer = client.viewer()
    except SiteError as error:
        log(f"  viewer  failed ({error}); continuing signed out")
        viewer = {}
    me = ((viewer.get("player") or {}).get("user") or {}).get("id") or ""
    (raw_dir / "menu.json").write_text(json.dumps(menu, indent=1))
    (raw_dir / "viewer.json").write_text(json.dumps(viewer, indent=1))

    cursor = db.execute(
        "INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES (?, ?, ?, ?)",
        (now_iso(), client.edition, me, str(raw_dir)),
    )
    snapshot_id = cursor.lastrowid

    menu_questions = [q for q in menu.get("questions") or [] if q.get("slug")]
    questions = [q for q in menu_questions if not slugs or q["slug"] in slugs]
    questions = fill_jev_requests(client, questions, workers, log)
    previous_goals = {row["slug"]: row["goal"] or "yes" for row in db.all("SELECT slug, goal FROM questions")}
    db.executemany(
        """INSERT OR REPLACE INTO questions
        (slug, snapshot_id, revision_id, title, instructions, kind, goal, baseline,
         yes_threshold, model_version, jev_request, raw)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                q["slug"],
                snapshot_id,
                q.get("revisionId"),
                q.get("title"),
                q.get("instructions"),
                q.get("kind"),
                q.get("goal"),
                q.get("baselineProbability"),
                q.get("yesThreshold"),
                q.get("modelVersion"),
                json.dumps(decode_jev_request(q.get("jevRequest"))),
                json.dumps(q),
            )
            for q in questions
        ],
    )
    modes_bit = f" · {mode_labels(modes)}" if modes else ""
    log(f"  #{snapshot_id}  {len(questions)} questions{modes_bit} · {me or 'anonymous'}")
    from ..publish import realign_vault_goals

    realign_vault_goals(db, previous_goals, {q["slug"] for q in questions}, log)
    # A refresh of named slugs, or an empty menu, is not a full picture of what left.
    if not slugs and menu_questions:
        _drop_absent(db, {q["slug"] for q in menu_questions}, log)

    jobs = [(q, mode) for q in questions for mode in modes]
    log(f"  pull  {len(jobs)} boards · {workers} at a time")

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
            if index % 40 == 0 or index == len(jobs):
                log(
                    progress_line(
                        client.edition,
                        index,
                        len(jobs),
                        time.monotonic() - started,
                        _fetch_errors(results),
                        section,
                    )
                )
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
                        (
                            slug,
                            mode,
                            turn["cumulativeState"],
                            turn["probability"],
                            "turn",
                            turn.get("completedAt") or seen,
                        )
                    )
    db.executemany("INSERT OR REPLACE INTO boards VALUES (?, ?, ?, ?, ?)", board_rows)
    db.executemany("INSERT OR REPLACE INTO word_impacts VALUES (?, ?, ?, ?)", word_rows)
    db.executemany("INSERT OR REPLACE INTO attempts VALUES (?, ?, ?, ?)", attempt_rows)
    db.executemany("INSERT OR REPLACE INTO site_scores VALUES (?, ?, ?, ?, ?, ?)", score_rows)
    errors = _fetch_errors(results)
    done = f"{mark} · done · {format_elapsed(time.monotonic() - started)}"
    if errors:
        done += f" · {errors} failed"
    log(log_rule(done))
    log(
        f"  kept  {len(board_rows)} boards · {len(word_rows)} word lists · {len(attempt_rows)} attempts · "
        f"{len(score_rows)} scores · {errors} failed"
    )
    log(f"  wrote  {raw_dir}")
    return snapshot_id
