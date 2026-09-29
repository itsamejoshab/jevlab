"""Publish queued vault winners to the live site, one at a time.

Separate from the lab on purpose: it never imports the search engine. For
each queued entry it re-reads the live board, confirms we still win,
re-scores once with the oracle, then builds the chain with TURN operations.
The board keeps a player's best-ever phrase, so a noisy final score can be
re-rolled by removing and re-adding the last word.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
import unicodedata
from datetime import datetime, timezone

from . import vault
from .config import DATA
from .db import DB
from .modes import GOLF, HIGH_SCORES, STRICT, from_board
from .objective import Leader, Objective, board_leader, site_round, target_rows
from .oracle import Oracle
from .rules import RuleError, check_phrase, drop_clashing, option_clash, rejected
from .site import SiteClient, SiteError, active_words

LOCK_PATH = DATA / "publish.lock"
# A Shortest-yes gamble that lands 1 roll in 5 still hits within 13 rolls (build + 12 re-rolls) ~95% of the time.
GAMBLE_REROLLS = 12
# The Golf board lists a new row a few seconds after its turn is scored.
GOLF_BOARD_LAG = 5.0
# Vault lines tried on one board after Jev rejects a word, before leaving the rest for the next run.
MAX_FALLBACKS = 5
# A line estimated from the other edition gets one oracle sample; it is skipped only when that sample plus this
# much still loses, since one noisy roll should not kill a line the site may still score higher.
ESTIMATE_MARGIN = 0.05


class PublishError(RuntimeError):
    pass


class WordRejected(PublishError):
    """Jev's strictWordCheck refused a word as more than one word; no re-roll or retry changes that."""

    def __init__(self, word: str, message: str):
        super().__init__(f"{word!r}: {message}")
        self.word = word


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def turn_with_retry(client: SiteClient, attempt: dict, operation: dict, log, retries: int = 6) -> tuple[dict, dict]:
    for tries in range(retries + 1):
        try:
            return client.turn(attempt, operation)
        except SiteError as error:
            if tries == retries:
                raise
            if error.busy:
                pause = 3.0 * (1 + tries * 0.5)
                log(f"    busy ({error}); retry in {pause:.0f}s")
                time.sleep(pause)
                fresh = client.attempt(attempt["revisionId"], attempt["playMode"])
                if fresh:
                    attempt = fresh
                continue
            if error.rate_limited:
                pause = 60.0
                log(f"    rate limited ({error}); backing off {pause:.0f}s")
                time.sleep(pause)
                client.delay = max(client.delay, 2.0)
                continue
            if error.server_down:
                pause = 5.0 * (1 + tries)
                log(f"    {error}; retry in {pause:.0f}s")
                time.sleep(pause)
                try:
                    fresh = client.attempt(attempt["revisionId"], attempt["playMode"])
                except SiteError as again:
                    log(f"    reload failed too: {again}")
                    continue
                if fresh and len(fresh.get("turns") or []) > len(attempt.get("turns") or []):
                    log("    the turn landed despite the error; keeping it")
                    return fresh, fresh["turns"][-1]
                if fresh:
                    attempt = fresh
                continue
            raise
    raise PublishError("ran out of retries")


def build_chain(client: SiteClient, attempt: dict, words: list[str], log) -> tuple[dict, float | None]:
    """Make the attempt's active words equal `words` with as few turns as possible."""
    current = active_words(attempt)
    last_p: float | None = None
    if current == words:
        turns = attempt.get("turns") or []
        return attempt, float(turns[-1]["probability"]) if turns else None
    keep = 0
    while keep < min(len(current), len(words)) and current[keep] == words[keep]:
        keep += 1
    extra = len(current) - keep
    # Popping `extra` tail words beats one clear plus re-appending the `keep` shared words.
    if extra and extra <= keep:
        for _ in range(extra):
            tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
            attempt, turn = turn_with_retry(client, attempt, {"kind": "remove", "tokenId": tokens[-1]["id"]}, log)
            last_p = float(turn.get("probability") or 0)
            log(f"    - {tokens[-1]['text']}")
        current = active_words(attempt)
    elif extra:
        attempt, turn = turn_with_retry(client, attempt, {"kind": "clear"}, log)
        last_p = float(turn.get("probability") or 0)
        current = active_words(attempt)
        log("    cleared chain")
    if current:
        log(f"    keeping {len(current)} words already on the chain")
    for word in words[len(current) :]:
        try:
            attempt, turn = turn_with_retry(client, attempt, {"kind": "append", "text": word}, log)
        except SiteError as error:
            if rejected.is_word_rejection(str(error)):
                raise WordRejected(word, str(error)) from error
            raise
        last_p = float(turn["probability"])
        log(f"    + {word:<16} {last_p:.2f}")
    if active_words(attempt) != words:
        raise PublishError(f"chain mismatch: {' '.join(active_words(attempt))!r}")
    return attempt, last_p


def reroll(client: SiteClient, attempt: dict, log) -> tuple[dict, float]:
    """Remove and re-append the last word to get a fresh score for the same phrase."""
    tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
    last = tokens[-1]
    attempt, _ = turn_with_retry(client, attempt, {"kind": "remove", "tokenId": last["id"]}, log)
    attempt, turn = turn_with_retry(client, attempt, {"kind": "append", "text": last["text"]}, log)
    return attempt, float(turn["probability"])


def golf_try(client: SiteClient, live: dict, text: str, log, target: str = "") -> float:
    """Golf scores one whole phrase per attempt, so every try (and re-roll) starts a fresh attempt."""
    attempt = client.start(live["revisionId"], GOLF)
    _attempt, turn = turn_with_retry(client, attempt, {"kind": "append", "text": text}, log)
    return credited(turn, target)


def refresh_board(db: DB, client: SiteClient, live: dict, mode: str) -> None:
    """Write the live boards for one question back into the snapshot tables."""
    try:
        boards = client.leaderboards(live["revisionId"], mode).get("leaderboards") or {}
    except SiteError:
        return
    row = db.latest_snapshot()
    snapshot_id = row["id"] if row else 0
    db.executemany(
        "INSERT OR REPLACE INTO boards VALUES (?, ?, ?, ?, ?)",
        [(live["slug"], mode, key, snapshot_id, json.dumps(value)) for key, value in boards.items()],
    )


def credited(turn: dict | None, target: str) -> float:
    """What a turn scores for `target`: the site files a row under Jev's picked answer, so a turn only counts
    toward the target when Jev picked it."""
    if not turn:
        return 0.0
    if target and turn.get("choice") != target:
        return 0.0
    return float(turn.get("probability") or 0.0)


def last_turn(attempt: dict | None) -> dict | None:
    turns = (attempt or {}).get("turns") or []
    return turns[-1] if turns else None


async def oracle_check(db: DB, question: dict, phrase: str, n: int = 3, target: str = "") -> float:
    oracle = Oracle(db, question["jev_request"], target=target)
    try:
        score = await oracle.score_one(phrase, n=n)
    finally:
        await oracle.close()
    return Objective(question.get("goal") or "yes", question["kind"]).p(score)


def publish_entry(db: DB, client: SiteClient, entry: dict, questions: dict[str, dict], me: str,
                  dry_run: bool, rerolls: int, verify: bool, log, long_rerolls: int = 5) -> str:
    slug, mode, phrase = entry["slug"], entry["mode"], entry["phrase"]
    board = entry.get("board") or HIGH_SCORES
    target = entry.get("target") or ""
    if mode not in (STRICT, GOLF):
        raise PublishError(f"mode {mode} is not supported by the publisher yet")
    game_mode = from_board(board, mode)
    u = game_mode.unit_abbr
    live = questions.get(slug)
    if live is None:
        raise PublishError("question is gone from the live menu")
    stored = db.question(slug)
    if stored is None:
        raise PublishError("question not in the snapshot")
    if stored["revision_id"] != live["revisionId"]:
        raise PublishError("question revision changed since the snapshot; re-snapshot and re-search")
    objective = Objective(live.get("goal") or "yes", live["kind"], unit=game_mode.unit, board=board)
    if game_mode.golf:
        text = unicodedata.normalize("NFC", phrase).strip()
        if not text:
            raise PublishError("empty golf phrase")
        words = [text]
        units = objective.units(text)
        clash = option_clash(text, list(live.get("choices") or []))
        if clash:
            raise PublishError(f"{clash[0]!r} contains part of the answer option {clash[1]!r}")
    else:
        known = next((w for w in phrase.split() if rejected.is_rejected(w)), None)
        if known:
            raise WordRejected(known, "rejected on an earlier publish")
        try:
            # Plateau boards grow lines past the search's usual 60-word cap; leaders there run to 231 words.
            words = check_phrase(phrase, list(live.get("choices") or []), max_words=400).split()
        except RuleError as error:
            raise PublishError(f"phrase breaks strict rules: {error}") from error
        units = len(words)
    boards = client.leaderboards(live["revisionId"], mode).get("leaderboards") or {}
    boards = {b: drop_clashing(r, live.get("choices"), me) if isinstance(r, list) else r for b, r in boards.items()}
    rows = target_rows(boards, board, target) if target else boards.get(board) or []
    leader = board_leader(rows, me, board=board)
    ours = max((r for r in rows if me and r.get("userId") == me),
               key=lambda r: objective.leader_key(Leader(float(r["probability"]), int(r["wordCount"]))), default=None)
    estimate = float(entry["p_mean"])
    gamble = objective.shortest and bool(entry.get("gamble")) and (leader is None or units < leader.units)
    aim = max(estimate, objective.entry_p(entry, leader)) if gamble else estimate
    where = f"{board} for {target!r}" if target else board
    lead = f"{leader.probability:.2f}/{leader.units}{u} {leader.name}" if leader else "empty board"
    log(f"  [{where}] live leader {lead}; ours {ours['probability']:.2f}/{ours['wordCount']}{u}" if ours
        else f"  [{where}] live leader {lead}")
    if ours and objective.leader_key(Leader(float(ours["probability"]), int(ours["wordCount"]))) \
            >= objective.key(aim, units):
        vault.set_status(slug, mode, phrase, "published", board, target=target,
                         detail="our board row is already as good")
        return "already"
    if objective.shortest and not game_mode.golf:
        log("  rebuilding the chain is safe: each board keeps our best-ever row, so Highest stays as it is")
    if not objective.beats(aim, units, leader):
        raise PublishError(f"no longer beats the live leader ({lead})")
    estimated = entry.get("estimated_from")
    if estimated and verify:
        fresh = asyncio.run(oracle_check(db, stored, phrase, n=1, target=target))
        log(f"  estimated from {estimated}: one-sample check {fresh:.3f} ({estimated} said {estimate:.3f})")
        if not objective.beats(min(fresh + ESTIMATE_MARGIN, 1.0), units, leader):
            raise PublishError(f"check {fresh:.3f} is well short of {lead}")
    if gamble:
        rerolls = max(rerolls, GAMBLE_REROLLS)
        log(f"  gamble: {entry.get('hits')} oracle rolls cleared {objective.threshold:.2f} (mean {estimate:.3f}); "
            f"skipping the oracle re-check, up to {rerolls} re-rolls until one lands")
    elif entry.get("long_shot"):
        rerolls = long_rerolls
        log(f"  long shot: lower bound {entry['p_lcb']:.3f} does not beat {lead}; skipping the oracle re-check, "
            f"up to {rerolls} re-rolls")
    elif verify and not estimated:
        fresh = asyncio.run(oracle_check(db, stored, phrase, target=target))
        log(f"  oracle re-check {fresh:.3f} (vault said {estimate:.3f})")
        if not objective.beats(fresh, units, leader):
            raise PublishError(f"oracle re-check {fresh:.3f} no longer beats {lead}")
    need = leader
    if dry_run:
        what = f"send {units} characters" if game_mode.golf else f"build {units} words"
        log(f"  dry run: would {what}: {phrase}")
        return "dry-run"

    if game_mode.golf:
        best = golf_try(client, live, words[0], log, target)
        log(f"    = {units}c {best:.2f}")
    else:
        attempt = client.attempt(live["revisionId"], mode) or client.start(live["revisionId"], mode)
        attempt, server_p = build_chain(client, attempt, words, log)
        best = credited(last_turn(attempt), target) if target else (server_p if server_p is not None else 0.0)
    tries = 0
    while not objective.beats(best, units, need) and tries < rerolls:
        tries += 1
        if game_mode.golf:
            p = golf_try(client, live, words[0], log, target)
        else:
            attempt, p = reroll(client, attempt, log)
            if target:
                p = credited(last_turn(attempt), target)
        best = max(best, p)
        log(f"    re-roll {tries}: {p:.2f} (best {best:.2f})")
    won = objective.beats(best, units, need)
    db.execute(
        "INSERT INTO publishes (slug, mode, phrase, estimate, server, status, detail, at, board) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (slug, mode, phrase, estimate, best, "won" if won else "short",
         json.dumps({"rerolls": tries} | ({"target": target} if target else {})), now_iso(), board),
    )
    if not target:
        # The site's own score for the line; later searches and triage learn from it.
        db.execute("INSERT OR REPLACE INTO site_scores VALUES (?, ?, ?, ?, ?, ?)",
                   (slug, mode, phrase, best, "publish", now_iso()))
    if game_mode.golf:
        time.sleep(GOLF_BOARD_LAG)
    refresh_board(db, client, live, mode)
    if won:
        vault.set_status(slug, mode, phrase, "published", board, target=target, server_p=best,
                         detail=f"rerolls={tries}")
        return f"won {site_round(best):.2f}/{units}{u}"
    vault.set_status(slug, mode, phrase, "failed", board, target=target, server_p=best,
                     detail=f"server {best:.2f} after {tries} re-rolls did not beat {lead}")
    return f"short {best:.2f}"


def reject_and_fall_through(entry: dict, error: WordRejected, used: int, log) -> dict | None:
    """Remember the rejected word, retire the line, and queue the next vault line for the same board."""
    board = entry.get("board") or HIGH_SCORES
    target = entry.get("target") or ""
    rejected.add(error.word, entry["slug"], str(error))
    vault.set_status(entry["slug"], entry["mode"], entry["phrase"], "rejected", board, target=target,
                     detail=str(error))
    if used >= MAX_FALLBACKS:
        log(f"  {used} fall-throughs on this board already; leaving the rest for the next run")
        return None
    data = vault.load(entry["slug"], entry["mode"], board, target)
    pool = [e for e in data["entries"]
            if e.get("status") == "candidate" and (e.get("beats") or e.get("gamble"))
            and not any(rejected.is_rejected(w) for w in e["phrase"].split())]
    if not pool:
        log("  no other winning vault line on this board; search it again")
        return None
    if from_board(board, entry["mode"]).shortest:
        pool.sort(key=lambda e: (e["units"], -round(e["p_mean"], 2), -e["p_lcb"]))
    else:
        pool.sort(key=lambda e: (-round(e["p_mean"], 2), e["units"], -e["p_lcb"]))
    pick = pool[0] | {"slug": entry["slug"], "mode": entry["mode"], "board": board, "target": target}
    vault.set_status(pick["slug"], pick["mode"], pick["phrase"], "queued", board, target=target)
    log(f"  falling through to the next vault line: {pick['phrase']}")
    return pick | {"status": "queued"}


def board_key(entry: dict) -> tuple[str, str, str, str]:
    return (entry["slug"], entry["mode"], entry.get("board") or HIGH_SCORES, entry.get("target") or "")


def golf_followups(db: DB, slug: str, log) -> list[dict]:
    """Queue this question's Golf cross-posts and return every queued Golf entry for it."""
    from .crosspost import picks, queue

    found = picks(db, slugs=[slug])
    if found:
        queue(db, found)
        log(f"  cross-posting {len(found)} line(s) to Golf")
    return [e for e in vault.queued() if e["mode"] == GOLF and e["slug"] == slug]


def publish(dry_run: bool = False, rerolls: int = 3, verify: bool = True, slugs: list[str] | None = None,
            delay: float = 0.01, log=print, entries: list[dict] | None = None,
            on_result=None, long_rerolls: int = 5, board: str | None = None, mode: str | None = None,
            crosspost: bool = True) -> int:
    """Publish queued entries (or exactly `entries`), from every board and play mode unless `board` or `mode`
    is given. With `crosspost`, every Strict line that lands (or already holds) its board is followed by the
    question's Golf cross-posts, queued and published in the same run.
    on_result(entry, result_text) fires per entry."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another publisher is running")
            return 1
        if entries is None:
            entries = [e for e in vault.queued(board)
                       if (not slugs or e["slug"] in slugs) and (not mode or e["mode"] == mode)]
        if not entries:
            log("nothing queued. `jevlab vault queue --q <slug> --best` to queue a winner")
            return 0
        db = DB()
        client = SiteClient(delay=delay)
        me = ((client.viewer().get("player") or {}).get("user") or {}).get("id") or ""
        if not me and not dry_run:
            log("not signed in: put a session cookie in session.json or JEV_COOKIE")
            return 1
        questions = {q["slug"]: q for q in client.menu().get("questions") or []}
        failures = 0
        entries = list(entries)
        seen = {board_key(e) + (e["phrase"],) for e in entries}
        crossposted: set[str] = set()
        fallbacks: dict[tuple[str, str, str, str], int] = {}
        for entry in entries:
            game_mode = from_board(entry.get("board"), entry.get("mode"))
            aimed = f" -> {entry['target']}" if entry.get("target") else ""
            log(f"\n== {entry['slug']} [{game_mode.label}{aimed}] "
                f"{entry['p_mean']:.3f}/{entry['units']}{game_mode.unit_abbr}: {entry['phrase']}")
            try:
                result = publish_entry(db, client, entry, questions, me, dry_run, rerolls, verify, log,
                                       long_rerolls)
                log(f"  -> {result}")
            except WordRejected as error:
                result = f"rejected: {error}"
                log(f"  -> {result}")
                if not dry_run:
                    key = board_key(entry)
                    fallback = reject_and_fall_through(entry, error, fallbacks.get(key, 0), log)
                    if fallback is None:
                        failures += 1
                    else:
                        fallbacks[key] = fallbacks.get(key, 0) + 1
                        seen.add(key + (fallback["phrase"],))
                        entries.append(fallback)
                else:
                    failures += 1
            except (PublishError, SiteError) as error:
                failures += 1
                result = f"failed: {error}"
                log(f"  -> {result}")
                if not dry_run:
                    vault.set_status(entry["slug"], entry["mode"], entry["phrase"], "failed",
                                     entry.get("board") or HIGH_SCORES, target=entry.get("target") or "",
                                     detail=str(error))
            if on_result:
                on_result(entry, result)
            landed = result.startswith("won") or result == "already"
            if crosspost and not dry_run and landed and entry["mode"] == STRICT and entry["slug"] not in crossposted:
                crossposted.add(entry["slug"])
                for golf in golf_followups(db, entry["slug"], log):
                    key = board_key(golf) + (golf["phrase"],)
                    if key not in seen:
                        seen.add(key)
                        entries.append(golf)
        client.save_session()
        return 1 if failures else 0
