"""Publish queued vault winners to the live site, one at a time.

Separate from the lab on purpose: it never imports the search engine. For
each queued entry it re-reads the live board, confirms we still win,
re-scores once with the oracle, then builds the chain with TURN operations.
The board keeps a player's best-ever phrase, so the requested re-rolls always
run, even after a win: a later roll can score higher. A re-roll removes and
re-adds the last word.
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
from .objective import Leader, Objective, board_leader, objective_for, site_round, target_rows
from .oracle import Oracle, question_key
from .rules import RuleError, check_phrase, drop_clashing, option_clash, rejected
from .rules.banned import BannedPhrase, hit as hit_ban, note as note_ban
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
# A re-check this close to 1 - vault score was stored under the opposite goal. Near 0.5 the two readings overlap.
GOAL_FLIP = 0.01


class PublishError(RuntimeError):
    pass


class WrongGoal(PublishError):
    """The vault score is the complement of the site goal. `phrases` were scored the other way and removed."""

    def __init__(self, message: str, slug: str, phrases: set[str]):
        super().__init__(message)
        self.slug = slug
        self.phrases = phrases


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
            named = note_ban(error.code, error.message)
            if error.code == "banned_phrase" or named:
                raise BannedPhrase(str(operation.get("text") or ""), named or "") from error
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


def crossed_finish(turn: dict | None, stop_at: float | None) -> bool:
    """A live chain is done once a turn reaches the round's finish line. Further words cannot count."""
    if stop_at is None or not turn:
        return False
    if turn.get("reachedYes"):
        return True
    try:
        return float(turn.get("probability") or 0) >= stop_at - 1e-9
    except (TypeError, ValueError):
        return False


def _round_closed(error: SiteError) -> bool:
    text = f"{error.code} {error.message}".casefold()
    return any(part in text for part in ("finished", "ended", "closed", "complete", "claimed", "round over"))


def _chain_matches(current: list[str], words: list[str]) -> bool:
    """True when the attempt already shows `words`, one token per word or one phrase per append."""
    return " ".join(current) == " ".join(words)


def build_chain(
    client: SiteClient,
    attempt: dict,
    words: list[str],
    log,
    stop_at: float | None = None,
    group: int = 1,
) -> tuple[dict, float | None]:
    """Make the attempt's active words equal `words` with as few turns as possible.

    `stop_at` is the live round's finish line. Crossing it holds the chain: the words already
    accepted stay, and the rest are not sent.

    `group` is how many words go in one append. Word-by-word play uses 1. A sentence round
    sends the phrase as one append.
    """
    current = active_words(attempt)
    last_p: float | None = None
    turns = attempt.get("turns") or []
    if crossed_finish(turns[-1] if turns else None, stop_at):
        last_p = float(turns[-1].get("probability") or 0)
        log(f"    chain already crossed the finish line at {last_p:.2f}; holding")
        return attempt, last_p
    if _chain_matches(current, words):
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
    rest = words[len(current) :]
    step = max(1, group)
    for start in range(0, len(rest), step):
        chunk = rest[start : start + step]
        text = chunk[0] if step == 1 else " ".join(chunk)
        try:
            attempt, turn = turn_with_retry(client, attempt, {"kind": "append", "text": text}, log)
        except SiteError as error:
            if rejected.is_word_rejection(str(error)):
                raise WordRejected(text, str(error)) from error
            if stop_at is not None and _round_closed(error):
                log(f"    round closed the chain ({error.message}); holding")
                return attempt, last_p
            raise
        last_p = float(turn["probability"])
        shown = text if step == 1 else f"{text} ({len(chunk)}w)"
        log(f"    + {shown:<16} {last_p:.2f}")
        if crossed_finish(turn, stop_at):
            placed = len(active_words(attempt))
            log(f"    finish line crossed at {last_p:.2f} after {placed} word{'s' if placed != 1 else ''}; holding")
            return attempt, last_p
    if not _chain_matches(active_words(attempt), words):
        raise PublishError(f"chain mismatch: {' '.join(active_words(attempt))!r}")
    return attempt, last_p


def reroll(client: SiteClient, attempt: dict, log) -> tuple[dict, float]:
    """Remove and re-append the last word to get a fresh score for the same phrase."""
    tokens = [t for t in attempt.get("tokens") or [] if t.get("active", True)]
    last = tokens[-1]
    attempt, _ = turn_with_retry(client, attempt, {"kind": "remove", "tokenId": last["id"]}, log)
    attempt, turn = turn_with_retry(client, attempt, {"kind": "append", "text": last["text"]}, log)
    return attempt, float(turn["probability"])


def should_reroll(done: int, asked: int, ceiling: int, ahead: bool) -> bool:
    """Spend `asked` re-rolls even after a win. Rolls past that, up to `ceiling`, continue only while short."""
    if done >= ceiling:
        return False
    if ahead and done >= asked:
        return False
    return True


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


def goal_flipped(vault_p: float, checked_p: float) -> bool:
    """True when `checked_p` is within 0.01 of the complement of `vault_p`, and the two readings can be told apart."""
    if abs(vault_p - 0.5) <= GOAL_FLIP:
        return False
    return abs(checked_p - (1.0 - vault_p)) <= GOAL_FLIP


def _raw_means(db: DB, qkey: str) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in db.all("SELECT state, noul FROM oracle_samples WHERE qkey = ? AND noul IS NOT NULL", (qkey,)):
        totals.setdefault(row["state"], []).append(float(row["noul"]))
    return {state: sum(values) / len(values) for state, values in totals.items()}


def _scored_wrong(p_mean: float, raw: float, site_goal: str) -> bool:
    """A stored score matches the goal the site does not use. Raw samples are always P(yes)."""
    if abs(raw - 0.5) <= GOAL_FLIP:
        return False
    wrong = raw if site_goal == "no" else 1.0 - raw
    right = 1.0 - wrong
    return abs(p_mean - wrong) <= GOAL_FLIP and abs(p_mean - right) > GOAL_FLIP


def realign_vault_goals(db: DB, previous: dict[str, str], slugs: set[str] | None, log) -> set[str]:
    """After a snapshot writes the site's goals, drop vault lines scored the other way.

    `previous` is slug to goal from before the replace. A changed goal is logged. Every refreshed
    question that already has vault lines is checked too: a live search can store P(yes) while the
    snapshot goal was already `no`.
    """
    vaulted = {entry["slug"] for entry in vault.all_entries()}
    removed: set[str] = set()
    for row in db.all("SELECT slug, goal, kind FROM questions"):
        slug = row["slug"]
        if slugs is not None and slug not in slugs:
            continue
        site = (row["goal"] or "yes").casefold()
        old = (previous.get(slug) or "").casefold()
        kind = (row["kind"] or "noul").casefold()
        changed = bool(old) and old != site
        if kind != "noul" or (not changed and slug not in vaulted):
            continue
        if changed:
            log(f"  {slug}: goal {old!r} -> {site!r}")
        question = db.question(slug)
        if question:
            removed |= forget_wrong_goal(db, question, site, log)
    return removed


def forget_wrong_goal(db: DB, question: dict, site_goal: str, log) -> set[str]:
    """Drop vault lines, and the guesses behind them, that were scored against the opposite of `site_goal`."""
    slug = question["slug"]
    qkey = question_key(question["jev_request"]) if question.get("jev_request") else ""
    means = _raw_means(db, qkey) if qkey else {}
    phrases = {
        entry["phrase"]
        for entry in vault.all_entries()
        if entry["slug"] == slug
        and entry["phrase"] in means
        and _scored_wrong(float(entry["p_mean"]), means[entry["phrase"]], site_goal)
    }
    if not phrases:
        return set()
    removed = vault.drop_phrases(slug, phrases)
    for phrase in phrases:
        if qkey:
            db.execute("DELETE FROM oracle_samples WHERE qkey = ? AND state = ?", (qkey, phrase))
            db.execute("DELETE FROM lab_candidates WHERE qkey = ? AND state = ?", (qkey, phrase))
    _scrub_memory(db, slug, phrases, [float(entry["p_mean"]) for entry in removed])
    opposite = "yes" if site_goal == "no" else "no"
    log(f"  removed {len(removed)} vault line(s) scored against goal {opposite!r}, and their guess history")
    return phrases


def _scrub_memory(db: DB, slug: str, phrases: set[str], removed_ps: list[float]) -> None:
    """Run memory stores the score of each guess. Drop the ones from the wrong-goal run."""
    for row in db.all("SELECT qkey, board, data FROM lab_memory WHERE slug = ?", (slug,)):
        try:
            memory = json.loads(row["data"] or "{}")
        except json.JSONDecodeError:
            continue
        memory["exhausted"] = [item for item in memory.get("exhausted") or [] if not item or item[0] not in phrases]
        memory["history"] = [
            item
            for item in memory.get("history") or []
            if not any(abs(float(item.get("best_p") or 0) - p) <= GOAL_FLIP for p in removed_ps)
        ]
        if any(abs(float(memory.get("best_p") or 0) - p) <= GOAL_FLIP for p in removed_ps):
            memory["best_p"] = 0.0
            memory["best_units"] = 0
        db.execute(
            "UPDATE lab_memory SET data = ? WHERE qkey = ? AND board = ?",
            (json.dumps(memory), row["qkey"], row["board"]),
        )


def settle_goal(db: DB, stored: dict, live: dict, vault_p: float, checked_p: float, dry_run: bool, log) -> float:
    """A complement re-check. Use the site goal, and drop lines that were scored the other way.

    Returns the re-check expressed as the site's P(goal). Raises WrongGoal when this vault line is one of them.
    """
    if not goal_flipped(vault_p, checked_p):
        return checked_p
    site = (live.get("goal") or "yes").casefold()
    stored_goal = (stored.get("goal") or "yes").casefold()
    kind = (stored.get("kind") or live.get("kind") or "noul").casefold()
    log(f"  re-check {checked_p:.3f} is within {GOAL_FLIP:.2f} of 1 - vault {vault_p:.3f}")
    log(f"  site goal {site!r}; snapshot goal {stored_goal!r}")
    if kind != "noul":
        raise PublishError(
            f"oracle re-check {checked_p:.3f} is the complement of the vault {vault_p:.3f}; site goal is {site!r}"
        )
    if site != stored_goal:
        checked_p = 1.0 - checked_p
        log(f"  snapshot goal {stored_goal!r} does not match the site; re-check on {site!r} is {checked_p:.3f}")
        if not dry_run:
            raw = dict(stored.get("raw") or {})
            raw["goal"] = site
            db.execute(
                "UPDATE questions SET goal = ?, raw = ? WHERE slug = ?",
                (site, json.dumps(raw), stored["slug"]),
            )
            stored["goal"] = site
            stored["raw"] = raw
            forget_wrong_goal(db, stored, site, log)
        return checked_p
    if dry_run:
        raise WrongGoal(
            f"vault {vault_p:.3f} was scored against the opposite of the site goal {site!r}",
            stored["slug"],
            set(),
        )
    removed = forget_wrong_goal(db, stored, site, log)
    opposite = "yes" if site == "no" else "no"
    raise WrongGoal(
        f"vault {vault_p:.3f} was scored against goal {opposite!r}; the site goal is {site!r}. "
        f"removed {len(removed)} vault line(s) and their guess history",
        stored["slug"],
        removed,
    )


async def oracle_check(db: DB, question: dict, phrase: str, n: int = 3, target: str = "") -> float:
    oracle = Oracle(db, question["jev_request"], target=target)
    try:
        score = await oracle.score_one(phrase, n=n)
    finally:
        await oracle.close()
    return Objective(question.get("goal") or "yes", question["kind"]).p(score)


def publish_entry(
    db: DB,
    client: SiteClient,
    entry: dict,
    questions: dict[str, dict],
    me: str,
    dry_run: bool,
    rerolls: int,
    verify: bool,
    log,
    long_rerolls: int = 5,
) -> str:
    slug, mode, phrase = entry["slug"], entry["mode"], entry["phrase"]
    board = entry.get("board") or HIGH_SCORES
    target = entry.get("target") or ""
    known_ban = hit_ban(phrase)
    if known_ban:
        raise BannedPhrase(phrase, known_ban)
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
    objective = objective_for(live, unit=game_mode.unit, board=board)
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
    ours = max(
        (r for r in rows if me and r.get("userId") == me),
        key=lambda r: objective.leader_key(Leader(float(r["probability"]), int(r["wordCount"]))),
        default=None,
    )
    estimate = float(entry["p_mean"])
    gamble = objective.shortest and bool(entry.get("gamble")) and (leader is None or units < leader.units)
    aim = max(estimate, objective.entry_p(entry, leader)) if gamble else estimate
    where = f"{board} for {target!r}" if target else board
    lead = f"{leader.probability:.2f}/{leader.units}{u} {leader.name}" if leader else "empty board"
    log(
        f"  [{where}] live leader {lead}; ours {ours['probability']:.2f}/{ours['wordCount']}{u}"
        if ours
        else f"  [{where}] live leader {lead}"
    )
    if ours and objective.leader_key(Leader(float(ours["probability"]), int(ours["wordCount"]))) >= objective.key(
        aim, units
    ):
        vault.set_status(
            slug, mode, phrase, "published", board, target=target, detail="our board row is already as good"
        )
        return "already"
    if objective.shortest and not game_mode.golf:
        log("  rebuilding the chain is safe: each board keeps our best-ever row, so Highest stays as it is")
    if not objective.beats(aim, units, leader):
        raise PublishError(f"no longer beats the live leader ({lead})")
    estimated = entry.get("estimated_from")
    if estimated:
        log(f"  estimated from {estimated}: posting the borrowed score without an oracle check")
    asked = max(0, rerolls)
    ceiling = asked
    if gamble:
        ceiling = max(asked, GAMBLE_REROLLS)
        log(
            f"  gamble: {entry.get('hits')} oracle rolls cleared {objective.threshold:.2f} (mean {estimate:.3f}); "
            f"skipping the oracle re-check"
        )
    elif entry.get("long_shot"):
        ceiling = max(asked, long_rerolls)
        log(f"  long shot: lower bound {entry['p_lcb']:.3f} does not beat {lead}; skipping the oracle re-check")
    elif verify and not estimated:
        fresh = asyncio.run(oracle_check(db, stored, phrase, target=target))
        log(f"  oracle re-check {fresh:.3f} (vault said {estimate:.3f})")
        fresh = settle_goal(db, stored, live, estimate, fresh, dry_run, log)
        if not objective.beats(fresh, units, leader):
            raise PublishError(f"oracle re-check {fresh:.3f} no longer beats {lead}")
    need = leader
    if dry_run:
        what = f"send {units} characters" if game_mode.golf else f"build {units} words"
        log(f"  dry run: would {what}: {phrase}")
        return "dry-run"

    picked = ""
    shown_p: float | None = None
    if game_mode.golf:
        best = golf_try(client, live, words[0], log, target)
        log(f"    = {units}c {best:.2f}")
    else:
        attempt = client.attempt(live["revisionId"], mode) or client.start(live["revisionId"], mode)
        attempt, server_p = build_chain(client, attempt, words, log)
        turn = last_turn(attempt)
        picked = str((turn or {}).get("choice") or "")
        shown_p = server_p
        best = credited(turn, target) if target else (server_p if server_p is not None else 0.0)
    tries = 0
    if asked:
        how = "re-scoring the phrase" if game_mode.golf else "re-rolling the last word"
        extra = f", then up to {ceiling} while short" if ceiling > asked else ""
        log(f"  {how} {asked} time(s); the board keeps the best score{extra}")
    while should_reroll(tries, asked, ceiling, objective.beats(best, units, need)):
        tries += 1
        if game_mode.golf:
            p = golf_try(client, live, words[0], log, target)
        else:
            attempt, raw = reroll(client, attempt, log)
            turn = last_turn(attempt)
            picked = str((turn or {}).get("choice") or "")
            shown_p = raw
            p = credited(turn, target) if target else raw
        best = max(best, p)
        log(f"    re-roll {tries}: {p:.2f} (best {best:.2f})")
    won = objective.beats(best, units, need)
    missed = bool(target and picked and picked != target and not won)
    if missed and shown_p is not None:
        log(
            f"  Jev picked {picked!r} ({shown_p:.2f}), not {target!r}; "
            "the row lands on that answer, so this board stays empty"
        )
    db.execute(
        "INSERT INTO publishes (slug, mode, phrase, estimate, server, status, detail, at, board) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            slug,
            mode,
            phrase,
            estimate,
            best,
            "won" if won else "short",
            json.dumps({"rerolls": tries} | ({"target": target} if target else {})),
            now_iso(),
            board,
        ),
    )
    if not target:
        # The site's own score for the line; later searches and triage learn from it.
        db.execute(
            "INSERT OR REPLACE INTO site_scores VALUES (?, ?, ?, ?, ?, ?)",
            (slug, mode, phrase, best, "publish", now_iso()),
        )
    if game_mode.golf:
        time.sleep(GOLF_BOARD_LAG)
    refresh_board(db, client, live, mode)
    if won:
        detail = f"rerolls={tries}"
    elif missed and shown_p is not None:
        detail = f"Jev picked {picked!r} ({shown_p:.2f}), not {target!r}"
    else:
        detail = f"server {best:.2f} after {tries} re-rolls did not beat {lead}"
    # The site score replaces the local estimate, so a later submit does not treat the estimate as still best.
    vault.apply_site_score(slug, mode, phrase, best, "published" if won else "failed", board, target, detail)
    if won:
        return f"won {site_round(best):.2f}/{units}{u}"
    if missed and shown_p is not None:
        return f"missed {target}; Jev picked {picked} ({shown_p:.2f})"
    return f"short {best:.2f}"


def sweep_banned(log, indent: str = "") -> list[dict]:
    """Delete vault lines that contain a named ban, on every question and board."""
    removed = vault.drop_banned()
    if removed:
        shown = "; ".join(e["phrase"] for e in removed[:5])
        more = f" (+{len(removed) - 5})" if len(removed) > 5 else ""
        log(f"{indent}removed {len(removed)} vault line(s) containing a banned phrase: {shown}{more}")
    return removed


def ban_fall_through(entry: dict, used: int, log) -> dict | None:
    """Queue the next vault line that does not contain a phrase the server has named."""
    board = entry.get("board") or HIGH_SCORES
    target = entry.get("target") or ""
    if used >= MAX_FALLBACKS:
        log(f"  {used} fall-throughs on this board already; leaving the rest for the next run")
        return None
    data = vault.load(entry["slug"], entry["mode"], board, target)
    pool = [
        e
        for e in data["entries"]
        if e.get("status") == "candidate"
        and (e.get("beats") or e.get("gamble"))
        and e.get("phrase") != entry["phrase"]
        and not hit_ban(e["phrase"])
        and not any(rejected.is_rejected(w) for w in e["phrase"].split())
    ]
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


def reject_and_fall_through(entry: dict, error: WordRejected, used: int, log) -> dict | None:
    """Remember the rejected word, retire the line, and queue the next vault line for the same board."""
    board = entry.get("board") or HIGH_SCORES
    target = entry.get("target") or ""
    rejected.add(error.word, entry["slug"], str(error))
    vault.set_status(entry["slug"], entry["mode"], entry["phrase"], "rejected", board, target=target, detail=str(error))
    if used >= MAX_FALLBACKS:
        log(f"  {used} fall-throughs on this board already; leaving the rest for the next run")
        return None
    data = vault.load(entry["slug"], entry["mode"], board, target)
    pool = [
        e
        for e in data["entries"]
        if e.get("status") == "candidate"
        and (e.get("beats") or e.get("gamble"))
        and not any(rejected.is_rejected(w) for w in e["phrase"].split())
    ]
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


def publish(
    dry_run: bool = False,
    rerolls: int = 3,
    verify: bool = True,
    slugs: list[str] | None = None,
    delay: float = 0.01,
    log=print,
    entries: list[dict] | None = None,
    on_result=None,
    long_rerolls: int = 5,
    board: str | None = None,
    mode: str | None = None,
    crosspost: bool = True,
    halt=None,
) -> int:
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
            entries = [
                e for e in vault.queued(board) if (not slugs or e["slug"] in slugs) and (not mode or e["mode"] == mode)
            ]
        if not entries:
            if not dry_run:
                sweep_banned(log)
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
        wiped: dict[str, set[str]] = {}

        def stopped() -> bool:
            return bool(halt and halt())

        for entry in entries:
            if stopped():
                log("stopped")
                break
            if entry["phrase"] in wiped.get(entry["slug"], ()):
                log(f"\n== {entry['slug']}: skipped {entry['phrase']}")
                log("  removed with the other lines scored against the wrong goal")
                continue
            game_mode = from_board(entry.get("board"), entry.get("mode"))
            aimed = f" -> {entry['target']}" if entry.get("target") else ""
            log(
                f"\n== {entry['slug']} [{game_mode.label}{aimed}] "
                f"{entry['p_mean']:.3f}/{entry['units']}{game_mode.unit_abbr}: {entry['phrase']}"
            )
            try:
                result = publish_entry(db, client, entry, questions, me, dry_run, rerolls, verify, log, long_rerolls)
                log(f"  -> {result}")
            except BannedPhrase as error:
                result = f"banned: {error}"
                log(f"  -> {result}")
                if error.named:
                    log(f"  recorded {error.named!r}; its words stay usable")
                else:
                    log("  server did not name the span; not adding a ban")
                if not dry_run:
                    if error.named:
                        sweep_banned(log, "  ")
                    vault.set_status(
                        entry["slug"],
                        entry["mode"],
                        entry["phrase"],
                        "failed",
                        entry.get("board") or HIGH_SCORES,
                        target=entry.get("target") or "",
                        detail=str(error),
                    )
                    key = board_key(entry)
                    fallback = ban_fall_through(entry, fallbacks.get(key, 0), log)
                    if fallback is None:
                        failures += 1
                    elif stopped():
                        log("  stopped; the next line stays queued")
                    else:
                        fallbacks[key] = fallbacks.get(key, 0) + 1
                        seen.add(key + (fallback["phrase"],))
                        entries.append(fallback)
                else:
                    failures += 1
            except WordRejected as error:
                result = f"rejected: {error}"
                log(f"  -> {result}")
                if not dry_run:
                    key = board_key(entry)
                    fallback = reject_and_fall_through(entry, error, fallbacks.get(key, 0), log)
                    if fallback is None:
                        failures += 1
                    elif stopped():
                        log("  stopped; the next line stays queued")
                    else:
                        fallbacks[key] = fallbacks.get(key, 0) + 1
                        seen.add(key + (fallback["phrase"],))
                        entries.append(fallback)
                else:
                    failures += 1
            except WrongGoal as error:
                failures += 1
                result = f"failed: {error}"
                log(f"  -> {result}")
                wiped.setdefault(error.slug, set()).update(error.phrases)
            except (PublishError, SiteError) as error:
                failures += 1
                result = f"failed: {error}"
                log(f"  -> {result}")
                if not dry_run:
                    vault.set_status(
                        entry["slug"],
                        entry["mode"],
                        entry["phrase"],
                        "failed",
                        entry.get("board") or HIGH_SCORES,
                        target=entry.get("target") or "",
                        detail=str(error),
                    )
            if on_result:
                on_result(entry, result)
            landed = result.startswith("won") or result == "already"
            if crosspost and not dry_run and landed and entry["mode"] == STRICT and entry["slug"] not in crossposted:
                crossposted.add(entry["slug"])
                followups = golf_followups(db, entry["slug"], log)
                if stopped():
                    log("  stopped; Golf lines stay queued")
                    break
                for golf in followups:
                    key = board_key(golf) + (golf["phrase"],)
                    if key not in seen:
                        seen.add(key)
                        entries.append(golf)
            elif stopped():
                log("stopped")
                break
        if not dry_run:
            sweep_banned(log)
        client.save_session()
        return 1 if failures else 0
