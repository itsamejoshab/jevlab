"""Plant decoy words on boards we already lead.

The site's words list is public, per play mode, and shared by Highest and Shortest.
Rivals mine the top of it. Defense posts the question's own words and a few local
synonyms first, then other boards' strongest words, skipping function words, so
that list stops being a copy of our winning phrase. The board keeps every row we
post, so a decoy that takes a lead stays there.
"""

from __future__ import annotations

import fcntl
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .db import DB
from .modes import CASUAL, HIGH_SCORES, SHORTEST_YES, STRICT
from .objective import Leader, board_leader, objective_for
from .publish import LOCK_PATH, WordRejected, turn_with_retry
from .rules import RuleError, check_word, drop_clashing, option_clash
from .rules.banned import BannedPhrase
from .site import SiteClient, SiteError, active_words

# A board this full, whose headline is mostly other people's words, is already noisy.
CROWD_WORDS = 40
# ours_in_top * QUIET_SHARE <= headline length means our words are at most a quarter of the top list.
QUIET_SHARE = 4

READY = "ready"
QUIET = "quiet"
BARE = "bare"
BLIND = "blind"
_ORDER = {READY: 0, QUIET: 1, BARE: 2, BLIND: 3}

# Synonyms kept per question word. WordNet's long tail is mostly other senses.
SYNONYMS_PER_WORD = 3
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_wordnet = None

# Function words. They rank at the top of almost every board, so planting them does not hide our phrase.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at
    be because been before being below between both but by
    can could did do does doing down during
    each few for from further
    had has have having he her here hers herself him himself his how
    i if in into is it its itself just
    may me might more most must my myself
    no nor not of off on once only or other our ours ourselves out over own
    same shall she should so some such
    than that the their theirs them themselves then there these they this those through to too
    under until up very
    was we were what when where which while who whom why will with without would
    yes you your yours yourself yourselves
    """.split()
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fold(word: str) -> str:
    return word.strip().casefold()


def sentence_words(text: str) -> list[str]:
    """Content words of the question, in order. An apostrophe splits, so grandpa's stays grandpa."""
    out: list[str] = []
    seen: set[str] = set()
    for part in _TOKEN.findall(text):
        word = part.lower()
        if len(word) < 3 or word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def _thesaurus():
    """WordNet, downloaded once into the usual NLTK directory. Missing data means no synonyms."""
    global _wordnet
    if _wordnet is not None:
        return _wordnet
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("word")
    except LookupError:
        import nltk

        nltk.download("wordnet", quiet=True)
        try:
            wn.synsets("word")
        except LookupError:
            return None
    _wordnet = wn
    return wn


def synonyms(word: str, limit: int = SYNONYMS_PER_WORD) -> list[str]:
    """Up to `limit` single-word WordNet synonyms, common senses first."""
    wn = _thesaurus()
    if wn is None:
        return []
    scored: dict[str, int] = {}
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name()
            if "_" in name or not name.isalpha():
                continue
            key = name.lower()
            if key == word or key in STOPWORDS or len(key) < 3 or len(key) > 16:
                continue
            scored[key] = max(scored.get(key, 0), lemma.count())
    return sorted(scored, key=lambda key: (-scored[key], key))[:limit]


def topic_decoys(text: str) -> list[str]:
    """The question's words, then a few synonyms of each."""
    words = sentence_words(text)
    out = list(words)
    seen = set(words)
    for word in words:
        for syn in synonyms(word):
            if syn in seen:
                continue
            seen.add(syn)
            out.append(syn)
    return out


def words_of(phrases: list[str]) -> set[str]:
    return {fold(word) for phrase in phrases for word in phrase.split() if fold(word)}


def present_words(impacts: list[dict]) -> set[str]:
    return {fold(str(row.get("word") or "")) for row in impacts if fold(str(row.get("word") or ""))}


def lexicon(groups: list[list[dict]]) -> list[str]:
    """One casing per word. Words that help on more boards come first, then the stronger impact."""
    stats: dict[str, dict] = {}
    for rows in groups:
        seen: set[str] = set()
        for row in rows:
            word = str(row.get("word") or "").strip()
            key = fold(word)
            if not key or key in seen:
                continue
            seen.add(key)
            impact = float(row.get("averageImpact") or 0)
            slot = stats.get(key)
            if slot is None:
                stats[key] = {"word": word, "boards": 1, "impact": impact}
                continue
            slot["boards"] += 1
            if impact > slot["impact"]:
                slot["impact"] = impact
                slot["word"] = word
    ranked = sorted(stats.values(), key=lambda slot: (-slot["boards"], -slot["impact"], slot["word"].casefold()))
    return [slot["word"] for slot in ranked]


def headline(impacts: list[dict], top_n: int) -> list[str]:
    """The words a rival sees first, by average impact."""
    ranked = sorted(
        impacts,
        key=lambda row: (-float(row.get("averageImpact") or 0), fold(str(row.get("word") or ""))),
    )
    top: list[str] = []
    seen: set[str] = set()
    for row in ranked:
        key = fold(str(row.get("word") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        top.append(key)
        if len(top) >= top_n:
            break
    return top


def already_quiet(ours_in_top: int, top_len: int, word_count: int) -> bool:
    """True when the public list is long and our winning words are already a minority of the headline."""
    if word_count < CROWD_WORDS or top_len == 0:
        return False
    return ours_in_top * QUIET_SHARE <= top_len


def pick_decoys(
    ranked: list[str],
    present: set[str],
    protected: set[str],
    top_n: int,
    choices: list | None = None,
) -> list[str]:
    """Up to `top_n` new words. Skips our phrase, words already on the board, stop words, and illegal tokens."""
    out: list[str] = []
    seen: set[str] = set()
    for word in ranked:
        key = fold(word)
        if not key or key in seen or key in present or key in protected or key in STOPWORDS:
            continue
        try:
            clean = check_word(word)
        except RuleError:
            continue
        if option_clash(clean, choices):
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= top_n:
            break
    return out


def safe_to_clear(active: list[str], phrases: list[str]) -> bool:
    """Empty, a single leftover word, or a prefix of a phrase already on our board."""
    if len(active) <= 1:
        return True
    for phrase in phrases:
        words = phrase.split()
        if active == words[: len(active)]:
            return True
    return False


def _hold(rows: list[dict], me: str, question: dict, board: str) -> tuple[bool, str]:
    """Whether our row leads `board`, and the phrase sitting on it."""
    if not me:
        return False, ""
    objective = objective_for(question, board=board)
    leader = board_leader(rows, me, board=board)
    mine = [row for row in rows or [] if row.get("userId") == me and row.get("probability") is not None]
    ours = board_leader(mine, me, include_ours=True, board=board)
    if not ours or not (leader is None or objective.leader_key(ours) > objective.leader_key(leader)):
        return False, ""
    phrase = ""
    best: tuple | None = None
    for row in mine:
        cand = Leader(float(row["probability"]), int(row.get("wordCount") or 0), row.get("name") or "", True)
        key = objective.leader_key(cand)
        if best is None or key > best:
            best = key
            phrase = str(row.get("phrase") or "")
    return True, phrase


@dataclass(frozen=True)
class Plan:
    status: str
    decoys: tuple[str, ...]
    ours_in_top: int
    top_len: int
    word_count: int
    phrases: tuple[str, ...]


def plan_board(
    impacts: list[dict],
    ranked: list[str],
    phrases: list[str],
    posted: set[str],
    top_n: int,
    choices: list | None,
    question: str = "",
) -> Plan:
    protected = words_of(phrases)
    public = present_words(impacts)
    top = headline(impacts, top_n)
    ours_in_top = sum(1 for word in top if word in protected)
    decoys = pick_decoys(
        topic_decoys(question) + ranked,
        public | {fold(word) for word in posted},
        protected,
        top_n,
        choices,
    )
    kept = [phrase for phrase in phrases if phrase.split()]
    if not kept:
        status = BLIND
    elif already_quiet(ours_in_top, len(top), len(public)):
        status = QUIET
    elif not decoys:
        status = BARE
    else:
        status = READY
    return Plan(status, tuple(decoys), ours_in_top, len(top), len(public), tuple(kept))


@dataclass(frozen=True)
class DefenseTarget:
    slug: str
    title: str
    play_mode: str
    held: tuple[str, ...]
    phrase: str
    plan: Plan

    @property
    def key(self) -> str:
        return f"{self.play_mode}:{self.slug}"

    @property
    def status(self) -> str:
        return self.plan.status

    @property
    def decoys(self) -> tuple[str, ...]:
        return self.plan.decoys

    @property
    def selected(self) -> bool:
        return self.status == READY


def _impacts(db: DB, mode: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in db.all("SELECT slug, rows FROM word_impacts WHERE mode = ?", (mode,)):
        out[row["slug"]] = json.loads(row["rows"]) if row["rows"] else []
    return out


def _posted(db: DB, mode: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in db.all("SELECT slug, word FROM defense_words WHERE mode = ?", (mode,)):
        out.setdefault(row["slug"], set()).add(row["word"])
    return out


def scan(db: DB, top_n: int = 20, include_casual: bool = False) -> list[DefenseTarget]:
    """Boards we lead, from the snapshot. Ready rows are the ones worth posting on."""
    me = db.me()
    questions = db.questions()
    modes = (STRICT, CASUAL) if include_casual else (STRICT,)
    found: list[DefenseTarget] = []
    for mode in modes:
        impacts = _impacts(db, mode)
        ranked = lexicon(list(impacts.values()))
        posted = _posted(db, mode)
        boards = db.boards_for(mode)
        for slug, question in questions.items():
            choices = list((question.get("raw") or {}).get("choices") or [])
            high_rows = boards.get((slug, HIGH_SCORES), [])
            short_rows = boards.get((slug, SHORTEST_YES), [])
            high_lead, high_phrase = _hold(high_rows, me, question, HIGH_SCORES)
            short_lead, short_phrase = _hold(short_rows, me, question, SHORTEST_YES)
            if not high_lead and not short_lead:
                continue
            phrases = [
                phrase for lead, phrase in ((high_lead, high_phrase), (short_lead, short_phrase)) if lead and phrase
            ]
            decided = plan_board(
                impacts.get(slug, []),
                ranked,
                phrases,
                posted.get(slug, set()),
                top_n,
                choices,
                question.get("title") or question.get("instructions") or "",
            )
            held = tuple(name for lead, name in ((high_lead, "Highest"), (short_lead, "Shortest")) if lead)
            found.append(
                DefenseTarget(
                    slug,
                    question.get("title") or slug,
                    mode,
                    held,
                    high_phrase if high_lead and high_phrase else short_phrase,
                    decided,
                )
            )
    return sorted(
        found,
        key=lambda target: (_ORDER.get(target.status, 9), -target.plan.ours_in_top, target.slug, target.play_mode),
    )


def _remember(db: DB, slug: str, mode: str, word: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO defense_words (slug, mode, word, at) VALUES (?, ?, ?, ?)",
        (slug, mode, fold(word), _now()),
    )
    row = db.one("SELECT snapshot_id, rows FROM word_impacts WHERE slug = ? AND mode = ?", (slug, mode))
    rows = json.loads(row["rows"]) if row and row["rows"] else []
    if fold(word) in present_words(rows):
        return
    rows.append({"word": word, "averageImpact": 0.0, "tries": 1})
    snapshot_id = row["snapshot_id"] if row else 0
    db.execute(
        "INSERT OR REPLACE INTO word_impacts (slug, mode, snapshot_id, rows) VALUES (?, ?, ?, ?)",
        (slug, mode, snapshot_id, json.dumps(rows)),
    )


def _live_question(raw: dict) -> dict:
    return {
        "goal": raw.get("goal") or "yes",
        "kind": raw.get("kind") or "noul",
        "yesThreshold": raw.get("yesThreshold"),
        "slug": raw.get("slug") or "",
    }


def _post_words(client: SiteClient, attempt: dict, words: list[str], log, halt) -> tuple[dict, list[str]]:
    posted: list[str] = []
    for word in words:
        if halt and halt():
            log("  stopped")
            break
        if active_words(attempt):
            attempt, _turn = turn_with_retry(client, attempt, {"kind": "clear"}, log)
        try:
            attempt, turn = turn_with_retry(client, attempt, {"kind": "append", "text": word}, log)
        except WordRejected as error:
            log(f"    {word} rejected ({error})")
            continue
        except BannedPhrase as error:
            log(f"    {word} banned ({error})")
            continue
        except SiteError as error:
            log(f"    {word} failed ({error})")
            break
        log(f"    + {word:<16} {float(turn.get('probability') or 0):.2f}")
        posted.append(word)
    if active_words(attempt):
        try:
            attempt, _turn = turn_with_retry(client, attempt, {"kind": "clear"}, log)
        except SiteError as error:
            log(f"    clear failed ({error})")
    return attempt, posted


@dataclass(frozen=True)
class DefenseRun:
    boards: int
    posted: int
    skipped: int
    planned: int = 0


def defend(
    jobs: list[tuple[str, str]],
    top_n: int = 20,
    dry_run: bool = False,
    log=print,
    halt=None,
    db: DB | None = None,
    client: SiteClient | None = None,
    lock_path=None,
) -> DefenseRun:
    """Post one-word decoys for each (slug, play mode). Live words and the live lead are re-checked first."""
    db = db or DB()
    path = lock_path or LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another site writer is running")
            return DefenseRun(0, 0, len(jobs))
        client = client or SiteClient(delay=0.05)
        viewer = client.viewer()
        me = ((viewer.get("player") or {}).get("user") or {}).get("id") or ""
        signed_in = bool(me)
        if not me:
            me = db.me()
        if not me or (not dry_run and not signed_in):
            log("not signed in: put a session cookie in session.json or JEV_COOKIE")
            return DefenseRun(0, 0, len(jobs))
        menu = {q["slug"]: q for q in client.menu().get("questions") or []}
        posted_words = 0
        touched = 0
        skipped = 0
        planned = 0
        ranked_by_mode: dict[str, list[str]] = {}
        for index, (slug, mode) in enumerate(jobs, start=1):
            if halt and halt():
                log("stopped")
                skipped += len(jobs) - index + 1
                break
            live = menu.get(slug)
            if live is None:
                log(f"\n== {slug} [{mode}] question is gone")
                skipped += 1
                continue
            try:
                payload = client.leaderboards(live["revisionId"], mode).get("leaderboards") or {}
                impacts = [row for row in (client.words(live["revisionId"], mode) or []) if isinstance(row, dict)]
            except SiteError as error:
                log(f"\n== {slug} [{mode}] {error}")
                skipped += 1
                continue
            question = _live_question(live)
            choices = list(live.get("choices") or [])
            high_rows = drop_clashing(list(payload.get(HIGH_SCORES) or []), choices, me)
            short_rows = drop_clashing(list(payload.get(SHORTEST_YES) or []), choices, me)
            high_lead, high_phrase = _hold(high_rows, me, question, HIGH_SCORES)
            short_lead, short_phrase = _hold(short_rows, me, question, SHORTEST_YES)
            where = " ".join(name for lead, name in ((high_lead, "Highest"), (short_lead, "Shortest")) if lead)
            log(f"\n== {slug} [{mode}] {where or 'not leading'}")
            if not high_lead and not short_lead:
                log("  no longer leading; skipping")
                skipped += 1
                continue
            phrases = [
                phrase for lead, phrase in ((high_lead, high_phrase), (short_lead, short_phrase)) if lead and phrase
            ]
            if mode not in ranked_by_mode:
                ranked_by_mode[mode] = lexicon(list(_impacts(db, mode).values()))
            decided = plan_board(
                impacts,
                ranked_by_mode[mode],
                phrases,
                _posted(db, mode).get(slug, set()),
                top_n,
                choices,
                live.get("title") or live.get("instructions") or "",
            )
            if decided.status == BLIND:
                log("  leading phrase is missing; skipping so we do not repost our own words")
                skipped += 1
                continue
            if not decided.decoys:
                log("  nothing new to add")
                skipped += 1
                continue
            if decided.status == QUIET:
                log(
                    f"  crowded ({decided.word_count} words, {decided.ours_in_top}/{decided.top_len} ours in the top); posting anyway"
                )
            preview = " ".join(decided.decoys[:12])
            extra = f" +{len(decided.decoys) - 12}" if len(decided.decoys) > 12 else ""
            if dry_run:
                log(f"  dry run: would post {len(decided.decoys)}: {preview}{extra}")
                planned += 1
                continue
            attempt = client.attempt(live["revisionId"], mode) or client.start(live["revisionId"], mode)
            active = active_words(attempt)
            if not safe_to_clear(active, phrases):
                log(f"  active chain is {' '.join(active)!r}; leaving it alone")
                skipped += 1
                continue
            if len(active) > 1:
                log("  clearing the active chain; the board keeps our best row")
            try:
                _attempt, planted = _post_words(client, attempt, list(decided.decoys), log, halt)
            except SiteError as error:
                log(f"  {error}")
                skipped += 1
                continue
            for word in planted:
                _remember(db, slug, mode, word)
            if planted:
                touched += 1
                posted_words += len(planted)
                log(f"  posted {len(planted)}")
            else:
                skipped += 1
        return DefenseRun(touched, posted_words, skipped, planned)
