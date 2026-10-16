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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from .db import DB
from .modes import CASUAL, GOLF, HIGH_SCORES, SHORTEST_YES, STRICT
from .objective import Leader, board_leader, objective_for
from .publish import LOCK_PATH, SENTENCE_WORDS, TURBO_BOARDS, WordRejected, turn_with_retry
from .rules import RuleError, check_word, drop_clashing, option_clash
from .rules.banned import BannedPhrase, hit, tokens
from .site import SiteClient, SiteError, active_words

# A board this full is long enough to judge.
CROWD_WORDS = 40
# answer words * QUIET_SHARE <= length means our answer is at most a quarter of that slice.
QUIET_SHARE = 4
# How many words of the public list count as the headline rivals read first.
HEADLINE_WORDS = 20
# Extra decoys so skipping a banned word still leaves the list quiet.
BAN_SPARE = 5

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


def already_quiet(ours_in_top: int, top_len: int, word_count: int, answer_words: int | None = None) -> bool:
    """True when a long public list no longer gives the winning phrase away.

    The headline test catches a top slice that is already mostly other words.
    The whole-list test is what stops a solo board: decoys score low, so they never
    enter that top slice, but a list that is at least four times the answer is enough cover.
    """
    if word_count < CROWD_WORDS:
        return False
    if top_len and ours_in_top * QUIET_SHARE <= top_len:
        return True
    if answer_words is None:
        return False
    return answer_words * QUIET_SHARE <= word_count


def decoys_needed(answer_words: int, word_count: int) -> int:
    """New words that bring this list up to the quiet line in one pass, plus a few spares.

    A banned word is left out of the submission. The spares are what still lets that
    one pass take the board off the list.
    """
    cover = max(CROWD_WORDS, answer_words * QUIET_SHARE)
    return max(0, cover - word_count) + BAN_SPARE


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
        if not key or key in seen or key in present or key in protected or key in STOPWORDS or hit(key):
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
    answer_words: int = 0
    need: int = 0


def plan_board(
    impacts: list[dict],
    ranked: list[str],
    phrases: list[str],
    posted: set[str],
    choices: list | None,
    question: str = "",
) -> Plan:
    protected = words_of(phrases)
    public = present_words(impacts)
    top = headline(impacts, HEADLINE_WORDS)
    ours_in_top = sum(1 for word in top if word in protected)
    answer_words = sum(1 for word in public if word in protected)
    quiet = already_quiet(ours_in_top, len(top), len(public), answer_words)
    need = 0 if quiet else decoys_needed(answer_words, len(public))
    decoys = pick_decoys(
        topic_decoys(question) + ranked,
        public | {fold(word) for word in posted},
        protected,
        need,
        choices,
    )
    kept = [phrase for phrase in phrases if phrase.split()]
    if not kept:
        status = BLIND
    elif quiet:
        status = QUIET
    elif not decoys:
        status = BARE
    else:
        status = READY
    return Plan(status, tuple(decoys), ours_in_top, len(top), len(public), tuple(kept), answer_words, need)


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


def scan(db: DB, include_casual: bool = False, include_golf: bool = False) -> list[DefenseTarget]:
    """Boards we lead, from the snapshot. Ready rows are the ones worth posting on."""
    me = db.me()
    questions = db.questions()
    modes = [STRICT]
    if include_golf:
        modes.append(GOLF)
    if include_casual:
        modes.append(CASUAL)
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


def _drop_span(words: list[str], span: str) -> list[str]:
    """Remove the first occurrence of a banned span. Its other words stay."""
    needle = tokens(span)
    if not needle:
        return list(words)
    kept = list(words)
    width = len(needle)
    for index in range(len(kept) - width + 1):
        if tokens(" ".join(kept[index : index + width])) == needle:
            del kept[index : index + width]
            return kept
    return kept


def without_known_bans(words: list[str]) -> list[str]:
    """Drop every stored banned word or phrase. Single words of a longer ban stay."""
    kept = list(words)
    while kept:
        span = hit(" ".join(kept))
        if not span:
            return kept
        shorter = _drop_span(kept, span)
        if len(shorter) == len(kept):
            return kept
        kept = shorter
    return kept


def _probe_until_ban(
    client: SiteClient,
    attempt: dict,
    chunk: list[str],
    rest: list[str],
    posted: list[str],
    log,
    halt,
) -> tuple[dict, list[str]]:
    """Send `chunk` one word at a time until a ban, skip that word, and hand back the rest."""
    for index, word in enumerate(chunk):
        if halt and halt():
            log("  stopped")
            return attempt, []
        try:
            attempt, _turn = turn_with_retry(client, attempt, {"kind": "append", "text": word}, log)
        except BannedPhrase as error:
            log(f"  skip {word} ({error})")
            return attempt, chunk[index + 1 :] + rest
        except WordRejected as error:
            log(f"  rejected ({error})")
            return attempt, []
        except SiteError as error:
            log(f"  {error}")
            return attempt, []
        posted.append(word)
    return attempt, rest


def _post_words(
    client: SiteClient, attempt: dict, words: list[str], log, halt, *, word_by_word: bool = False
) -> tuple[dict, list[str]]:
    """Append the decoys, skip anything banned, then clear. The score on the reply is ignored.

    The site's words table is one row per append, so a sentence lands as a single row.
    `word_by_word` sends each decoy alone, which is what splits that table into one row per word.

    A ban we already know is left out. A ban the server does not name is found by sending
    the sentence one word at a time, and that word is skipped.
    """
    pending = without_known_bans(words)
    posted: list[str] = []
    step = 1 if word_by_word else SENTENCE_WORDS
    while pending:
        if halt and halt():
            log("  stopped")
            break
        chunk = pending[:step]
        rest = pending[step:]
        text = " ".join(chunk)
        try:
            attempt, _turn = turn_with_retry(client, attempt, {"kind": "append", "text": text}, log)
        except WordRejected as error:
            log(f"  rejected ({error})")
            break
        except BannedPhrase as error:
            if error.named:
                log(f"  banned {error.named!r}; omitting it")
                shorter = without_known_bans(_drop_span(chunk + rest, error.named))
                if len(shorter) < len(chunk) + len(rest):
                    pending = shorter
                    continue
            if word_by_word:
                log(f"  skip {chunk[0]} ({error})")
                pending = rest
                continue
            log("  banned, span not named; submitting one word at a time")
            attempt, pending = _probe_until_ban(client, attempt, chunk, rest, posted, log, halt)
            continue
        except SiteError as error:
            log(f"  {error}")
            break
        posted.extend(chunk)
        pending = rest
    if posted and active_words(attempt):
        try:
            attempt, _turn = turn_with_retry(client, attempt, {"kind": "clear"}, log)
        except SiteError as error:
            log(f"  clear failed ({error})")
    return attempt, posted


@dataclass(frozen=True)
class DefenseRun:
    boards: int
    posted: int
    skipped: int
    planned: int = 0

    def __add__(self, other: DefenseRun) -> DefenseRun:
        return DefenseRun(
            self.boards + other.boards,
            self.posted + other.posted,
            self.skipped + other.skipped,
            self.planned + other.planned,
        )


def _tagged(log, index: int, total: int):
    """Prefix each line so overlapping boards can still be told apart."""
    mark = f"[{index}/{total}]"

    def write(message: str) -> None:
        lines = str(message).splitlines()
        if not lines:
            log(message)
            return
        log("\n".join(f"{mark} {line}" if line.strip() else line for line in lines))

    return write


def _defend_one(
    db: DB,
    client: SiteClient,
    menu: dict,
    me: str,
    ranked: list[str],
    slug: str,
    mode: str,
    dry_run: bool,
    word_by_word: bool,
    log,
    halt,
) -> DefenseRun:
    """Post decoys for one board. The caller decides whether to start it."""
    live = menu.get(slug)
    if live is None:
        log(f"\n== {slug} [{mode}] question is gone")
        return DefenseRun(0, 0, 1)
    try:
        payload = client.leaderboards(live["revisionId"], mode).get("leaderboards") or {}
        impacts = [row for row in (client.words(live["revisionId"], mode) or []) if isinstance(row, dict)]
    except SiteError as error:
        log(f"\n== {slug} [{mode}] {error}")
        return DefenseRun(0, 0, 1)
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
        return DefenseRun(0, 0, 1)
    phrases = [phrase for lead, phrase in ((high_lead, high_phrase), (short_lead, short_phrase)) if lead and phrase]
    decided = plan_board(
        impacts,
        ranked,
        phrases,
        _posted(db, mode).get(slug, set()),
        choices,
        live.get("title") or live.get("instructions") or "",
    )
    if decided.status == BLIND:
        log("  leading phrase is missing; skipping so we do not repost our own words")
        return DefenseRun(0, 0, 1)
    if not decided.decoys:
        log("  nothing new to add")
        return DefenseRun(0, 0, 1)
    if decided.status == QUIET:
        log(f"  crowded ({decided.answer_words} of {decided.word_count} listed words are our answer); skipping")
        return DefenseRun(0, 0, 1)
    sentence = " ".join(decided.decoys)
    if dry_run:
        log(f"  {sentence}\n  dry run.")
        return DefenseRun(0, 0, 0, 1)
    how = " one word at a time" if word_by_word else ""
    log(f"  {sentence}\n  submitting...{how}")
    attempt = client.attempt(live["revisionId"], mode) or client.start(live["revisionId"], mode)
    try:
        _attempt, planted = _post_words(client, attempt, list(decided.decoys), log, halt, word_by_word=word_by_word)
    except SiteError as error:
        log(f"  {error}")
        return DefenseRun(0, 0, 1)
    for word in planted:
        _remember(db, slug, mode, word)
    if planted:
        log("  done.")
        return DefenseRun(1, len(planted), 0)
    return DefenseRun(0, 0, 1)


def defend(
    jobs: list[tuple[str, str]],
    dry_run: bool = False,
    log=print,
    halt=None,
    db: DB | None = None,
    client: SiteClient | None = None,
    lock_path=None,
    word_by_word: bool = False,
    turbo: bool = False,
) -> DefenseRun:
    """Post decoys for each (slug, play mode). Live words and the live lead are re-checked first.

    Decoys go up as a sentence unless `word_by_word` is set, which appends one word per turn.
    `turbo` posts up to TURBO_BOARDS boards at once. Words on one board stay in order.
    """
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
        modes = {mode for _, mode in jobs}
        ranked_by_mode = {mode: lexicon(list(_impacts(db, mode).values())) for mode in modes}
        if turbo:
            _thesaurus()

        def one(item: tuple[int, tuple[str, str]]) -> tuple[bool, DefenseRun]:
            index, (slug, mode) = item
            if halt and halt():
                return True, DefenseRun(0, 0, 1)
            board_log = _tagged(log, index, len(jobs)) if turbo else log
            return False, _defend_one(
                db,
                client,
                menu,
                me,
                ranked_by_mode[mode],
                slug,
                mode,
                dry_run,
                word_by_word,
                board_log,
                halt,
            )

        indexed = list(enumerate(jobs, start=1))
        if turbo and len(jobs) > 1:
            with ThreadPoolExecutor(min(TURBO_BOARDS, len(jobs))) as pool:
                outcomes = list(pool.map(one, indexed))
        else:
            outcomes = []
            for item in indexed:
                stopped, result = one(item)
                outcomes.append((stopped, result))
                if stopped:
                    break
        total = DefenseRun(0, 0, 0)
        stopped_any = False
        for stopped, result in outcomes:
            total += result
            stopped_any = stopped_any or stopped
        if stopped_any:
            skipped = total.skipped + (len(jobs) - len(outcomes))
            total = DefenseRun(total.boards, total.posted, skipped, total.planned)
            log("stopped")
        return total
