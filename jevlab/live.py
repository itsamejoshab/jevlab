"""Live round play: search the current site round and post on a fast or slow pace."""

from __future__ import annotations

import re

from .config import JEV_MODEL
from .modes import HIGH_SCORES
from .objective import Objective, board_leader, Leader
from .rules.banned import contains as phrase_banned


def live_slug(text: str) -> str:
    folded = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return f"live-{folded[:48] or 'question'}"


def live_question(text: str) -> dict:
    """Build a yes/no oracle question from typed text. Not a site question: no board, no vault."""
    title = " ".join(text.split())
    if not title:
        raise ValueError("type a question first")
    return {
        "slug": live_slug(title),
        "snapshot_id": None,
        "revision_id": "",
        "title": title,
        "instructions": title,
        "kind": "noul",
        "goal": "yes",
        "baseline": None,
        "yes_threshold": None,
        "model_version": "",
        "jev_request": {
            "state": "",
            "model": JEV_MODEL,
            "questions": {"q": {"type": "noul", "instructions": title}},
        },
        "raw": {"live": True},
    }


def finish_threshold(state: dict) -> float:
    for milestone in state.get("milestones") or []:
        if milestone.get("id") == "finish" and milestone.get("at") is not None:
            return float(milestone["at"])
    question = state.get("question") or {}
    if question.get("yesThreshold") is not None:
        return float(question["yesThreshold"])
    return 0.5


def carried_question(state: dict) -> dict | None:
    """Question named by this poll, including the gap after a round while the payload still has it."""
    question = state.get("question") or {}
    if not (question.get("title") or question.get("instructions") or question.get("slug")):
        return None
    return question_from_round(state)


def search_question(state: dict, previous: dict | None = None) -> dict | None:
    """What to search now. A gap that does not name a next question keeps the last one."""
    current = carried_question(state)
    if current is not None:
        return current
    if (state.get("status") or "") == "live":
        return None
    return previous


def round_goal(question: dict) -> str:
    """The goal the live payload names. A round that does not say is searched as P(yes)."""
    goal = str(question.get("goal") or "yes").casefold()
    return goal if goal in ("yes", "no") else "yes"


def question_from_round(state: dict) -> dict:
    """Engine-ready question for the site's live round. Search races that round's goal."""
    from .site.client import decode_jev_request

    q = state.get("question") or {}
    title = q.get("title") or q.get("instructions") or ""
    request = q.get("jevRequest") or q.get("jev_request")
    if isinstance(request, str):
        request = decode_jev_request(request)
    if not isinstance(request, dict):
        request = {
            "state": "",
            "model": JEV_MODEL,
            "questions": {"q": {"type": q.get("kind") or "noul", "instructions": title}},
        }
    return {
        "slug": q.get("slug") or live_slug(title),
        "snapshot_id": None,
        "revision_id": q.get("revisionId") or q.get("revision_id") or "",
        "title": title,
        "instructions": title,
        "kind": q.get("kind") or "noul",
        "goal": round_goal(q),
        "baseline": q.get("baselineProbability"),
        "yes_threshold": q.get("yesThreshold"),
        "model_version": q.get("modelVersion") or "",
        "jev_request": request,
        "raw": {**q, "live": True, "playMode": state.get("playMode") or "word_chain", "roundId": state.get("id")},
    }


OPENING_WORDS = (
    "please",
    "yes",
    "definitely",
    "absolutely",
    "obviously",
    "clearly",
    "honestly",
    "certainly",
    "exactly",
    "guaranteed",
)


def opening_word(question: dict | None = None) -> str:
    """One legal word to fire the instant a live round appears."""
    slug = (question or {}).get("slug") or (question or {}).get("title") or ""
    return OPENING_WORDS[sum(map(ord, slug)) % len(OPENING_WORDS)]


class LiveSession:
    """Play `/play?round=live`. Fast: first word immediately, then every score gain. Slow: only when not 1st."""

    def __init__(self, pace: str = "slow", chain: str = "casual", post: bool = True):
        self.pace = "fast" if pace == "fast" else "slow"
        self.chain = "strict" if chain == "strict" else "casual"
        self.post_enabled = post
        self.state: dict | None = None
        self.me = ""
        self.question: dict | None = None
        self.leader: Leader | None = None
        self.we_lead = False
        self.threshold = 0.5
        self.play_mode = "strict_chain" if self.chain == "strict" else "word_chain"
        self.round_id = ""
        self.status = ""
        self.posted = ""
        self.posted_p = 0.0
        self.posted_units = 0
        self.finished = False  # a turn already crossed this round's finish line; hold the chain
        self.objective = Objective("yes", "noul", board=HIGH_SCORES)

    def poll_interval(self) -> float:
        return 0.35 if self.pace == "fast" else 3.0

    def observe(self, state: dict, me: str = "") -> bool:
        """Fold in a round poll. True when the search question (round id or slug) changed."""
        self.state = state
        self.me = me
        self.status = state.get("status") or ""
        self.play_mode = "strict_chain" if self.chain == "strict" else "word_chain"
        self.threshold = finish_threshold(state)
        incoming = state.get("question") or {}
        if incoming:
            self.objective = Objective(round_goal(incoming), incoming.get("kind") or "noul", board=HIGH_SCORES)
        rows = list(state.get("leaders") or [])
        self.leader = board_leader(rows, me, board=HIGH_SCORES)
        ours = board_leader([r for r in rows if me and r.get("userId") == me], me, include_ours=True, board=HIGH_SCORES)
        self.we_lead = bool(ours) and (
            self.leader is None or self.objective.leader_key(ours) >= self.objective.leader_key(self.leader)
        )
        if self.status != "live" or not state.get("question"):
            return False
        question = question_from_round(state)
        new_id = state.get("id") or ""
        changed = self.question is None or question["slug"] != self.question["slug"] or new_id != self.round_id
        if changed:
            self.posted = ""
            self.posted_p = 0.0
            self.posted_units = 0
            self.finished = False
        self.question = question
        self.round_id = new_id
        return changed

    def should_post(self, p: float, units: int, phrase: str = "") -> bool:
        if not self.post_enabled or self.status != "live" or self.finished:
            return False
        if phrase and phrase == self.posted:
            return False
        if self.pace == "fast":
            if not self.posted:
                return True
            return self.objective.key(p, units) > self.objective.key(self.posted_p, self.posted_units)
        if self.we_lead:
            return False
        if self.leader is None:
            return p >= self.threshold - 1e-9
        return self.objective.beats(p, units, self.leader)

    def mark_posted(self, phrase: str, p: float = 0.0, units: int = 0) -> None:
        self.posted = phrase
        self.posted_p = p
        self.posted_units = units


def chain_token(play_mode: str, phrase: str) -> str | None:
    """Casual can post a whole phrase as one word. Strict rejects dashes, so it stays word by word."""
    words = phrase.split()
    if play_mode != "word_chain" or not words:
        return None
    return "-".join(words)


def best_saved(question: dict, entries: list[dict]) -> dict | None:
    """Best vault line for this live question. The vault score is already P(goal)."""
    slug = question.get("slug") or ""
    title = " ".join((question.get("title") or "").casefold().split())
    best: dict | None = None
    for entry in entries:
        if entry.get("status") in ("dropped", "rejected"):
            continue
        entry_title = " ".join((entry.get("title") or "").casefold().split())
        if not ((slug and entry.get("slug") == slug) or (title and entry_title == title)):
            continue
        phrase = entry.get("phrase") or ""
        if not phrase or phrase_banned(phrase):
            continue
        p = float(entry["p_mean"])
        units = int(entry.get("units") or len(phrase.split()))
        if best is None or (p, -units) > (best["p"], -best["units"]):
            best = {"phrase": phrase, "p": p, "units": units}
    return best


def saved_line(question: dict) -> dict | None:
    """Vault line to fire the moment this live question is recognized."""
    from . import vault
    from .modes import HIGH_SCORES, STRICT

    entries = [
        entry
        for entry in vault.all_entries(HIGH_SCORES)
        if entry.get("mode") == STRICT and entry.get("status") not in ("dropped", "rejected")
    ]
    return best_saved(question, entries)


def post_phrase(client, question: dict, play_mode: str, phrase: str, log, stop_at: float | None = None) -> float:
    """Build the phrase on the live round's play mode and return the site probability.

    `stop_at` is the round's finish line. The first word that crosses it is kept, and the rest
    of the phrase is not sent.
    """
    from .modes import GOLF
    from .publish import build_chain, golf_try
    from .rules import check_phrase
    from .rules.banned import BannedPhrase, hit as hit_ban

    live = {"revisionId": question["revision_id"], "slug": question["slug"]}
    if span := hit_ban(phrase):
        raise BannedPhrase(phrase, span)
    if play_mode == GOLF:
        return golf_try(client, live, phrase, log)
    token = chain_token(play_mode, phrase)
    if token:
        words = [token]
    else:
        words = check_phrase(phrase, list((question.get("raw") or {}).get("choices") or []), max_words=400).split()
    attempt = client.attempt(question["revision_id"], play_mode) or client.start(question["revision_id"], play_mode)
    attempt, scored = build_chain(client, attempt, words, log, stop_at=stop_at)
    return float(scored or 0.0)
