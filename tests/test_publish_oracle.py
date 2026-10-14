"""A submit spends site rolls first. The local oracle runs only after those rolls lose."""

import json
import unicodedata

from jevlab import vault
from jevlab.boards import standings
from jevlab.db import DB
from jevlab.modes import GOLF, HIGH_SCORES
from jevlab.oracle import question_key
from jevlab.publish import PublishError, WrongGoal, publish_entry

REQUEST = {"state": "", "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "Is it rude?"}}}
PHRASE = "relentlessly keep swiping"
OTHER = "swiping ritual reversed"
UNITS = len(unicodedata.normalize("NFC", PHRASE).strip())


def _question(db: DB, slug: str, goal: str = "yes") -> dict:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Is it rude?', '', 'noul', ?, 0.2, 0.5, 'm', ?, ?)",
        (slug, goal, json.dumps(REQUEST), json.dumps({"goal": goal})),
    )
    return db.question(slug)


def _save(slug: str, phrase: str, p: float) -> dict:
    return vault.save(
        slug,
        GOLF,
        phrase,
        p_mean=p,
        p_lcb=p,
        spread=0.01,
        n=5,
        units=UNITS,
        leader=None,
        beats=True,
        title=slug,
    )


def _samples(db: DB, qkey: str, phrase: str, samples: list[float]) -> None:
    for noul in samples:
        db.execute(
            "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey) "
            "VALUES (?, 'jev-latest', ?, ?, '', 1, 0, 0, ?)",
            (f"{phrase}:{noul}", phrase, noul, qkey),
        )


class _Site:
    """One golf phrase. The board shows the posted score only after the first attempt starts."""

    def __init__(self, score: float, order: list[str]):
        self.score = score
        self.order = order
        self.posted = False

    def leaderboards(self, revision_id: str, mode: str) -> dict:
        rows = [{"userId": "them", "probability": 0.50, "wordCount": 40, "name": "them"}]
        if self.posted:
            rows.append({"userId": "me", "probability": self.score, "wordCount": UNITS, "name": "me"})
        return {"leaderboards": {HIGH_SCORES: rows}}

    def start(self, revision_id: str, mode: str) -> dict:
        self.order.append("post")
        self.posted = True
        return {"revisionId": revision_id, "playMode": mode, "tokens": [], "turns": []}

    def turn(self, attempt: dict, operation: dict) -> tuple[dict, dict]:
        turn = {"probability": self.score, "choice": ""}
        return {**attempt, "turns": [turn]}, turn


def _oracle(order: list[str], fresh: float):
    async def check(db, question, phrase, n=3, target=""):
        order.append("oracle")
        return fresh

    return check


def _publish(db, client, entry, live, order, fresh, *, rerolls=1, verify=True, dry_run=False):
    notes: list[str] = []
    try:
        result = publish_entry(
            db,
            client,
            entry,
            {entry["slug"]: live},
            "me",
            dry_run,
            rerolls,
            verify,
            notes.append,
            5,
        )
    except WrongGoal:
        raise
    except PublishError as error:
        result = f"failed: {error}"
    return result, notes


def _live(slug: str, goal: str = "yes") -> dict:
    return {"slug": slug, "revisionId": "r1", "goal": goal, "kind": "noul", "choices": []}


def test_a_win_is_posted_without_a_local_recheck(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    _question(db, slug)
    entry = _save(slug, PHRASE, 0.97) | {"slug": slug, "mode": GOLF, "board": HIGH_SCORES}
    order: list[str] = []
    monkeypatch.setattr("jevlab.publish.oracle_check", _oracle(order, 0.10))

    result, _notes = _publish(db, _Site(0.99, order), entry, _live(slug), order, 0.10)

    saved = vault.load(slug, GOLF, HIGH_SCORES)["entries"][0]
    assert result.startswith("won")
    assert saved["status"] == "published"
    assert saved["server_p"] == 0.99
    assert order == ["post", "post"]


def test_a_loss_records_the_site_score_and_checks_the_goal_after_the_rolls(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    _question(db, slug)
    entry = _save(slug, PHRASE, 0.97) | {"slug": slug, "mode": GOLF, "board": HIGH_SCORES}
    order: list[str] = []
    monkeypatch.setattr("jevlab.publish.oracle_check", _oracle(order, 0.10))

    result, _notes = _publish(db, _Site(0.20, order), entry, _live(slug), order, 0.10)

    saved = vault.load(slug, GOLF, HIGH_SCORES)["entries"][0]
    posted = db.all("SELECT status, server FROM publishes")
    row = next(item for item in standings(db, GOLF) if item.slug == slug)
    assert result.startswith("short")
    assert saved["status"] == "failed"
    assert saved["server_p"] == 0.20
    assert saved["p_mean"] == 0.20
    assert posted[0]["status"] == "short"
    assert posted[0]["server"] == 0.20
    assert row.entry_beats is False
    assert order.index("post") < order.index("oracle")


def test_a_loss_whose_recheck_is_the_complement_drops_the_wrong_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    question = _question(db, slug, "no")
    qkey = question_key(question["jev_request"])
    entry = _save(slug, PHRASE, 0.978) | {"slug": slug, "mode": GOLF, "board": HIGH_SCORES}
    _save(slug, OTHER, 0.896)
    _samples(db, qkey, PHRASE, [0.98, 0.97, 0.98])
    _samples(db, qkey, OTHER, [0.10, 0.11, 0.10])
    order: list[str] = []
    monkeypatch.setattr("jevlab.publish.oracle_check", _oracle(order, 0.022))

    try:
        _publish(db, _Site(0.20, order), entry, _live(slug, "no"), order, 0.022)
    except WrongGoal as error:
        removed = error.phrases
    else:
        raise AssertionError("expected the complement re-check to drop the wrong-goal line")

    posted = db.all("SELECT status, server FROM publishes")
    assert removed == {PHRASE}
    assert posted and posted[0]["status"] == "short" and posted[0]["server"] == 0.20
    assert {item["phrase"] for item in vault.all_entries()} == {OTHER}
    assert order.index("post") < order.index("oracle")
    assert not any(item["status"] == "published" for item in vault.all_entries())


def test_a_dry_run_does_not_spend_a_local_recheck(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    _question(db, slug)
    entry = _save(slug, PHRASE, 0.97) | {"slug": slug, "mode": GOLF, "board": HIGH_SCORES}
    order: list[str] = []
    monkeypatch.setattr("jevlab.publish.oracle_check", _oracle(order, 0.10))

    result, _notes = _publish(db, _Site(0.99, order), entry, _live(slug), order, 0.10, dry_run=True)

    assert result == "dry-run"
    assert order == []
    assert vault.load(slug, GOLF, HIGH_SCORES)["entries"][0]["status"] == "candidate"
