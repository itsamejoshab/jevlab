import json

from jevlab import vault
from jevlab.db import DB
from jevlab.modes import STRICT
from jevlab.oracle import question_key
from jevlab.publish import WrongGoal, goal_flipped, realign_vault_goals, settle_goal

REQUEST = {"state": "", "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "Is it rude?"}}}
WRONG = "relentlessly keep swiping"
RIGHT = "swiping ritual reversed"


def test_goal_flipped_is_the_complement_within_one_hundredth():
    assert goal_flipped(0.978, 0.022)
    assert goal_flipped(0.978, 0.030)
    assert not goal_flipped(0.978, 0.040)
    assert not goal_flipped(0.896, 0.890)
    assert not goal_flipped(0.50, 0.50)


def _question(db: DB, slug: str, goal: str) -> dict:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Is it rude?', '', 'noul', ?, 0.2, 0.5, 'm', ?, ?)",
        (slug, goal, json.dumps(REQUEST), json.dumps({"goal": goal})),
    )
    return db.question(slug)


def _save(slug: str, phrase: str, p: float) -> None:
    vault.save(
        slug,
        STRICT,
        phrase,
        p_mean=p,
        p_lcb=p,
        spread=0.01,
        n=5,
        units=len(phrase.split()),
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
    db.execute(
        "INSERT INTO lab_candidates (qkey, state, origin, parent, created) VALUES (?, ?, 'genetic', '', 0)",
        (qkey, phrase),
    )


def test_publish_drops_lines_scored_against_the_other_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    question = _question(db, slug, "no")
    qkey = question_key(question["jev_request"])
    _save(slug, WRONG, 0.978)
    _save(slug, RIGHT, 0.896)
    _samples(db, qkey, WRONG, [0.98, 0.97, 0.98])
    _samples(db, qkey, RIGHT, [0.10, 0.11, 0.10])
    db.execute(
        "INSERT INTO lab_memory (qkey, board, slug, data, updated) VALUES (?, 'highScores', ?, ?, 0)",
        (
            qkey,
            slug,
            json.dumps(
                {
                    "best_p": 0.978,
                    "best_units": 3,
                    "exhausted": [[WRONG, 0.978, 3], [RIGHT, 0.896, 3]],
                    "history": [{"best_p": 0.978, "units": 3}, {"best_p": 0.896, "units": 3}],
                }
            ),
        ),
    )

    try:
        settle_goal(db, question, {"goal": "no", "kind": "noul"}, 0.978, 0.022, False, lambda _m: None)
    except WrongGoal as error:
        assert error.phrases == {WRONG}
    else:
        raise AssertionError("expected WrongGoal")

    assert {entry["phrase"] for entry in vault.all_entries()} == {RIGHT}
    assert db.all("SELECT state FROM oracle_samples WHERE state = ?", (WRONG,)) == []
    assert db.all("SELECT state FROM lab_candidates WHERE state = ?", (WRONG,)) == []
    assert db.one("SELECT state FROM lab_candidates WHERE state = ?", (RIGHT,))["state"] == RIGHT
    memory = json.loads(db.one("SELECT data FROM lab_memory WHERE slug = ?", (slug,))["data"])
    assert memory["best_p"] == 0
    assert memory["exhausted"] == [[RIGHT, 0.896, 3]]
    assert memory["history"] == [{"best_p": 0.896, "units": 3}]


def test_snapshot_goal_is_corrected_to_the_site(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    question = _question(db, slug, "yes")
    qkey = question_key(question["jev_request"])
    _save(slug, WRONG, 0.978)
    _save(slug, RIGHT, 0.896)
    _samples(db, qkey, WRONG, [0.98, 0.97, 0.98])
    _samples(db, qkey, RIGHT, [0.10, 0.11, 0.10])

    fresh = settle_goal(db, question, {"goal": "no", "kind": "noul"}, 0.896, 0.104, False, lambda _m: None)

    assert abs(fresh - 0.896) < 1e-9
    assert db.question(slug)["goal"] == "no"
    assert db.question(slug)["raw"]["goal"] == "no"
    assert {entry["phrase"] for entry in vault.all_entries()} == {RIGHT}


def test_snapshot_refresh_cleans_inverted_lines_when_the_goal_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    question = _question(db, slug, "no")
    qkey = question_key(question["jev_request"])
    _save(slug, WRONG, 0.978)
    _save(slug, RIGHT, 0.896)
    _samples(db, qkey, WRONG, [0.98, 0.97, 0.98])
    _samples(db, qkey, RIGHT, [0.10, 0.11, 0.10])
    logs = []

    removed = realign_vault_goals(db, {slug: "yes"}, {slug}, logs.append)

    assert removed == {WRONG}
    assert any("goal 'yes' -> 'no'" in line for line in logs)
    assert {entry["phrase"] for entry in vault.all_entries()} == {RIGHT}
    assert db.all("SELECT state FROM oracle_samples WHERE state = ?", (WRONG,)) == []


def test_snapshot_refresh_cleans_inverted_lines_when_the_goal_already_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    slug = "hinge"
    question = _question(db, slug, "no")
    qkey = question_key(question["jev_request"])
    _save(slug, WRONG, 0.978)
    _save(slug, RIGHT, 0.896)
    _samples(db, qkey, WRONG, [0.98, 0.97, 0.98])
    _samples(db, qkey, RIGHT, [0.10, 0.11, 0.10])

    removed = realign_vault_goals(db, {slug: "no"}, {slug}, lambda _m: None)

    assert removed == {WRONG}
    assert {entry["phrase"] for entry in vault.all_entries()} == {RIGHT}
