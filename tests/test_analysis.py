import json

from jevlab.analysis import (
    assemble,
    baseline_questions,
    baseline_value,
    gaps,
    leading_row,
    phrase_values,
    plays_for,
    probability_of,
    shown_score,
)
from jevlab.config import EDITION_MODELS
from jevlab.db import DB
from jevlab.oracle import question_key


def test_plays_keep_yes_no_and_small_unranked_choice():
    assert plays_for({"kind": "noul", "goal": "no", "raw": {}}) == ["no"]
    assert plays_for({"kind": "score", "goal": "yes", "raw": {}}) == []
    small = [{"option": "Vanilla"}, {"option": "Chocolate"}]
    assert plays_for({"kind": "choice", "goal": "yes", "raw": {"choices": small}}) == ["Vanilla", "Chocolate"]
    ranked = {"ranked": True, "choices": [{"option": "A"}, {"option": "B"}]}
    assert plays_for({"kind": "choice", "goal": "yes", "raw": ranked}) == []
    wide = [{"option": str(i)} for i in range(8)]
    assert plays_for({"kind": "choice", "goal": "yes", "raw": {"choices": wide}}) == []
    assert plays_for({"kind": "choice", "goal": "yes", "raw": {"choices": []}}) == []


def test_leading_row_is_the_board_leader_including_ours():
    rows = [
        {"phrase": "long", "probability": 0.984, "wordCount": 5, "choice": "", "name": "Ann"},
        {"phrase": "short", "probability": 0.981, "wordCount": 1, "choice": "", "name": "Me"},
        {"phrase": "higher", "probability": 0.991, "wordCount": 9, "choice": "", "name": "Me"},
    ]
    assert leading_row(rows, "yes", "noul")["phrase"] == "higher"
    assert leading_row(rows[:2], "yes", "noul")["phrase"] == "short"
    choice = [
        {"phrase": "fragaria", "probability": 0.9, "wordCount": 1, "choice": "Vanilla"},
        {"phrase": "cacao", "probability": 0.4, "wordCount": 1, "choice": "Chocolate"},
    ]
    assert leading_row(choice, "Chocolate", "choice")["phrase"] == "cacao"
    assert leading_row(choice, "Strawberry", "choice") is None
    assert leading_row([], "yes", "noul") is None


def test_baselines_are_one_row_per_question():
    table = {
        "rows": [
            {"slug": "sky", "title": "Sky", "kind": "noul", "answer": "no", "phrase": "not blue"},
            {"slug": "sky", "title": "Sky", "kind": "noul", "answer": "no", "phrase": "grey"},
            {"slug": "ice", "title": "Ice", "kind": "choice", "answer": "Vanilla", "phrase": "fragaria"},
            {"slug": "ice", "title": "Ice", "kind": "choice", "answer": "Chocolate", "phrase": "cacao"},
            {"slug": "ice", "title": "Ice", "kind": "choice", "answer": "Vanilla", "phrase": "fragaria"},
        ]
    }
    found = {row["slug"]: row for row in baseline_questions(table)}
    assert set(found) == {"sky", "ice"}
    assert found["sky"]["goal"] == "no"
    assert found["sky"]["phrases"] == ["not blue", "grey"]
    assert found["ice"]["goal"] == ""
    assert found["ice"]["phrases"] == ["fragaria", "cacao"]
    assert baseline_value("noul", "no", 0.25) == 0.75
    assert baseline_value("choice", "", 0.66) == 0.66


def test_choice_score_uses_the_named_option_and_no_flips():
    payload = json.dumps({"type": "choice", "probabilities": {"Vanilla": 0.2, "Chocolate": 0.8}})
    assert probability_of(payload, "Vanilla") == 0.2
    assert probability_of(payload, "Mint") is None
    assert phrase_values([(0.8, payload), (0.8, payload)], "choice", "Vanilla") == [0.2, 0.2]
    assert phrase_values([(0.25, None), (0.35, None)], "noul", "no") == [0.75, 0.65]
    assert shown_score("noul", "no", 0.25) == 0.75
    assert shown_score("choice", "Vanilla", 0.2) == 0.2


def _question(db: DB, slug: str, kind: str, goal: str, choices=None, ranked=False, model: str = "jev-latest"):
    raw = {"choices": choices or [], "ranked": ranked, "goal": goal, "kind": kind}
    request = {"state": "", "model": model, "questions": {"q": {"type": kind, "instructions": slug}}}
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r', ?, '', ?, ?, 0.2, 0.5, 'm', ?, ?)",
        (slug, slug, kind, goal, json.dumps(request), json.dumps(raw)),
    )


def _board(db: DB, slug: str, rows: list, mode: str = "strict_chain", board: str = "highScores"):
    db.execute("INSERT INTO boards VALUES (?, ?, ?, 0, ?)", (slug, mode, board, json.dumps(rows)))


def _sample(db: DB, question: dict, model: str, phrase: str, noul: float, answer: str | None = None):
    qkey = question_key(question["jev_request"], model)
    db.execute(
        "INSERT INTO oracle_samples (request_hash, model, state, noul, answer, latency_ms, cost, at, qkey)"
        " VALUES ('h', ?, ?, ?, ?, 1, 0, 0, ?)",
        (model, phrase, noul, answer, qkey),
    )


def _row(phrase: str, probability: float, words: int = 1, choice: str = "", name: str = "Ann"):
    return {
        "userId": "u",
        "name": name,
        "phrase": phrase,
        "choice": choice,
        "probability": probability,
        "wordCount": words,
    }


def test_assemble_keeps_only_leaders_and_records_blanks(tmp_path):
    jev = DB(tmp_path / "jev.db")
    kev = DB(tmp_path / "kev.db")
    luna = DB(tmp_path / "luna.db")
    decider = DB(tmp_path / "decider.db")
    jev_model = EDITION_MODELS["jev"]
    kev_model = EDITION_MODELS["kev"]
    luna_model = EDITION_MODELS["luna"]
    decider_model = EDITION_MODELS["decider"]

    _question(jev, "sky", "noul", "no", model=jev_model)
    _board(
        jev,
        "sky",
        [_row("nope", 0.91, 2), _row("weaker", 0.4, 1)],
    )
    _question(jev, "sky-short", "noul", "yes", model=jev_model)
    _board(jev, "sky-short", [_row("brief", 0.99)], board="shortestYes")
    _question(jev, "golf-only", "noul", "yes", model=jev_model)
    _board(jev, "golf-only", [_row("casual", 0.99)], mode="golf")

    _question(jev, "ice", "choice", "yes", [{"option": "Vanilla"}, {"option": "Chocolate"}], model=jev_model)
    _board(
        jev,
        "ice",
        [_row("fragaria", 0.88, 1, "Vanilla"), _row("cacao", 0.3, 1, "Chocolate"), _row("runner-up", 0.5, 1, "Vanilla")],
    )
    wide = [{"option": f"opt-{i}"} for i in range(8)]
    _question(jev, "wide", "choice", "yes", wide, model=jev_model)
    _board(jev, "wide", [_row("too-many", 0.99, 1, "opt-0")])
    _question(jev, "ranked", "choice", "yes", [{"option": "A"}, {"option": "B"}], ranked=True, model=jev_model)
    _board(jev, "ranked", [_row("placed", 0.99, 1, "A")])
    _question(jev, "shark", "score", "yes", model=jev_model)
    _board(jev, "shark", [_row("ten", 0.99)])

    _question(jev, "plain", "noul", "yes", model=jev_model)
    _board(jev, "plain", [_row("yes-please", 0.77)])
    _question(kev, "plain", "noul", "yes", model=kev_model)
    _question(luna, "plain", "noul", "yes", model=luna_model)
    _question(decider, "plain", "noul", "yes", model=decider_model)

    _question(kev, "ice", "choice", "yes", [{"option": "Vanilla"}, {"option": "Chocolate"}], model=kev_model)
    ice = kev.question("ice")
    payload = json.dumps({"type": "choice", "probabilities": {"Vanilla": 0.2, "Chocolate": 0.8}})
    _sample(kev, ice, kev_model, "fragaria", 0.8, payload)
    _sample(kev, ice, kev_model, "fragaria", 0.8, json.dumps({"probabilities": {"Vanilla": 0.4, "Chocolate": 0.6}}))

    _question(kev, "rude", "noul", "no", model=kev_model)
    _question(jev, "rude", "noul", "no", model=jev_model)
    _board(jev, "rude", [_row("nope-again", 0.8)])
    _sample(kev, kev.question("rude"), kev_model, "nope-again", 0.25, json.dumps({"type": "noul", "noul": 0.25}))

    _question(luna, "ice", "choice", "yes", [{"option": "Vanilla"}, {"option": "Chocolate"}], model=luna_model)
    luna_qkey = question_key(luna.question("ice")["jev_request"], luna_model)
    luna.execute(
        "INSERT INTO oracle_refusals (qkey, state, detail, at) VALUES (?, 'fragaria', 'refused', 0)",
        (luna_qkey,),
    )
    _question(decider, "ice", "choice", "yes", [{"option": "Vanilla"}, {"option": "Chocolate"}], model=decider_model)

    table = assemble({"jev": jev, "kev": kev, "luna": luna, "decider": decider})
    by_play = {}
    for row in table["rows"]:
        by_play.setdefault(row["play"], []).append(row)

    assert set(by_play) == {"sky | no", "ice | Vanilla", "ice | Chocolate", "plain | yes", "rude | no"}
    assert [row["edition"] for row in by_play["sky | no"]] == ["jev"]
    sky = by_play["sky | no"][0]
    assert sky["phrase"] == "nope"
    assert sky["scores"]["jev"] == 0.91
    assert sky["from"]["jev"] == "board"
    assert sky["blank"]["kev"] == "missing-question"
    assert sky["blank"]["laya"] == "missing-edition"

    ice_rows = {row["answer"]: row for row in by_play["ice | Vanilla"] + by_play["ice | Chocolate"]}
    vanilla = ice_rows["Vanilla"]
    chocolate = ice_rows["Chocolate"]
    assert vanilla["phrase"] == "fragaria"
    assert chocolate["phrase"] == "cacao"
    assert vanilla["scores"]["kev"] == 0.3
    assert vanilla["n"]["kev"] == 2
    assert vanilla["from"]["kev"] == "oracle"
    assert vanilla["blank"]["luna"] == "refused"
    assert vanilla["blank"]["decider"] == "down"
    assert chocolate["scores"]["kev"] is None
    assert "kev" not in chocolate["blank"]

    rude = by_play["rude | no"][0]
    assert rude["scores"]["kev"] == 0.75

    pending = {(gap.slug, gap.answer, gap.scorer) for gap in gaps(table)}
    assert ("plain", "yes", "kev") in pending
    assert ("plain", "yes", "luna") in pending
    assert ("plain", "yes", "decider") not in pending
    assert ("ice", "Vanilla", "luna") not in pending
    assert ("ice", "Vanilla", "decider") not in pending
    assert ("ice", "Chocolate", "kev") in pending
    assert ("ice", "Chocolate", "luna") in pending
    assert ("sky", "no", "kev") not in pending
    assert all(row["slug"] not in {"wide", "ranked", "shark", "sky-short", "golf-only"} for row in table["rows"])
