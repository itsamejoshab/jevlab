import json

from jevlab.db import DB


def make_db(tmp_path, kind="noul", leader=(0.99, 40)):
    db = DB(tmp_path / "t.db")
    db.execute("INSERT INTO questions VALUES (?, 0, 'r1', 'Is it?', '', ?, 'yes', 0.2, 0.5, 'm', ?, '{}')",
               ("q", kind, json.dumps({"state": "", "questions": []})))
    rows = [{"userId": "other", "name": "rival", "probability": leader[0], "wordCount": leader[1]}]
    for board in ("highScores", "shortestYes"):
        db.execute("INSERT INTO boards VALUES ('q', 'strict_chain', ?, 0, ?)", (board, json.dumps(rows)))
    return db
