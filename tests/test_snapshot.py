"""The menu no longer carries jevRequest. A snapshot still has to store one."""

import json

from jevlab.db import DB
from jevlab.site.snapshot import take_snapshot

REQUEST = (
    '{"state":"","model":"jev-latest","questions":{"q":{"type":"noul",'
    '"instructions":"Is cereal a soup?"}}}'
)


class Menu:
    edition = "jev"
    asked = []

    def menu(self):
        return {
            "questions": [
                {
                    "slug": "is-cereal-a-soup",
                    "revisionId": "rev-cereal",
                    "title": "Is cereal a soup?",
                    "kind": "noul",
                    "goal": "yes",
                    "baselineProbability": 0.2,
                    "yesThreshold": 0.5,
                },
                {
                    "slug": "already-there",
                    "revisionId": "rev-old",
                    "title": "Already has a request",
                    "kind": "noul",
                    "goal": "yes",
                    "jevRequest": REQUEST,
                },
            ]
        }

    def viewer(self):
        return {}

    def question(self, revision_id):
        self.asked.append(revision_id)
        assert revision_id == "rev-cereal"
        return {
            "revisionId": revision_id,
            "instructions": "Is cereal a soup?",
            "modelVersion": "jev-1.13.0",
            "jevRequest": REQUEST,
        }


def test_snapshot_fills_jev_request_the_menu_omits(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.site.snapshot.SNAPSHOTS", tmp_path / "snapshots")
    db = DB(tmp_path / "t.db")
    client = Menu()
    take_snapshot(db, client, modes=(), log=lambda *_: None)
    stored = db.question("is-cereal-a-soup")
    assert stored["jev_request"]["questions"]["q"]["instructions"] == "Is cereal a soup?"
    assert stored["model_version"] == "jev-1.13.0"
    assert json.loads(db.one("SELECT raw FROM questions WHERE slug = 'is-cereal-a-soup'")["raw"])["jevRequest"]
    assert client.asked == ["rev-cereal"]
    kept = db.question("already-there")
    assert kept["jev_request"]["model"] == "jev-latest"
