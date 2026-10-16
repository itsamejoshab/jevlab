"""The menu no longer carries jevRequest. A snapshot still has to store one."""

import json

from jevlab.db import DB
from jevlab.site.snapshot import (
    format_elapsed,
    log_rule,
    progress_line,
    rule_title,
    take_snapshot,
)

REQUEST = (
    '{"state":"","model":"jev-latest","questions":{"q":{"type":"noul",'
    '"instructions":"Is cereal a soup?"}}}'
)


class Menu:
    edition = "jev"

    def __init__(self):
        self.asked = []

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

    def leaderboards(self, _revision, _mode):
        return {"leaderboards": {}}

    def words(self, _revision, _mode):
        return None

    def attempt(self, _revision, _mode):
        return None


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


def test_snapshot_log_names_the_edition_section(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.site.snapshot.SNAPSHOTS", tmp_path / "snapshots")
    seen = []
    take_snapshot(DB(tmp_path / "t.db"), Menu(), modes=("strict_chain", "golf"), log=seen.append, section="2/5")
    text = "\n".join(seen)
    assert log_rule("Jev") not in text
    assert "  snapshot 1 · 2 questions · Strict, Golf · anonymous" in seen
    assert "  jevRequest · 1/1 filled" in seen
    assert "  fetching 4 boards · 16 at a time" in seen
    assert any(line.startswith("  2/5 · Jev · 4/4 · 100%") for line in seen)
    assert any(line.startswith("──────────── 2/5 · Jev · done ·") for line in seen)
    assert rule_title(next(line for line in seen if "· done ·" in line)) == next(
        line for line in seen if "· done ·" in line
    ).removeprefix("──────────── ").removesuffix(" ────────────")
    assert any(line.startswith("  0 boards · 0 word lists ·") and line.endswith("· 0 failed") for line in seen)


def test_progress_line_keeps_the_section_after_the_header_scrolls_away():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(65) == "1m 05s"
    assert format_elapsed(3661) == "1h 01m"
    assert progress_line("clef-flash", 40, 120, 12, section="3/9") == "  3/9 · Clef-flash · 40/120 · 33% · 12s"
    assert progress_line("kev", 8, 8, 4, failed=2) == "  Kev · 8/8 · 100% · 4s · 2 failed"
    assert rule_title(log_rule("1/3 · Jev")) == "1/3 · Jev"
    assert rule_title("  1/3 · Jev · 40/120 · 33% · 12s") is None
