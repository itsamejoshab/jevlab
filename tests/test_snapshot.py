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

REQUEST = '{"state":"","model":"jev-latest","questions":{"q":{"type":"noul","instructions":"Is cereal a soup?"}}}'


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


def _seed_question(db: DB, slug: str) -> None:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'rev', ?, '', 'noul', 'yes', 0.2, 0.5, 'm', '{}', '{}')",
        (slug, slug),
    )
    db.execute("INSERT INTO boards VALUES (?, 'strict_chain', 'highScores', 0, '[]')", (slug,))
    db.execute("INSERT INTO word_impacts VALUES (?, 'strict_chain', 0, '[]')", (slug,))
    db.execute("INSERT INTO attempts VALUES (?, 'strict_chain', 0, '{}')", (slug,))
    db.execute(
        "INSERT INTO site_scores VALUES (?, 'strict_chain', 'phrase', 0.5, 'board:highScores', 't')",
        (slug,),
    )
    db.execute("INSERT INTO defense_words VALUES (?, 'strict_chain', 'decoy', 't')", (slug,))


def _stored_tables(db: DB, slug: str) -> set[str]:
    tables = ("questions", "boards", "word_impacts", "attempts", "site_scores", "defense_words")
    return {table for table in tables if db.one(f"SELECT 1 FROM {table} WHERE slug = ?", (slug,))}


class OneQuestion:
    """A menu that still lists only the question we mean to keep."""

    edition = "jev"

    def menu(self):
        return {
            "questions": [
                {
                    "slug": "still-here",
                    "revisionId": "rev-keep",
                    "title": "Still here",
                    "kind": "noul",
                    "goal": "yes",
                    "jevRequest": REQUEST,
                }
            ]
        }

    def viewer(self):
        return {}

    def question(self, _revision_id):
        raise AssertionError("the menu already carried jevRequest")

    def leaderboards(self, _revision, _mode):
        return {"leaderboards": {}}

    def words(self, _revision, _mode):
        return []

    def attempt(self, _revision, _mode):
        return None


def test_snapshot_drops_questions_the_menu_no_longer_lists(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.site.snapshot.SNAPSHOTS", tmp_path / "snapshots")
    db = DB(tmp_path / "t.db")
    _seed_question(db, "still-here")
    _seed_question(db, "serverless-postgres-with-branching-and-scale-to-zero-pick")
    seen = []
    take_snapshot(db, OneQuestion(), modes=(), log=seen.append)

    assert db.question("still-here")["title"] == "Still here"
    assert _stored_tables(db, "still-here") == {
        "questions",
        "boards",
        "word_impacts",
        "attempts",
        "site_scores",
        "defense_words",
    }
    assert _stored_tables(db, "serverless-postgres-with-branching-and-scale-to-zero-pick") == set()
    assert "  removed 1 question(s) no longer on the menu" in seen


def test_snapshot_keeps_other_questions_when_only_some_slugs_are_refreshed(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.site.snapshot.SNAPSHOTS", tmp_path / "snapshots")
    db = DB(tmp_path / "t.db")
    _seed_question(db, "still-here")
    _seed_question(db, "left-the-menu")
    take_snapshot(db, OneQuestion(), slugs=["still-here"], modes=(), log=lambda *_: None)

    assert db.question("still-here") is not None
    assert _stored_tables(db, "left-the-menu") == {
        "questions",
        "boards",
        "word_impacts",
        "attempts",
        "site_scores",
        "defense_words",
    }


def test_snapshot_does_not_wipe_when_the_menu_lists_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.site.snapshot.SNAPSHOTS", tmp_path / "snapshots")
    db = DB(tmp_path / "t.db")
    _seed_question(db, "still-here")

    class Empty(OneQuestion):
        def menu(self):
            return {"questions": []}

    take_snapshot(db, Empty(), modes=(), log=lambda *_: None)
    assert _stored_tables(db, "still-here") == {
        "questions",
        "boards",
        "word_impacts",
        "attempts",
        "site_scores",
        "defense_words",
    }


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
