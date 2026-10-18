"""The app calibrates the oracle against site scores once those scores exist."""

import asyncio
import json
from types import SimpleNamespace

from textual.app import App, ComposeResult

from jevlab.calibrate import calibration_status
from jevlab.db import DB
from jevlab.modes import HIGHEST
from jevlab.oracle import Score

REQUEST = json.dumps({"state": "", "questions": {"q": {"type": "noul"}}})


class Site:
    """One signed-in board. `probability` is the site score a snapshot will store."""

    edition = "jev"
    probability = 0.5
    kind = "noul"

    def menu(self):
        return {
            "questions": [
                {
                    "slug": "is-cereal-a-soup",
                    "revisionId": "rev-cereal",
                    "title": "Is cereal a soup?",
                    "kind": self.kind,
                    "goal": "yes",
                    "baselineProbability": 0.2,
                    "yesThreshold": 0.5,
                    "jevRequest": REQUEST,
                }
            ]
        }

    def viewer(self):
        return {"player": {"user": {"id": "me"}}}

    def leaderboards(self, _revision, _mode):
        return {
            "leaderboards": {
                "highScores": [{"userId": "me", "phrase": "cereal is a soup", "probability": self.probability}]
            }
        }

    def words(self, _revision, _mode):
        return None

    def attempt(self, _revision, _mode):
        return None


def _patch_report(monkeypatch, path):
    monkeypatch.setattr("jevlab.calibrate.CALIBRATION_PATH", path)
    monkeypatch.setattr("jevlab.site.snapshot.SNAPSHOTS", path.parent / "snapshots")


def _recording_oracle(calls: list):
    class Oracle:
        def __init__(self, *_args, **_kwargs):
            calls.append("open")

        async def score(self, states, n=3):
            calls.append(list(states))
            return {state: Score(state, [0.5] * n) for state in states}

        async def close(self):
            return None

    return Oracle


def _download(monkeypatch, db, calls, edition="jev"):
    monkeypatch.setattr("jevlab.config.EDITION", edition)
    monkeypatch.setattr("jevlab.calibrate.Oracle", _recording_oracle(calls))
    monkeypatch.setattr("jevlab.config.activate", lambda edition=None: edition)
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: db)
    monkeypatch.setattr("jevlab.site.snapshot.SiteClient", Site)

    from jevlab.tui.pickers import run_snapshots

    run_snapshots(("jev",), ("strict_chain",), log=lambda _message: None)


def test_same_model_download_says_na(tmp_path, monkeypatch):
    report = tmp_path / "calibration.json"
    _patch_report(monkeypatch, report)
    Site.probability = 0.5
    Site.kind = "noul"
    db = DB(tmp_path / "t.db")
    calls: list = []
    _download(monkeypatch, db, calls)
    assert calibration_status() == "n/a"
    assert calls == []
    assert not report.exists()


def test_download_calibrates_site_scores_once(tmp_path, monkeypatch):
    report = tmp_path / "calibration.json"
    _patch_report(monkeypatch, report)
    Site.probability = 0.5
    Site.kind = "noul"
    db = DB(tmp_path / "t.db")
    calls: list = []
    _download(monkeypatch, db, calls, edition="clef")

    status = calibration_status()
    assert status.startswith("cal MAE")
    assert "n=1" in status
    assert json.loads(report.read_text())["pairs"] == 1

    _download(monkeypatch, db, calls, edition="clef")
    assert calls.count("open") == 1
    assert json.loads(report.read_text())["pairs"] == 1

    Site.probability = 0.8
    _download(monkeypatch, db, calls, edition="clef")
    assert calls.count("open") == 2
    assert calibration_status().startswith("cal MAE")


def test_opening_the_app_on_the_site_model_says_na(tmp_path, monkeypatch):
    report = tmp_path / "calibration.json"
    _patch_report(monkeypatch, report)
    db = DB(tmp_path / "t.db")
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Is cereal a soup?', '', 'noul', 'yes', 0.2, 0.5, 'm', ?, '{}')",
        ("is-cereal-a-soup", REQUEST),
    )
    db.execute(
        "INSERT INTO site_scores VALUES (?, 'strict_chain', 'cereal is a soup', 0.5, 'board:highScores', 't')",
        ("is-cereal-a-soup",),
    )
    calls: list = []
    monkeypatch.setattr("jevlab.calibrate.Oracle", _recording_oracle(calls))
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: db)

    class _App(App):
        def __init__(self):
            super().__init__()
            self.game_mode = HIGHEST

        def on_mount(self) -> None:
            from jevlab.tui.pickers import HomeScreen

            self.push_screen(HomeScreen())

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(30):
                if report.exists():
                    break
                await pilot.pause(0.1)
            assert calibration_status() == "n/a"
            assert calls == []
            assert not report.exists()

    asyncio.run(check())


def test_download_without_comparable_scores_stays_uncalibrated(tmp_path, monkeypatch):
    report = tmp_path / "calibration.json"
    _patch_report(monkeypatch, report)
    Site.probability = 0.5
    Site.kind = "choice"
    db = DB(tmp_path / "t.db")
    calls: list = []
    _download(monkeypatch, db, calls, edition="clef")
    assert calibration_status() == "uncalibrated"
    assert calls == []
    assert not report.exists()


def test_cli_snapshot_on_the_site_model_says_na(tmp_path, monkeypatch):
    report = tmp_path / "calibration.json"
    _patch_report(monkeypatch, report)
    Site.probability = 0.5
    Site.kind = "noul"
    db = DB(tmp_path / "t.db")
    calls: list = []
    monkeypatch.setattr("jevlab.calibrate.Oracle", _recording_oracle(calls))
    monkeypatch.setattr("jevlab.db.DB", lambda *_args, **_kwargs: db)
    monkeypatch.setattr("jevlab.site.snapshot.SiteClient", Site)

    from jevlab.cli import cmd_snapshot

    assert cmd_snapshot(SimpleNamespace(q=None, modes=None, workers=1)) == 0
    assert calibration_status() == "n/a"
    assert calls == []
    assert not report.exists()
