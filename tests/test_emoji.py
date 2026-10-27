"""Emoji search, the grammar the site enforces, and the local ranker."""

from __future__ import annotations

import asyncio
import inspect
import json

import numpy as np
import pytest

from jevlab import vault
from jevlab.crosspost import picks
from jevlab.db import DB
from jevlab.emoji.learn import TRUST_RHO, Model, atom_vectors, fit
from jevlab.emoji.rank import Atom, shortlist
from jevlab.emoji.rules import ZWJ, RuleError, check_phrase, check_word
from jevlab.emoji.search import load_catalog, search
from jevlab.modes import EMOJI, HIGH_SCORES, SHORTEST_YES, STRICT
from jevlab.oracle import Score
from jevlab.publish import publish_entry


def test_emoji_words_match_the_site_checker():
    assert check_word("😀") == "😀"
    assert check_word("👍🏻") == "👍🏻"
    assert check_word("🇺🇸") == "🇺🇸"
    assert check_word("1️⃣") == "1️⃣"
    assert check_word("❤") == "❤\ufe0f"
    assert check_word("👨‍👩‍👧") == "👨‍👩‍👧"
    assert check_word("😀\u200d🥣") == "😀\u200d🥣"
    for text in ("yes", "😀🎉", "😀 🎉", "😀\u200d", "★"):
        with pytest.raises(RuleError):
            check_word(text)


def test_catalog_atoms_are_legal_emoji_words():
    atoms = load_catalog()
    assert len(atoms) > 1000
    assert all(check_word(atom.text) == atom.text for atom in atoms)


def test_cold_rank_prefers_a_name_that_overlaps_the_question():
    atoms = [Atom("🚗", "automobile"), Atom("🥣", "bowl")]
    order = shortlist("a bowl of soup", atoms, {})
    assert [atom.name for atom in order] == ["bowl", "automobile"]


def test_untrained_model_matches_name_cosine_and_a_weak_fit_stays_cold():
    names = {
        "🥣": np.array([1.0, 0.0]),
        "🚗": np.array([0.0, 1.0]),
    }
    fresh = Model.untrained(names)
    question = np.array([1.0, 0.0])
    assert fresh.atom_score(question, "🥣") == pytest.approx(1.0)
    assert fresh.atom_score(question, "🚗") == pytest.approx(0.0)

    rows = []
    rng = np.random.default_rng(0)
    for _ in range(12):
        qvec = rng.normal(size=2)
        rows.append((qvec, ["🥣"], float(rng.normal())))
    fitted = fit(rows, names)
    assert np.isfinite(fitted.rho)

    atoms = [Atom("🥣", "bowl"), Atom("🚗", "automobile")]
    cold = [atom.name for atom in shortlist("a bowl of soup", atoms, {})]
    weak = Model.untrained(names)
    weak.rho = TRUST_RHO / 2
    weak.bias = {"🥣": -5.0, "🚗": 5.0}
    gated = [atom.name for atom in shortlist("a bowl of soup", atoms, {}, question, names, weak)]
    assert gated == cold

    trusted = Model.untrained(names)
    trusted.rho = 0.9
    trusted.bias = {"🥣": -5.0, "🚗": 5.0}
    learned = shortlist("a bowl of soup", atoms, {}, question, names, trusted)
    assert learned[0].name == "automobile"


def test_atom_vectors_stores_an_embedding_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr("jevlab.emoji.learn.vector_path", lambda: tmp_path / "vecs.npz")

    def embed(texts: list[str]) -> np.ndarray:
        return np.arange(len(texts) * 2, dtype=np.float64).reshape(len(texts), 2)

    found = atom_vectors([("🥣", "bowl"), ("🚗", "automobile")], embed=embed)
    assert set(found) == {"🥣", "🚗"}
    again = atom_vectors([("🥣", "bowl"), ("🚗", "automobile")], embed=lambda _texts: np.zeros((0, 2)))
    assert np.allclose(again["🥣"], found["🥣"])


def _question(db: DB, slug: str = "cereal") -> None:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Is cereal a soup?', '', 'noul', 'yes', 0.2, 0.5, 'm', ?, ?)",
        (slug, json.dumps({"state": ""}), json.dumps({"goal": "yes", "kind": "noul"})),
    )


def test_search_stops_at_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.9, "wordCount": 1}]),
        ),
    )
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    catalog = [Atom(f"{index}\u20e3", f"digit {index}") for index in range(6)]
    phrase = asyncio.run(
        search(db, "cereal", budget=3, log=lambda _message: None, score=score, catalog=catalog, embed=False)
    )
    assert len(calls) <= 3
    assert phrase is None


def test_search_skips_a_one_emoji_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.99, "wordCount": 1}]),
        ),
    )
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    phrase = asyncio.run(
        search(
            db,
            "cereal",
            budget=10,
            log=lambda _message: None,
            score=score,
            catalog=[Atom("🥣", "bowl")],
            embed=False,
        )
    )
    assert phrase is None
    assert calls == []


def test_search_stops_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.9, "wordCount": 1}]),
        ),
    )
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    catalog = [Atom(f"{index}\u20e3", f"digit {index}") for index in range(6)]
    phrase = asyncio.run(
        search(
            db,
            "cereal",
            budget=10,
            log=lambda _message: None,
            score=score,
            catalog=catalog,
            embed=False,
            halt=lambda: len(calls) >= 1,
        )
    )
    assert calls == [catalog[0].text]
    assert phrase is None


def test_search_reports_the_best_and_a_win(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    seen: list[dict] = []

    async def score(phrase: str, n: int = 1) -> Score:
        return Score(phrase, [0.8] * n)

    catalog = [Atom("🥣", "bowl")]
    phrase = asyncio.run(
        search(
            db,
            "cereal",
            budget=1,
            log=lambda _message: None,
            score=score,
            catalog=catalog,
            embed=False,
            progress=seen.append,
        )
    )
    assert phrase == "🥣"
    assert seen[-1]["state"] == "done"
    assert seen[-1]["best"] == "0.800/1w"
    assert seen[-1]["win"] is True
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.7, "wordCount": 1}]),
        ),
    )
    short: list[dict] = []

    async def low(phrase: str, n: int = 1) -> Score:
        return Score(phrase, [0.4] * n)

    missed = asyncio.run(
        search(
            db,
            "cereal",
            budget=1,
            log=lambda _message: None,
            score=low,
            catalog=catalog,
            embed=False,
            progress=short.append,
        )
    )
    assert missed is None
    assert short[-1]["best"] == "0.400/1w"
    assert short[-1]["win"] is False
    assert short[-1]["phrase"] == "🥣"


def test_search_moves_on_as_soon_as_a_line_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.5, "wordCount": 1}]),
        ),
    )
    bowl, spoon, car = Atom("🥣", "bowl"), Atom("🥄", "spoon"), Atom("🚗", "car")
    values = {bowl.text: 0.2, spoon.text: 0.8, car.text: 0.9}
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [values[phrase]] * n)

    phrase = asyncio.run(
        search(
            db,
            "cereal",
            budget=10,
            log=lambda _message: None,
            score=score,
            catalog=[bowl, spoon, car],
            embed=False,
            win_mode=True,
        )
    )
    assert calls == [bowl.text, spoon.text]
    assert phrase == spoon.text
    saved = vault.load("cereal", EMOJI)["entries"]
    assert saved[0]["phrase"] == spoon.text
    assert saved[0]["beats"] is True
    calls.clear()
    asyncio.run(
        search(
            db,
            "cereal",
            budget=3,
            log=lambda _message: None,
            score=score,
            catalog=[bowl, spoon, car],
            embed=False,
            win_mode=False,
        )
    )
    assert calls == [bowl.text, spoon.text, car.text]


def test_highest_keeps_climbing_after_the_first_chain_stalls(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.6, "wordCount": 1}]),
        ),
    )
    bowl, spoon, car = Atom("🥣", "bowl of cereal"), Atom("🥄", "spoon"), Atom("🚗", "car")
    pair = check_word(spoon.text + ZWJ + car.text)

    def value(phrase: str) -> float:
        if phrase == pair:
            return 0.9
        return {bowl.text: 0.5, spoon.text: 0.4, car.text: 0.2}.get(phrase, 0.3)

    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [value(phrase)] * n)

    phrase = asyncio.run(
        search(
            db,
            "cereal",
            budget=20,
            log=lambda _message: None,
            score=score,
            catalog=[bowl, spoon, car],
            embed=False,
            win_mode=False,
        )
    )
    assert pair in calls
    assert phrase == pair

    calls.clear()
    asyncio.run(
        search(
            db,
            "cereal",
            board=SHORTEST_YES,
            budget=20,
            log=lambda _message: None,
            score=score,
            catalog=[bowl, spoon, car],
            embed=False,
            win_mode=False,
        )
    )
    assert pair not in calls


def test_publish_splits_emoji_words_and_keeps_a_zwj_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    sent: list[str] = []

    class Site:
        def leaderboards(self, revision_id, mode):
            return {"leaderboards": {HIGH_SCORES: []}}

        def attempt(self, revision_id, mode):
            return None

        def start(self, revision_id, mode):
            return {"revisionId": revision_id, "playMode": mode, "nextTurn": 1, "tokens": [], "turns": []}

        def turn(self, attempt, operation):
            tokens = [token for token in attempt.get("tokens") or [] if token.get("active", True)]
            if operation["kind"] == "append":
                sent.append(operation["text"])
                tokens.append({"id": f"t{len(sent)}", "text": operation["text"], "active": True})
            turn = {"probability": 0.4}
            return {**attempt, "tokens": tokens, "turns": [turn], "nextTurn": 2}, turn

    live = {"cereal": {"slug": "cereal", "revisionId": "r1", "goal": "yes", "kind": "noul", "choices": []}}

    def entry(phrase: str, units: int) -> dict:
        saved = vault.save(
            "cereal",
            EMOJI,
            phrase,
            p_mean=0.8,
            p_lcb=0.7,
            spread=0.01,
            n=3,
            units=units,
            leader=None,
            beats=True,
            title="cereal",
        )
        return saved | {"slug": "cereal", "mode": EMOJI, "board": HIGH_SCORES}

    publish_entry(db, Site(), entry("😀 🥣", 2), live, "me", False, 0, False, lambda _message: None)
    assert sent == ["😀", "🥣"]
    sent.clear()
    chain = check_phrase("😀\u200d🥣")
    publish_entry(db, Site(), entry(chain, 1), live, "me", False, 0, False, lambda _message: None)
    assert sent == [chain]


def test_crosspost_never_targets_emoji(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    vault.save(
        "cereal",
        STRICT,
        "cereal is soup",
        p_mean=0.8,
        p_lcb=0.7,
        spread=0.01,
        n=3,
        units=3,
        leader=None,
        beats=True,
        title="cereal",
    )
    found = picks(db)
    assert found
    assert {pick.mode.play_mode for pick in found} == {"golf"}
    default = inspect.signature(picks).parameters["modes"].default
    assert all(mode.play_mode != EMOJI for mode in default)


def test_emoji_setup_lists_the_question_and_the_budget(monkeypatch):
    from textual.app import App, ComposeResult
    from textual.widgets import Select

    from jevlab.boards import Standing
    from jevlab.modes import EMOJI_HIGHEST
    from jevlab.objective import Leader
    from jevlab.tui.emoji import EmojiSetupScreen

    rows = [
        Standing("cereal", "Is cereal a soup", "noul", "yes", Leader(0.4, 3, "Ada"), None, None, False, False),
        Standing("blank", "Empty board", "noul", "yes", None, None, None, False, False),
        Standing("ours", "We lead this", "noul", "yes", Leader(0.2, 2, "Bea"), Leader(0.9, 1), None, True, False),
        Standing("flavor", "Best flavor", "choice", "yes", Leader(0.5, 4, "Ada"), None, None, False, False),
    ]
    monkeypatch.setattr("jevlab.tui.emoji.standings", lambda *args, **kwargs: rows)
    monkeypatch.setattr("jevlab.tui.emoji.DB", lambda: None)

    def chosen(picker) -> set[str]:
        return set(picker.selected)

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.pause()
            screen = app.screen
            picker = screen.query_one("#picker")
            shown = []
            for index in range(picker.option_count):
                prompt = picker.get_option_at_index(index).prompt
                shown.append(prompt.plain if hasattr(prompt, "plain") else str(prompt))
            assert any("Is cereal a soup" in line for line in shown)
            budget = screen.query_one("#budget")
            assert budget.value == "40"
            assert budget.region.width >= 16
            assert budget.content_region.width >= 6
            assert screen.query_one("#win-mode").value is True
            await pilot.click("#win-mode")
            await pilot.pause()
            assert screen.query_one("#win-mode").value is False
            assert screen.query_one("EditionMark").content.plain.endswith(" | Emoji Highest Mode")
            assert chosen(picker) == {"cereal", "blank", "flavor"}
            await pilot.click("#none")
            await pilot.pause()
            assert chosen(picker) == set()
            await pilot.click("#all")
            await pilot.pause()
            assert chosen(picker) == {"cereal", "blank", "ours", "flavor"}
            await pilot.click("#not-leading")
            await pilot.pause()
            assert chosen(picker) == {"cereal", "blank", "flavor"}
            await pilot.click("#empty")
            await pilot.pause()
            assert chosen(picker) == {"blank"}
            await pilot.click("#no-choice")
            await pilot.pause()
            assert all("Best flavor" not in line for line in _shown(picker))
            screen.query_one("#target-player", Select).value = "Ada"
            await pilot.pause()
            assert chosen(picker) == {"cereal"}
            style = picker.get_visual_style("option-list--option")
            original = picker._get_option_render(picker.get_option("cereal"), style)[0].text
            assert "WIN" not in original
            screen.apply_progress(
                {"slug": "cereal", "state": "running", "best": "0.800/1w", "win": True, "phrase": "🥣"}
            )
            marked = picker.get_option("cereal").prompt
            marked_text = marked.plain if hasattr(marked, "plain") else str(marked)
            assert "0.800/1w" in marked_text
            assert "WIN" in marked_text
            assert "🥣" in marked_text
            drawn = picker._get_option_render(picker.get_option("cereal"), style)[0].text
            assert "WIN" in drawn
            screen.apply_progress({"slug": "blank", "state": "running"})
            drawn_again = picker._get_option_render(picker.get_option("cereal"), style)[0].text
            assert "WIN" in drawn_again
            assert screen.query_one("#stop").disabled
            screen.busy = True
            screen.refresh_bindings()
            screen.query_one("#stop").disabled = False
            await pilot.click("#stop")
            assert screen.stop_requested
            screen.stop_requested = False
            await pilot.press("s")
            assert screen.stop_requested

    def _shown(picker) -> list[str]:
        lines = []
        for index in range(picker.option_count):
            prompt = picker.get_option_at_index(index).prompt
            lines.append(prompt.plain if hasattr(prompt, "plain") else str(prompt))
        return lines

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(EmojiSetupScreen(EMOJI_HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    asyncio.run(check())
