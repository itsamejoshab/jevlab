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
from jevlab.emoji.learn import TRUST_RHO, Model, fit
from jevlab.emoji.rank import Atom, shortlist
from jevlab.emoji.rules import RuleError, check_phrase, check_word
from jevlab.emoji.search import load_catalog, search
from jevlab.modes import EMOJI, HIGH_SCORES, STRICT
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


def _question(db: DB, slug: str = "cereal") -> None:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', 'Is cereal a soup?', '', 'noul', 'yes', 0.2, 0.5, 'm', ?, ?)",
        (slug, json.dumps({"state": ""}), json.dumps({"goal": "yes", "kind": "noul"})),
    )


def test_search_stops_at_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    catalog = [Atom(f"{index}\u20e3", f"digit {index}") for index in range(6)]
    phrase = asyncio.run(
        search(db, "cereal", budget=3, log=lambda _message: None, score=score, catalog=catalog, embed=False)
    )
    assert len(calls) <= 3
    assert phrase is not None


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

    from jevlab.boards import Standing
    from jevlab.modes import EMOJI_HIGHEST
    from jevlab.objective import Leader
    from jevlab.tui.emoji import EmojiSetupScreen

    rows = [Standing("cereal", "Is cereal a soup", "noul", "yes", Leader(0.4, 3, "Ada"), None, None, False, False)]
    monkeypatch.setattr("jevlab.tui.emoji.standings", lambda *args, **kwargs: rows)
    monkeypatch.setattr("jevlab.tui.emoji.DB", lambda: None)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(EmojiSetupScreen(EMOJI_HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            picker = screen.query_one("#picker")
            prompt = picker.get_option_at_index(0).prompt
            text = prompt.plain if hasattr(prompt, "plain") else str(prompt)
            assert "Is cereal a soup" in text
            assert screen.query_one("#budget").value == "40"
            assert screen.query_one("EditionMark").content.plain.endswith(" | Emoji Highest Mode")

    asyncio.run(check())
