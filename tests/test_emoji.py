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
from jevlab.emoji.lexicon import Index
from jevlab.emoji.project import glyph_for, project_phrase, projections
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


@pytest.fixture(autouse=True)
def _empty_edition_vault(tmp_path, monkeypatch):
    """Search reads this edition's Strict vault. Tests must not see the real one."""
    monkeypatch.setattr("jevlab.transfer.edition_vault", lambda _edition: tmp_path / "edition-vault")


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


def _plant_strict(tmp_path, monkeypatch, phrase: str, p_mean: float = 0.8) -> None:
    root = tmp_path / "vault"
    monkeypatch.setattr(vault, "VAULT", root)
    monkeypatch.setattr("jevlab.transfer.edition_vault", lambda _edition: root)
    vault.save(
        "cereal",
        STRICT,
        phrase,
        p_mean=p_mean,
        p_lcb=p_mean - 0.05,
        spread=0.01,
        n=3,
        units=len(phrase.split()),
        leader=None,
        beats=True,
        title="cereal",
    )


def test_projection_keeps_phrase_order_and_drops_a_stop():
    index = Index.build(
        [Atom("🚫", "no entry sign"), Atom("🥄", "spoon"), Atom("🚗", "car")]
    )
    text = project_phrase("never the spoon car", {"spoon": 0.5, "car": 0.4}, index)
    assert text == "🚫 🥄 🚗"


def test_cosine_below_the_floor_does_not_map():
    index = Index.build([Atom("😀", "grinning face"), Atom("😁", "beaming face")])
    names = {
        "😀": np.array([1.0, 0.0, 0.0]),
        "😁": np.array([0.0, 1.0, 0.0]),
    }
    assert glyph_for("mystery", index, names, np.array([0.0, 0.0, 1.0])) is None


def test_second_projection_skips_a_matching_bearing_set():
    index = Index.build([Atom("🚫", "no entry sign"), Atom("🥄", "spoon")])
    lines = [{"phrase": "never spoon"}, {"phrase": "spoon never"}]
    assert projections(lines, {"spoon": 1.0}, index) == ["🚫 🥄"]


def test_project_scores_the_spaced_pair_and_vaults_it(tmp_path, monkeypatch):
    db = DB(tmp_path / "t.db")
    _question(db)
    _plant_strict(tmp_path, monkeypatch, "spoon car")
    db.execute(
        "INSERT INTO word_impacts VALUES (?, ?, 0, ?)",
        (
            "cereal",
            STRICT,
            json.dumps(
                [
                    {"word": "spoon", "averageImpact": 0.4},
                    {"word": "car", "averageImpact": 0.3},
                ]
            ),
        ),
    )
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.6, "wordCount": 1}]),
        ),
    )
    spoon, car = Atom("🥄", "spoon"), Atom("🚗", "car")
    pair = f"{spoon.text} {car.text}"
    joined = spoon.text + ZWJ + car.text

    def value(phrase: str) -> float:
        if phrase == pair:
            return 0.9
        return 0.2

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
            catalog=[spoon, car],
            embed=False,
            win_mode=False,
        )
    )
    assert pair in calls
    assert phrase == pair
    assert joined not in calls or value(joined) < 0.9

    calls.clear()
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            SHORTEST_YES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.8, "wordCount": 1}]),
        ),
    )

    async def low(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            board=SHORTEST_YES,
            budget=20,
            log=lambda _message: None,
            score=low,
            catalog=[spoon, car],
            embed=False,
            win_mode=False,
        )
    )
    assert all(ZWJ not in phrase for phrase in calls)


def test_usage_atom_joins_singles_and_a_raising_probe_adds_none(tmp_path, monkeypatch):
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
    bowl = Atom("🥣", "bowl")
    fire = Atom("🔥", "fire")
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            budget=10,
            log=lambda _message: None,
            score=score,
            catalog=[bowl],
            embed=False,
            usage=lambda _phrase, _index: [fire],
        )
    )
    assert fire.text in calls

    calls.clear()

    def boom(_phrase, _index):
        raise RuntimeError("probe down")

    asyncio.run(
        search(
            db,
            "cereal",
            budget=10,
            log=lambda _message: None,
            score=score,
            catalog=[bowl],
            embed=False,
            usage=boom,
        )
    )
    assert bowl.text in calls
    assert fire.text not in calls


def test_usage_glyphs_are_empty_when_the_model_is_not_cached(monkeypatch):
    from jevlab.emoji import usage

    def missing():
        usage._PROBLEM = "model not cached"
        usage._BUNDLE = None
        usage._TRIED = True
        return None

    monkeypatch.setattr(usage, "_load", missing)
    found = usage.usage_glyphs("the soup is hot", Index.build([Atom("🔥", "fire")]))
    assert found == []
    assert usage.usage_problem() == "model not cached"


def test_embed_false_does_not_load_an_embedder_or_bertweet(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    db = DB(tmp_path / "t.db")
    _question(db)

    def boom(*_args, **_kwargs):
        raise AssertionError("model loaded")

    import sys

    module = sys.modules["jevlab.emoji.search"]
    monkeypatch.setattr(module, "_local_embed", boom)
    monkeypatch.setattr(module, "question_vector", boom)
    monkeypatch.setattr(module, "atom_vectors", boom)
    monkeypatch.setattr(module, "load_model", boom)
    monkeypatch.setattr(module, "usage_glyphs", boom)

    async def score(phrase: str, n: int = 1) -> Score:
        return Score(phrase, [0.2] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            budget=1,
            log=lambda _message: None,
            score=score,
            catalog=[Atom("🥣", "bowl")],
            embed=False,
        )
    )


def test_compact_drops_a_glyph_whose_round_holds(tmp_path, monkeypatch):
    db = DB(tmp_path / "t.db")
    _question(db)
    _plant_strict(tmp_path, monkeypatch, "alpha beta gamma")
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.7, "wordCount": 1}]),
        ),
    )
    alpha, beta, gamma = Atom("😀", "alpha"), Atom("😁", "beta"), Atom("😂", "gamma")
    full = f"{alpha.text} {beta.text} {gamma.text}"
    kept = f"{alpha.text} {gamma.text}"
    fallen = f"{alpha.text} {beta.text}"

    def value(phrase: str) -> float:
        if phrase in {full, kept}:
            return 0.80
        return 0.40

    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [value(phrase)] * n)

    phrase = asyncio.run(
        search(
            db,
            "cereal",
            budget=40,
            log=lambda _message: None,
            score=score,
            catalog=[alpha, beta, gamma],
            embed=False,
        )
    )
    assert kept in calls
    assert fallen in calls
    assert phrase == kept


def test_call_count_never_exceeds_the_budget(tmp_path, monkeypatch):
    db = DB(tmp_path / "t.db")
    _question(db)
    _plant_strict(tmp_path, monkeypatch, "spoon automobile")
    db.execute(
        "INSERT INTO boards VALUES (?, ?, ?, 0, ?)",
        (
            "cereal",
            EMOJI,
            HIGH_SCORES,
            json.dumps([{"userId": "them", "name": "them", "probability": 0.9, "wordCount": 1}]),
        ),
    )
    catalog = [Atom("🥄", "spoon"), Atom("🚗", "automobile"), Atom("🥣", "cereal bowl")]
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            budget=5,
            log=lambda _message: None,
            score=score,
            catalog=catalog,
            embed=False,
        )
    )
    assert calls
    assert len(calls) <= 5


def test_patience_ends_the_question_after_that_many_stalls(tmp_path, monkeypatch):
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
    catalog = [Atom(f"{index}\u20e3", f"digit {index}") for index in range(30)]
    notes: list[str] = []
    calls: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        calls.append(phrase)
        return Score(phrase, [0.2] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            budget=40,
            log=notes.append,
            score=score,
            catalog=catalog,
            embed=False,
            patience=1,
        )
    )
    one = len(calls)
    assert one <= 40
    assert any("patience 1/1" in line for line in notes)

    calls.clear()
    asyncio.run(
        search(
            db,
            "cereal",
            budget=40,
            log=lambda _message: None,
            score=score,
            catalog=catalog,
            embed=False,
            patience=2,
        )
    )
    assert len(calls) > one
    assert len(calls) <= 40


def test_a_better_top_score_resets_patience(tmp_path, monkeypatch):
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
    catalog = [Atom(f"{index}\u20e3", f"digit {index}") for index in range(30)]
    notes: list[str] = []
    two_word = 0

    async def score(phrase: str, n: int = 1) -> Score:
        nonlocal two_word
        if len(phrase.split()) != 2:
            return Score(phrase, [0.2] * n)
        two_word += 1
        # The first pair becomes the top line. A later pair is only 0.004 higher,
        # still the same rounded score, and that has to zero patience again.
        return Score(phrase, [0.304 if two_word == 8 else 0.3] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            budget=80,
            log=notes.append,
            score=score,
            catalog=catalog,
            embed=False,
            patience=2,
        )
    )
    resets = [index for index, line in enumerate(notes) if line.strip() == "patience 0"]
    ended = next(index for index, line in enumerate(notes) if "patience 2/2" in line)
    assert len(resets) == 2
    assert resets[-1] < ended


def test_compact_runs_after_patience_on_a_two_glyph_line(tmp_path, monkeypatch):
    db = DB(tmp_path / "t.db")
    _question(db)
    _plant_strict(tmp_path, monkeypatch, "alpha beta")
    alpha, beta = Atom("😀", "alpha"), Atom("😁", "beta")
    pair = f"{alpha.text} {beta.text}"
    notes: list[str] = []

    async def score(phrase: str, n: int = 1) -> Score:
        return Score(phrase, [0.80 if phrase == pair else 0.20] * n)

    asyncio.run(
        search(
            db,
            "cereal",
            budget=40,
            log=notes.append,
            score=score,
            catalog=[alpha, beta, Atom("😂", "gamma")],
            embed=False,
            patience=1,
        )
    )
    assert any(line.strip() == "compact" for line in notes)
    assert any("patience 1/1" in line for line in notes)


def test_emoji_go_opens_the_lab(monkeypatch):
    from textual.app import App, ComposeResult

    from jevlab.boards import Standing
    from jevlab.modes import EMOJI_HIGHEST
    from jevlab.objective import Leader
    from jevlab.tui.emoji import EmojiSetupScreen
    from jevlab.tui.emoji_lab import EmojiLabScreen

    rows = [
        Standing("cereal", "Is cereal a soup", "noul", "yes", Leader(0.4, 3, "Ada"), None, None, False, False),
    ]
    monkeypatch.setattr("jevlab.tui.emoji.standings", lambda *args, **kwargs: rows)
    monkeypatch.setattr("jevlab.tui.emoji.DB", lambda: None)
    monkeypatch.setattr("jevlab.tui.emoji_lab.DB", lambda: object())
    monkeypatch.setattr("jevlab.tui.emoji_lab.standings", lambda *args, **kwargs: [])

    async def fake_search(*_args, **kwargs):
        progress = kwargs["progress"]
        progress(
            {
                "slug": "cereal",
                "state": "batch",
                "arm": "project",
                "calls": 1,
                "budget": kwargs["budget"],
                "scored": [{"phrase": "🥣", "p": 0.8, "units": 1, "arm": "project", "win": True}],
                "best": "0.800/1w",
                "win": True,
                "phrase": "🥣",
            }
        )
        progress(
            {
                "slug": "cereal",
                "state": "done",
                "arm": "project",
                "calls": 1,
                "budget": kwargs["budget"],
                "best": "0.800/1w",
                "win": True,
                "phrase": "🥣",
            }
        )
        return "🥣"

    monkeypatch.setattr("jevlab.tui.emoji_lab.search", fake_search)

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#go")
            for _ in range(40):
                await pilot.pause()
                if isinstance(app.screen, EmojiLabScreen) and not app.screen.busy:
                    break
            screen = app.screen
            assert isinstance(screen, EmojiLabScreen)
            assert screen.query_one("#feed").border_title == "now trying"
            assert "project" in str(screen.query_one("#insight").content)
            assert screen.query_one("#table").row_count == 1
            win_cell = screen.query_one("#table").get_row_at(0)[4]
            assert win_cell.plain == "WIN"
            assert "green" in repr(win_cell)
            assert "WIN" in str(screen.query_one("#lab-header").content)
            assert "WIN" in str(screen.query_one("#queue").content)
            assert "green" in repr(screen.query_one("#queue").content)
            await pilot.press("q")
            await pilot.pause()
            assert isinstance(app.screen, EmojiSetupScreen)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(EmojiSetupScreen(EMOJI_HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    asyncio.run(check())


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
            assert screen.query_one("#patience").value == "6"
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
