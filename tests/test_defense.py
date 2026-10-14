"""Defense picks decoys from other boards and posts each one by itself."""

import asyncio
import json
import re

from textual.app import App, ComposeResult
from textual.widgets import Button

from jevlab.db import DB
from jevlab.defense import QUIET, READY, already_quiet, defend, lexicon, pick_decoys, safe_to_clear, scan
from jevlab.modes import HIGHEST
from jevlab.tui.defense import DefenseLog, DefenseScreen
from jevlab.tui.pickers import HomeScreen


def _db(tmp_path) -> DB:
    db = DB(tmp_path / "t.db")
    db.execute("INSERT INTO snapshots (taken_at, edition, me, raw_dir) VALUES ('t', 'jev', 'me', '')")
    return db


def _question(db: DB, slug: str, title: str = "Title") -> None:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, 'r1', ?, '', 'noul', 'yes', 0.2, 0.5, 'm', '{}', '{}')",
        (slug, title),
    )


def _lead(db: DB, slug: str, board: str, phrase: str, probability: float = 0.99, mode: str = "strict_chain") -> None:
    rows = [
        {
            "userId": "me",
            "name": "us",
            "probability": probability,
            "wordCount": len(phrase.split()),
            "phrase": phrase,
        },
        {"userId": "other", "name": "them", "probability": 0.4, "wordCount": 8, "phrase": "nope"},
    ]
    db.execute("INSERT INTO boards VALUES (?, ?, ?, 0, ?)", (slug, mode, board, json.dumps(rows)))


def _impacts(db: DB, slug: str, rows: list[dict], mode: str = "strict_chain") -> None:
    db.execute("INSERT INTO word_impacts VALUES (?, ?, 0, ?)", (slug, mode, json.dumps(rows)))


def test_lexicon_dedupes_and_prefers_words_that_travel():
    ranked = lexicon(
        [
            [{"word": "Gamma", "averageImpact": 0.2}, {"word": "gamma", "averageImpact": 0.1}],
            [{"word": "gamma", "averageImpact": 0.9}, {"word": "solo", "averageImpact": 0.99}],
        ]
    )
    assert ranked[0].casefold() == "gamma"
    assert ranked.count("gamma") + ranked.count("Gamma") == 1
    assert ranked.index("solo") > 0


def test_pick_decoys_skips_stop_words():
    decoys = pick_decoys(
        ["the", "that", "with", "not", "The", "and", "is", "a", "gamma", "exceptions"],
        present=set(),
        protected=set(),
        top_n=5,
    )
    assert decoys == ["gamma", "exceptions"]


def test_pick_decoys_skips_our_words_words_already_there_and_answer_fragments():
    decoys = pick_decoys(
        ["alpha", "beta", "gamma", "Gamma", "delta", "not a word", "epsilon"],
        present={"beta"},
        protected={"alpha"},
        top_n=2,
        choices=[{"option": "delta"}],
    )
    assert decoys == ["gamma", "epsilon"]


def test_already_quiet_only_when_the_board_is_crowded_and_our_words_are_buried():
    assert not already_quiet(0, 0, 0)
    assert not already_quiet(0, 10, 39)
    assert already_quiet(0, 20, 40)
    assert already_quiet(5, 20, 40)
    assert not already_quiet(6, 20, 40)


def test_safe_to_clear_keeps_an_unrelated_chain():
    assert safe_to_clear([], ["alpha beta"])
    assert safe_to_clear(["left"], ["alpha beta"])
    assert safe_to_clear(["alpha", "beta"], ["alpha beta"])
    assert not safe_to_clear(["secret", "sauce"], ["alpha beta"])


def test_decoys_open_with_the_question_and_its_synonyms(tmp_path):
    db = _db(tmp_path)
    _question(db, "steak", "Is it okay to season your steak with grandpa's ashes")
    _lead(db, "steak", "highScores", "alpha beta")
    _impacts(db, "steak", [{"word": "alpha", "averageImpact": 0.9}])
    _impacts(db, "pool", [{"word": "database", "averageImpact": 0.99}, {"word": "gamma", "averageImpact": 0.2}])

    decoys = next(target.decoys for target in scan(db, top_n=20) if target.slug == "steak")
    assert list(decoys[:5]) == ["okay", "season", "steak", "grandpa", "ashes"]
    assert "flavor" in decoys
    assert "grandfather" in decoys
    assert "ash" in decoys
    assert decoys.index("flavor") < decoys.index("database")
    assert decoys.index("grandfather") < decoys.index("database")
    assert decoys.index("ash") < decoys.index("database")
    assert "alpha" not in decoys
    assert "beta" not in decoys


def test_scan_preselects_a_leaking_board_and_holds_back_the_rest(tmp_path):
    db = _db(tmp_path)
    _question(db, "q", "The")
    _question(db, "crowd", "The")
    _question(db, "short", "The")
    _question(db, "casual", "The")
    _lead(db, "q", "highScores", "alpha beta")
    _lead(db, "crowd", "highScores", "alpha beta")
    _lead(db, "short", "shortestYes", "one two three")
    _lead(db, "casual", "highScores", "alpha beta", mode="word_chain")
    _impacts(db, "q", [{"word": "alpha", "averageImpact": 0.5}, {"word": "beta", "averageImpact": 0.4}])
    _impacts(
        db,
        "pool",
        [{"word": "gamma", "averageImpact": 0.8}, {"word": "delta", "averageImpact": 0.2}],
    )
    crowd = [{"word": f"w{i:02d}", "averageImpact": 0.05} for i in range(40)]
    crowd += [{"word": "alpha", "averageImpact": 0.01}, {"word": "beta", "averageImpact": 0.01}]
    _impacts(db, "crowd", crowd)
    _impacts(db, "short", [{"word": "one", "averageImpact": 0.01}])
    _impacts(db, "casual", [{"word": "alpha", "averageImpact": 0.5}], mode="word_chain")

    rows = {target.slug: target for target in scan(db, top_n=20)}
    assert set(rows) == {"q", "crowd", "short"}
    assert rows["q"].status == READY
    assert rows["q"].selected
    assert rows["q"].decoys[:2] == ("gamma", "delta")
    assert "alpha" not in rows["q"].decoys
    assert rows["crowd"].status == QUIET
    assert not rows["crowd"].selected
    assert rows["short"].status == READY
    assert rows["short"].selected

    with_casual = {target.slug: target.play_mode for target in scan(db, top_n=20, include_casual=True)}
    assert with_casual["casual"] == "word_chain"


class _Site:
    def __init__(self, lead=True, choices=None, chain=None):
        self.lead = lead
        self.choices = choices or []
        self.ops: list[str] = []
        tokens = [{"id": f"id{i}", "text": word, "active": True} for i, word in enumerate(chain or [])]
        self.current = {
            "id": "a",
            "nextTurn": 1,
            "revisionId": "r1",
            "playMode": "strict_chain",
            "tokens": tokens,
            "turns": [],
        }

    def viewer(self):
        return {"player": {"user": {"id": "me"}}}

    def menu(self):
        return {
            "questions": [{"slug": "q", "revisionId": "r1", "kind": "noul", "goal": "yes", "choices": self.choices}]
        }

    def leaderboards(self, _rev, _mode):
        rows = [{"userId": "other", "name": "them", "probability": 0.2, "wordCount": 6, "phrase": "nope here now"}]
        if self.lead:
            rows.insert(0, {"userId": "me", "name": "us", "probability": 0.99, "wordCount": 2, "phrase": "alpha beta"})
        return {"leaderboards": {"highScores": rows, "shortestYes": []}}

    def words(self, _rev, _mode):
        return [{"word": "alpha", "averageImpact": 0.4}, {"word": "beta", "averageImpact": 0.3}]

    def attempt(self, _rev, _mode):
        return self.current

    def start(self, _rev, _mode):
        return self.current

    def turn(self, attempt, operation):
        self.ops.append(operation["kind"] + ":" + str(operation.get("text") or ""))
        tokens = [token for token in attempt.get("tokens") or [] if token.get("active", True)]
        if operation["kind"] == "clear":
            tokens = []
        elif operation["kind"] == "append":
            tokens.append({"id": f"t{len(self.ops)}", "text": operation["text"], "active": True})
        updated = dict(attempt)
        updated["tokens"] = tokens
        updated["turns"] = list(attempt.get("turns") or []) + [{"id": f"n{len(self.ops)}", "probability": 0.22}]
        updated["nextTurn"] = int(attempt.get("nextTurn") or 1) + 1
        self.current = updated
        return updated, updated["turns"][-1]


def _pool(db: DB) -> None:
    _impacts(db, "q", [{"word": "alpha", "averageImpact": 0.4}, {"word": "beta", "averageImpact": 0.3}])
    _impacts(db, "other", [{"word": "gamma", "averageImpact": 0.8}, {"word": "delta", "averageImpact": 0.5}])


def test_defend_posts_each_decoy_alone_and_does_not_repeat_it(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site()
    result = defend(
        [("q", "strict_chain")], top_n=20, log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock"
    )
    assert result.posted == 2
    assert site.ops == ["append:gamma", "clear:", "append:delta", "clear:"]
    again = defend(
        [("q", "strict_chain")], top_n=20, log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock"
    )
    assert again.posted == 0
    assert site.ops == ["append:gamma", "clear:", "append:delta", "clear:"]


def test_defend_skips_a_live_chain_that_is_not_our_phrase(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site(chain=["secret", "sauce"])
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert result.posted == 0
    assert site.ops == []


def test_defend_dry_run_sends_nothing(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site()
    result = defend(
        [("q", "strict_chain")], dry_run=True, log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock"
    )
    assert result.planned == 1
    assert result.posted == 0
    assert site.ops == []


def test_defend_skips_a_board_we_no_longer_lead(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site(lead=False)
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert result.posted == 0
    assert result.skipped == 1
    assert site.ops == []


def test_defense_log_marks_each_board_with_a_cyan_counter():
    dry = DefenseLog()
    opening = dry.render("DRY RUN: defending 54 board(s), top 20")
    assert "[1/" not in opening.plain
    first = dry.render("\n== steak [strict_chain] Highest")
    assert "[1/54]" in first.plain
    assert "steak" in first.plain and "Highest" in first.plain
    counter = next(span for span in first.spans if "[1/54]" in first.plain[span.start : span.end])
    assert "cyan" in str(counter.style)

    live = DefenseLog()
    live.render("defending 2 board(s), top 20")
    shown = live.render("\n== other [word_chain] not leading")
    assert "[1/2]" in shown.plain
    word = live.render("    + database 0.04")
    assert re.search(r"\[\d+/\d+\]", word.plain) is None
    assert "+ database" in word.plain
    again = live.render("\n== third [strict_chain] Shortest")
    assert "[2/2]" in again.plain


def test_home_opens_defense(monkeypatch, tmp_path):
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "home.db"))

    class _App(App):
        def __init__(self):
            super().__init__()
            self.game_mode = HIGHEST

        def on_mount(self) -> None:
            self.push_screen(HomeScreen())

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.pause()
            button = app.screen.query_one("#defense", Button)
            assert str(button.label) == "Defense"
            app.screen.action_defense()
            await pilot.pause()
            assert isinstance(app.screen, DefenseScreen)

    asyncio.run(check())


def test_defense_screen_preselects_only_ready_boards(monkeypatch, tmp_path):
    db = _db(tmp_path)
    _question(db, "q", "Ready")
    _question(db, "short", "Short")
    _lead(db, "q", "highScores", "alpha beta")
    _lead(db, "short", "shortestYes", "one two three")
    _pool(db)
    _impacts(db, "short", [{"word": "one", "averageImpact": 0.4}])
    monkeypatch.setattr("jevlab.tui.defense.DB", lambda: db)
    monkeypatch.setattr("jevlab.activity.log", lambda *_args, **_kwargs: None)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(DefenseScreen())

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            selected = set(app.screen.query_one("#def-picker").selected)
            assert selected == {"strict_chain:q", "strict_chain:short"}

    asyncio.run(check())
