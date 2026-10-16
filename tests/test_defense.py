"""Defense picks decoys from other boards and posts them as one sentence."""

import asyncio
import json
import re
import threading

from textual.app import App, ComposeResult
from textual.widgets import Button, Select

from jevlab.db import DB
from jevlab.defense import QUIET, READY, DefenseRun, already_quiet, defend, lexicon, pick_decoys, scan
from jevlab.rules import banned
from jevlab.rules.banned import BannedPhrase
from jevlab.modes import HIGHEST
from jevlab.tui.defense import DefenseLog, DefenseScreen
from jevlab.tui.pickers import HomeScreen
from jevlab.tui.publish_log import TURBO_BANNER


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


def test_decoys_open_with_the_question_and_its_synonyms(tmp_path):
    db = _db(tmp_path)
    _question(db, "steak", "Is it okay to season your steak with grandpa's ashes")
    _lead(db, "steak", "highScores", "alpha beta")
    _impacts(db, "steak", [{"word": "alpha", "averageImpact": 0.9}])
    _impacts(db, "pool", [{"word": "database", "averageImpact": 0.99}, {"word": "gamma", "averageImpact": 0.2}])

    decoys = next(target.decoys for target in scan(db) if target.slug == "steak")
    assert list(decoys[:5]) == ["okay", "season", "steak", "grandpa", "ashes"]
    assert "flavor" in decoys
    assert "grandfather" in decoys
    assert "ash" in decoys
    assert decoys.index("flavor") < decoys.index("database")
    assert decoys.index("grandfather") < decoys.index("database")
    assert decoys.index("ash") < decoys.index("database")
    assert "alpha" not in decoys
    assert "beta" not in decoys


def test_scan_holds_back_a_solo_board_once_the_answer_is_a_quarter_of_the_list(tmp_path):
    """Low-impact decoys never enter the top, so a long list has to count even when the headline is still the answer."""
    db = _db(tmp_path)
    _question(db, "solo", "The")
    _question(db, "thin", "The")
    answer = [f"ans{i:02d}" for i in range(19)]
    phrase = " ".join(answer)
    _lead(db, "solo", "highScores", phrase)
    _lead(db, "thin", "highScores", phrase)

    def listed(extra: int) -> list[dict]:
        rows = [{"word": word, "averageImpact": 1 - index * 0.01} for index, word in enumerate(answer)]
        rows += [{"word": f"decoy{i:02d}", "averageImpact": 0.01} for i in range(extra)]
        return rows

    # 19 answer words are a quarter of 76, and still more than a quarter of 75.
    _impacts(db, "solo", listed(57))
    _impacts(db, "thin", listed(56))
    _impacts(db, "pool", [{"word": "gamma", "averageImpact": 0.8}])

    rows = {target.slug: target for target in scan(db)}
    assert rows["solo"].plan.ours_in_top == 19
    assert rows["solo"].plan.top_len == 20
    assert rows["solo"].status == QUIET
    assert not rows["solo"].selected
    assert rows["thin"].status == READY
    assert rows["thin"].selected


def test_scan_sizes_each_board_so_one_pass_covers_the_answer(tmp_path):
    """A two-word answer needs enough decoys to reach 40 listed words; a 19-word answer needs enough to reach 76."""
    db = _db(tmp_path)
    _question(db, "short", "The")
    _question(db, "long", "The")
    _question(db, "gap", "The")
    _lead(db, "short", "highScores", "alpha beta")
    answer = [f"ans{i:02d}" for i in range(19)]
    _lead(db, "long", "highScores", " ".join(answer))
    _lead(db, "gap", "highScores", " ".join(answer))
    _impacts(db, "short", [{"word": "alpha", "averageImpact": 0.9}, {"word": "beta", "averageImpact": 0.8}])
    _impacts(db, "long", [{"word": word, "averageImpact": 1 - i * 0.01} for i, word in enumerate(answer)])
    gap = [{"word": word, "averageImpact": 1 - i * 0.01} for i, word in enumerate(answer)]
    gap += [{"word": f"decoy{i:02d}", "averageImpact": 0.01} for i in range(56)]
    _impacts(db, "gap", gap)
    _impacts(db, "pool", [{"word": f"pool{i:02d}", "averageImpact": 0.5} for i in range(80)])

    rows = {target.slug: target for target in scan(db)}
    assert len(rows["short"].decoys) == 43
    assert rows["short"].selected
    assert "alpha" not in rows["short"].decoys
    assert len(rows["long"].decoys) == 62
    assert rows["long"].selected
    assert len(rows["gap"].decoys) == 6
    assert rows["gap"].selected


def test_scan_preselects_a_leaking_board_and_holds_back_the_rest(tmp_path):
    db = _db(tmp_path)
    _question(db, "q", "The")
    _question(db, "crowd", "The")
    _question(db, "short", "The")
    _question(db, "casual", "The")
    _question(db, "golf", "The")
    _lead(db, "q", "highScores", "alpha beta")
    _lead(db, "crowd", "highScores", "alpha beta")
    _lead(db, "short", "shortestYes", "one two three")
    _lead(db, "casual", "highScores", "alpha beta", mode="word_chain")
    _lead(db, "golf", "highScores", "alpha beta", mode="golf")
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
    _impacts(db, "golf", [{"word": "alpha", "averageImpact": 0.5}], mode="golf")

    rows = {target.slug: target for target in scan(db)}
    assert set(rows) == {"q", "crowd", "short"}
    assert rows["q"].status == READY
    assert rows["q"].selected
    assert rows["q"].decoys[:2] == ("gamma", "delta")
    assert "alpha" not in rows["q"].decoys
    assert rows["crowd"].status == QUIET
    assert not rows["crowd"].selected
    assert rows["short"].status == READY
    assert rows["short"].selected

    with_casual = {target.slug: target.play_mode for target in scan(db, include_casual=True)}
    assert with_casual["casual"] == "word_chain"
    assert "golf" not in with_casual

    with_golf = {target.slug: target.play_mode for target in scan(db, include_golf=True)}
    assert with_golf["golf"] == "golf"
    assert "casual" not in with_golf


class _Site:
    def __init__(self, lead=True, choices=None, chain=None, phrase="alpha beta", word_rows=None):
        self.lead = lead
        self.phrase = phrase
        self.word_rows = word_rows
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
            rows.insert(
                0,
                {
                    "userId": "me",
                    "name": "us",
                    "probability": 0.99,
                    "wordCount": len(self.phrase.split()),
                    "phrase": self.phrase,
                },
            )
        return {"leaderboards": {"highScores": rows, "shortestYes": []}}

    def words(self, _rev, _mode):
        if self.word_rows is not None:
            return self.word_rows
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


def test_defend_does_not_add_decoys_once_the_answer_is_a_quarter_of_the_list(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    answer = [f"ans{i:02d}" for i in range(19)]
    rows = [{"word": word, "averageImpact": 1 - index * 0.01} for index, word in enumerate(answer)]
    rows += [{"word": f"decoy{i:02d}", "averageImpact": 0.01} for i in range(57)]
    site = _Site(phrase=" ".join(answer), word_rows=rows)
    lines: list[str] = []
    result = defend([("q", "strict_chain")], log=lines.append, db=db, client=site, lock_path=tmp_path / "lock")
    assert result.posted == 0
    assert site.ops == []
    assert any("crowded" in line for line in lines)


def _reset_bans(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(banned, "PATH", tmp_path / "banned_phrases.json")
    monkeypatch.setattr(banned, "_phrases", {})
    monkeypatch.setattr(banned, "_mtime", None)
    monkeypatch.setattr(banned, "_checked", 0.0)


class _BanSite(_Site):
    """Rejects an append that contains `word`. `named` is the span the server quotes, or empty."""

    def __init__(self, word: str, named: str):
        super().__init__()
        self.word = word
        self.named = named

    def turn(self, attempt, operation):
        text = str(operation.get("text") or "")
        if operation["kind"] == "append" and self.word in text.split():
            self.ops.append(operation["kind"] + ":" + text)
            raise BannedPhrase(text, self.named)
        return super().turn(attempt, operation)


def _ban_pool(db) -> None:
    rows = [{"word": "nope", "averageImpact": 0.99}]
    rows += [{"word": f"pool{i:02d}", "averageImpact": 0.5} for i in range(50)]
    _impacts(db, "pool", rows)


def test_defend_omits_a_known_banned_word_and_still_posts_the_rest(tmp_path, monkeypatch):
    _reset_bans(tmp_path, monkeypatch)
    banned.add("nope")
    db = _db(tmp_path)
    _ban_pool(db)
    site = _Site()
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    texts = [op.split(":", 1)[1] for op in site.ops if op.startswith("append:")]
    assert texts
    assert all("nope" not in text.split() for text in texts)
    assert result.posted == 43
    assert site.ops[-1] == "clear:"


def test_defend_skips_an_unnamed_ban_by_submitting_one_word_at_a_time(tmp_path, monkeypatch):
    _reset_bans(tmp_path, monkeypatch)
    db = _db(tmp_path)
    _ban_pool(db)
    site = _BanSite("nope", "")
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert "append:nope" in site.ops
    assert result.posted == 42
    assert site.ops[-1] == "clear:"
    later = site.ops[site.ops.index("append:nope") + 1 :]
    assert later[-1] == "clear:"
    assert all("nope" not in op.split(":", 1)[1].split() for op in later if op.startswith("append:"))


def test_defend_omits_a_named_ban_without_probing_every_word(tmp_path, monkeypatch):
    _reset_bans(tmp_path, monkeypatch)
    db = _db(tmp_path)
    _ban_pool(db)
    site = _BanSite("nope", "nope")
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert "append:nope" not in site.ops
    assert result.posted == 42
    assert site.ops[-1] == "clear:"


def test_defend_omits_a_known_banned_phrase(tmp_path, monkeypatch):
    _reset_bans(tmp_path, monkeypatch)
    banned.add("pool00 pool01")
    db = _db(tmp_path)
    _impacts(db, "pool", [{"word": f"pool{i:02d}", "averageImpact": 0.5} for i in range(50)])
    site = _Site()
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    texts = [op.split(":", 1)[1] for op in site.ops if op.startswith("append:")]
    for text in texts:
        words = text.split()
        pairs = [words[index : index + 2] for index in range(len(words) - 1)]
        assert ["pool00", "pool01"] not in pairs
    assert result.posted == 41
    assert site.ops[-1] == "clear:"


def test_defend_posts_the_whole_cover_in_one_pass(tmp_path):
    db = _db(tmp_path)
    _impacts(db, "pool", [{"word": f"pool{i:02d}", "averageImpact": 0.5} for i in range(40)])
    site = _Site()
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert result.posted == 40
    posted = [op.split(":", 1)[1] for op in site.ops if op.startswith("append:")]
    assert [len(text.split()) for text in posted] == [25, 15]
    assert site.ops[-1] == "clear:"


def test_defend_word_by_word_posts_each_decoy_alone(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site()
    lines: list[str] = []
    result = defend(
        [("q", "strict_chain")],
        log=lines.append,
        db=db,
        client=site,
        lock_path=tmp_path / "lock",
        word_by_word=True,
    )
    assert result.posted == 2
    assert site.ops == ["append:gamma", "append:delta", "clear:"]
    assert any("one word at a time" in line for line in lines)


def test_defend_posts_the_decoys_as_one_sentence_and_does_not_repeat_it(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site()
    lines: list[str] = []
    result = defend([("q", "strict_chain")], log=lines.append, db=db, client=site, lock_path=tmp_path / "lock")
    assert result.posted == 2
    assert site.ops == ["append:gamma delta", "clear:"]
    text = "\n".join(lines)
    assert "gamma delta" in text
    assert "submitting..." in text
    assert "done." in text
    assert "0.22" not in text
    again = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert again.posted == 0
    assert site.ops == ["append:gamma delta", "clear:"]


def test_defend_appends_onto_a_live_chain_and_clears_after(tmp_path):
    db = _db(tmp_path)
    _pool(db)
    site = _Site(chain=["secret", "sauce"])
    result = defend([("q", "strict_chain")], log=lambda _m: None, db=db, client=site, lock_path=tmp_path / "lock")
    assert result.posted == 2
    assert site.ops == ["append:gamma delta", "clear:"]


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
    opening = dry.render("DRY RUN: defending 54 board(s)")
    assert "[1/" not in opening.plain
    first = dry.render("\n== steak [strict_chain] Highest")
    assert "[1/54]" in first.plain
    assert "steak" in first.plain and "Highest" in first.plain
    counter = next(span for span in first.spans if "[1/54]" in first.plain[span.start : span.end])
    assert "cyan" in str(counter.style)

    live = DefenseLog()
    live.render("defending 2 board(s)")
    shown = live.render("\n== other [word_chain] not leading")
    assert "[1/2]" in shown.plain
    word = live.render("    + database 0.04")
    assert re.search(r"\[\d+/\d+\]", word.plain) is None
    assert "+ database" in word.plain
    again = live.render("\n== third [strict_chain] Shortest")
    assert "[2/2]" in again.plain


def test_defense_log_colors_the_turbo_banner_and_keeps_a_boards_own_mark():
    log = DefenseLog()
    banner = log.render(TURBO_BANNER)
    assert banner.plain == TURBO_BANNER
    assert any("yellow" in str(span.style) for span in banner.spans)
    log.render("defending 2 board(s)")
    shown = log.render("[2/2] == steak [strict_chain] Highest")
    assert shown.plain.startswith("[2/2] ")
    assert shown.plain.count("[2/2]") == 1
    assert log.count == 0
    mark = next(span for span in shown.spans if "[2/2]" in shown.plain[span.start : span.end])
    assert "cyan" in str(mark.style)
    done = log.render("[2/2]   done.")
    assert "[2/2]" in done.plain and "done." in done.plain


def test_home_indicator_follows_the_selected_mode(monkeypatch, tmp_path):
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
            mark = app.screen.query_one("EditionMark")
            assert mark.content.plain.startswith("Home | ")
            assert mark.content.plain.endswith(" | Strict Highest Mode")
            app.screen.query_one("#game-mode", Select).value = "shortest"
            await pilot.pause()
            assert app.screen.query_one("EditionMark").content.plain.endswith(" | Strict Shortest Mode")

    asyncio.run(check())


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
            assert str(button.label) == "Countermeasures"
            app.screen.action_defense()
            await pilot.pause()
            assert isinstance(app.screen, DefenseScreen)

    asyncio.run(check())


def test_defense_word_by_word_defaults_off_and_golf_is_marked_gone(monkeypatch, tmp_path):
    monkeypatch.setattr("jevlab.tui.defense.DB", lambda: DB(tmp_path / "empty.db"))

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(DefenseScreen())

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            words = app.screen.query_one("#by-word")
            golf = app.screen.query_one("#golf")
            turbo = app.screen.query_one("#turbo")
            assert words.value is False
            assert str(words.label) == "Word by word"
            assert golf.value is False
            assert str(golf.label) == "Include golf (no longer on website)"
            assert turbo.value is False
            assert str(turbo.label) == "Turbo"

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


class _ParallelSite:
    """Two boards. The first append waits until the second board has read its leaderboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ops: list[tuple[str, str]] = []
        self.attempts: dict[str, dict] = {}
        self.reads = 0
        self.second_read = threading.Event()
        self._append_started = False

    def viewer(self):
        return {"player": {"user": {"id": "me"}}}

    def menu(self):
        return {
            "questions": [
                {"slug": "a", "revisionId": "ra", "kind": "noul", "goal": "yes"},
                {"slug": "b", "revisionId": "rb", "kind": "noul", "goal": "yes"},
            ]
        }

    def leaderboards(self, _rev, _mode):
        with self.lock:
            self.reads += 1
            if self.reads >= 2:
                self.second_read.set()
        rows = [
            {
                "userId": "me",
                "name": "us",
                "probability": 0.99,
                "wordCount": 2,
                "phrase": "alpha beta",
            }
        ]
        return {"leaderboards": {"highScores": rows, "shortestYes": []}}

    def words(self, _rev, _mode):
        return [{"word": "alpha", "averageImpact": 0.4}, {"word": "beta", "averageImpact": 0.3}]

    def attempt(self, rev, mode):
        return self._attempt(rev, mode)

    def start(self, rev, mode):
        return self._attempt(rev, mode)

    def _attempt(self, rev, mode):
        with self.lock:
            current = self.attempts.get(rev)
            if current is None:
                current = {
                    "id": rev,
                    "nextTurn": 1,
                    "revisionId": rev,
                    "playMode": mode,
                    "tokens": [],
                    "turns": [],
                }
                self.attempts[rev] = current
            return current

    def turn(self, attempt, operation):
        if operation["kind"] == "append":
            with self.lock:
                first = not self._append_started
                self._append_started = True
            if first and not self.second_read.wait(2):
                raise AssertionError("second board did not start reading before the first append returned")
        with self.lock:
            rev = attempt["revisionId"]
            text = operation["kind"] + ":" + str(operation.get("text") or "")
            self.ops.append((rev, text))
            tokens = [token for token in attempt.get("tokens") or [] if token.get("active", True)]
            if operation["kind"] == "clear":
                tokens = []
            elif operation["kind"] == "append":
                tokens.append({"id": f"t{len(self.ops)}", "text": operation["text"], "active": True})
            updated = dict(attempt)
            updated["tokens"] = tokens
            updated["turns"] = list(attempt.get("turns") or []) + [{"id": f"n{len(self.ops)}", "probability": 0.22}]
            updated["nextTurn"] = int(attempt.get("nextTurn") or 1) + 1
            self.attempts[rev] = updated
            return updated, updated["turns"][-1]


def test_defend_turbo_posts_two_boards_at_once_and_tags_each_line(tmp_path):
    db = _db(tmp_path)
    _impacts(db, "pool", [{"word": "gamma", "averageImpact": 0.8}, {"word": "delta", "averageImpact": 0.5}])
    site = _ParallelSite()
    lines: list[str] = []
    lock = threading.Lock()

    def write(message: str) -> None:
        with lock:
            lines.append(message)

    result = defend(
        [("a", "strict_chain"), ("b", "strict_chain")],
        log=write,
        db=db,
        client=site,
        lock_path=tmp_path / "lock",
        turbo=True,
    )
    assert result.posted == 4
    assert result.boards == 2
    by_rev: dict[str, list[str]] = {}
    for rev, op in site.ops:
        by_rev.setdefault(rev, []).append(op)
    assert by_rev["ra"] == ["append:gamma delta", "clear:"]
    assert by_rev["rb"] == ["append:gamma delta", "clear:"]
    text = "\n".join(lines)
    assert "[1/2]" in text and "[2/2]" in text
    assert any("[1/2]" in line and "done." in line for line in lines)
    assert any("[2/2]" in line and "done." in line for line in lines)


def test_defense_turbo_log_starts_with_the_warning(monkeypatch, tmp_path):
    db = _db(tmp_path)
    _question(db, "q", "Ready")
    _lead(db, "q", "highScores", "alpha beta")
    _pool(db)
    monkeypatch.setattr("jevlab.tui.defense.DB", lambda: db)
    monkeypatch.setattr("jevlab.activity.log", lambda *_args, **_kwargs: None)
    seen: list[dict] = []

    def fake_defend(*_args, **kwargs):
        seen.append(kwargs)
        return DefenseRun(0, 0, 0)

    monkeypatch.setattr("jevlab.tui.defense.defend", fake_defend)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(DefenseScreen())

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.query_one("#turbo").value = True
            screen.start()
            for _ in range(20):
                await pilot.pause()
                if seen and not screen.busy:
                    break
            assert seen[0]["turbo"] is True
            text = "\n".join(line.text for line in screen.query_one("#def-log").lines)
            assert text.strip().startswith("Turbo is on")
            assert text.find("Turbo is on") < text.find("defending")

    asyncio.run(check())
