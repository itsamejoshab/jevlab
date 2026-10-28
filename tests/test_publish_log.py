"""The submit screen lays the log out as one block per phrase."""

import asyncio
import threading
import time

from textual.app import App, ComposeResult

from jevlab.config import EDITION, EDITIONS
from jevlab.db import DB
from jevlab.modes import GOLF, HIGH_SCORES, HIGHEST, STRICT
from jevlab.publish import TURBO_BOARDS, XPOST_BANNER, publish
from jevlab.tui.pickers import PublishScreen, SnapshotScreen
from jevlab.tui.publish_log import TURBO_BANNER, PublishLog


def test_every_edition_banner_has_glyphs():
    from jevlab.tui.pickers import BANNER_GLYPHS

    for name in ("JEVLAB", "KEVLAB", "LAYALAB", "CLEFLAB", "LUNALAB", "DECLAB"):
        for letter in name:
            rows = BANNER_GLYPHS[letter]
            assert len(rows) == 6
            assert len({len(row) for row in rows}) == 1


def test_home_banner_stays_jevlab_and_names_the_theatre(monkeypatch):
    from jevlab.tui import pickers

    for edition in ("jev", "kev", "laya", "clef", "clef-flash", "clef-omni", "luna", "decider", "microsoft"):
        monkeypatch.setattr(pickers, "EDITION", edition)
        assert pickers._banner_word() == "JEVLAB"
        assert pickers.edition_status() == f"{pickers._theatre()} Edition Activated"


def test_publish_log_groups_a_phrase_and_marks_a_better_roll():
    log = PublishLog()
    opening = log.render("DRY RUN: publishing 2 line(s), re-rolls 3").plain
    assert opening == "dry run  2 lines · 3 re-rolls"

    block = log.render(
        "\n== religion-most-likely-to-help-you [Golf Highest -> yang of unity] 0.960/81c: let help denote the world"
    ).plain
    assert block.splitlines() == [
        "",
        "▸ 1/2  Golf Highest  →  yang of unity",
        "  religion-most-likely-to-help-you  0.960/81c",
        "  let help denote the world",
    ]

    leader = log.render("  [golfHighest for 'yang of unity'] live leader 0.31/105c Ada; ours 0.20/80c").plain
    assert "leader  0.31/105c Ada" in leader
    assert "ours  0.20/80c" in leader
    assert "golfHighest" not in leader

    assert (
        "borrowed from jev"
        in log.render("  estimated from jev: posting the borrowed score without an oracle check").plain
    )
    assert log.render("    = 81c 0.96").plain == "  posted  81c  0.96"
    same = log.render("    re-roll 1: 0.96 (best 0.96)")
    better = log.render("    re-roll 2: 0.99 (best 0.99)")
    assert "0.96" in same.plain
    assert not any("green" in str(span.style) for span in same.spans)
    assert any("green" in str(span.style) for span in better.spans)
    won = log.render("  -> won 0.99/81c").plain
    assert won.startswith("  ✓ won 0.99/81c")
    assert won.endswith("religion-most-likely-to-help-you")

    done = log.render("\nfinished: 1/2 on top (dry run, nothing sent)").plain
    assert done.splitlines()[-1] == "1/2 on top  · nothing sent"


def test_publish_log_sets_extra_cross_posts_aside_from_the_counter():
    log = PublishLog()
    log.render("publishing 1 line(s), re-rolls 5")
    first = log.render("\n== cereal [Strict Shortest] 0.910/2w: exactly more").plain
    assert "▸ 1/1  " in first
    banner = log.render("---RUNNING ADD'L XPOST OPS---")
    assert banner.plain == "---RUNNING ADD'L XPOST OPS---"
    assert any("yellow" in str(span.style) for span in banner.spans)
    extra = log.render("\n== cereal [Golf Highest] 0.910/12c: exactly more").plain
    assert "▸ 1/1" not in extra and "▸ 2/2" not in extra
    assert "Golf Highest" in extra
    another = log.render("\n== pigeons [Golf Shortest] 0.779/5c: Books").plain
    assert "▸ 3/3" not in another
    assert "Golf Shortest" in another


def test_publish_log_keeps_lines_it_does_not_recognize():
    shown = PublishLog().render("  publisher crashed: RuntimeError('nope')").plain
    assert "publisher crashed" in shown
    short = PublishLog().render("  -> short 0.88").plain
    assert short.startswith("  ✗ short 0.88")
    missed = (
        PublishLog()
        .render("  -> missed 45 minutes plus “one more thing”; Jev picked A hostage situation with slides (0.87)")
        .plain
    )
    assert missed.startswith("  ✗ missed 45 minutes")
    assert "Jev picked" in missed


def test_publish_log_chain_and_oracle_stay_on_one_phrase():
    log = PublishLog()
    log.render("\n== cereal [Strict Highest] 0.910/2w: exactly more")
    chain = log.render("    + exactly           0.40")
    assert chain.plain == "  +exactly"
    assert any("cyan" in str(span.style) for span in chain.spans)
    assert not log.extends_chain
    more = log.render("    + more              0.55")
    assert more.plain == "  +exactly +more"
    assert log.extends_chain
    popped = log.render("    - more")
    assert popped.plain == "  +exactly +more -more"
    assert any("yellow" in str(span.style) for span in popped.spans)
    oracle = log.render("  oracle re-check 0.905 (vault said 0.910)")
    assert oracle.plain == "  oracle  0.905  vault 0.910"
    assert not log.extends_chain
    fresh = log.render("    + again             0.20")
    assert fresh.plain == "  +again"
    miss = log.render("  -> failed: oracle re-check 0.905 no longer beats empty board").plain
    assert miss.startswith("  ✗ failed:")
    assert miss.endswith("cereal")


def test_publish_log_turbo_warns_without_changing_the_opening_line():
    log = PublishLog()
    log.turbo = True
    assert str(TURBO_BOARDS) in TURBO_BANNER
    banner = log.render(TURBO_BANNER)
    assert banner.plain == TURBO_BANNER
    assert any("yellow" in str(span.style) for span in banner.spans)
    opening = log.render("publishing 2 line(s), re-rolls 3")
    assert opening.plain == "2 lines · 3 re-rolls"
    assert "Turbo" not in opening.plain
    assert not log.extends_chain


def test_publish_log_turbo_keeps_each_chain_word_and_each_questions_best():
    log = PublishLog()
    log.turbo = True
    log.render("[1/2] == cereal [Strict Highest] 0.910/2w: exactly more")
    first = log.render("[1/2] + exactly           0.40")
    assert "[1/2]" in first.plain and "+exactly" in first.plain
    assert not log.extends_chain
    other = log.render("[2/2] + other             0.20")
    assert "[2/2]" in other.plain and "+other" in other.plain
    assert "+exactly" not in other.plain
    assert any("cyan" in str(span.style) for span in other.spans)
    low = log.render("[1/2] re-roll 1: 0.40 (best 0.40)")
    assert "green" not in "".join(str(span.style) for span in low.spans)
    high = log.render("[1/2] re-roll 2: 0.80 (best 0.80)")
    assert any("green" in str(span.style) for span in high.spans)
    # Question 2 has its own best. 0.60 beats its 0.50, not question 1's 0.80.
    log.render("[2/2] re-roll 1: 0.50 (best 0.50)")
    caught_up = log.render("[2/2] re-roll 2: 0.60 (best 0.60)")
    assert any("green" in str(span.style) for span in caught_up.spans)


def test_publish_word_by_word_defaults_off_and_is_sent(monkeypatch, tmp_path):
    from textual.widgets.selection_list import Selection

    from jevlab.boards import Standing
    from jevlab.modes import STRICT

    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.activity.log", lambda *args, **kwargs: None)
    monkeypatch.setattr("jevlab.tui.pickers.vault.set_status", lambda *args, **kwargs: None)
    calls: list[dict] = []

    def fake_publish(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("jevlab.publish.publish", fake_publish)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            words = screen.query_one("#by-word")
            assert words.value is False
            assert str(words.label) == "Word by word"
            entry = {
                "slug": "cereal",
                "mode": STRICT,
                "phrase": "alpha beta",
                "p_mean": 0.9,
                "p_lcb": 0.8,
                "spread": 0.01,
                "n": 5,
                "units": 2,
                "status": "candidate",
            }
            standing = Standing("cereal", "Cereal", "noul", "yes", None, None, entry, False, True)
            screen.rows = [standing]
            screen.query_one("#pub-picker").add_option(Selection("cereal", standing.key, True))
            words.value = True
            screen.start_publish()
            for _ in range(20):
                await pilot.pause()
                if calls and not screen.busy:
                    break
            assert calls[0]["word_by_word"] is True

    asyncio.run(check())


def test_publish_screen_chain_words_overwrite_one_line(monkeypatch, tmp_path):
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.activity.log", lambda *args, **kwargs: None)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.note("== cereal [Strict Highest] 0.910/2w: exactly more")
            screen.note("    + exactly           0.40")
            screen.note("    + more              0.55")
            screen.note("    + words             0.60")
            await pilot.pause()
            log = screen.query_one("#pub-log")
            chain = [line.text.rstrip() for line in log.lines if "+exactly" in line.text]
            assert chain == ["  +exactly +more +words"]
            before = len(log.lines)
            for index in range(40):
                screen.note(f"    + w{index:<16} 0.60")
            await pilot.pause()
            assert len(log.lines) == before
            chain = [line.text.rstrip() for line in log.lines if "+w" in line.text]
            assert len(chain) == 1
            assert chain[0].startswith("…")
            assert "+w39" in chain[0]
            assert "+w0" not in chain[0]
            for index in range(30):
                screen.note(f"  filler line {index} so the log can scroll")
            await pilot.pause()
            log.scroll_home(animate=False)
            await pilot.pause()
            assert log.scroll_y == 0
            screen.note("    + still             0.70")
            await pilot.pause()
            assert log.scroll_y == 0

    asyncio.run(check())


def test_publish_screen_log_is_a_full_height_column(monkeypatch, tmp_path):
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: [])

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            screen = app.screen
            side = screen.query_one("#pub-side")
            log = screen.query_one("#pub-log")
            actions = screen.query_one("#pub-actions")
            picker = screen.query_one("#pub-picker")
            dry = screen.query_one("#dry")
            near = screen.query_one("#long")
            hint = dry.parent.query_one(".hint")
            assert log.region.x >= side.region.x + side.region.width - 1
            assert log.region.height >= 30
            assert actions.region.y < picker.region.y
            assert dry.region.y < near.region.y
            assert hint.region.x >= dry.region.x + 8
            assert "Dry run" in str(dry.label)

    asyncio.run(check())


def _prompt(option) -> str:
    prompt = option.prompt
    return prompt.plain if hasattr(prompt, "plain") else str(prompt)


def test_include_me_leading_is_off_until_asked(monkeypatch, tmp_path):
    from jevlab.boards import Standing
    from jevlab.crosspost import Pick
    from jevlab.modes import GOLF_HIGHEST
    from jevlab.objective import Leader

    mine = {
        "phrase": "already mine",
        "p_mean": 0.944,
        "p_lcb": 0.944,
        "units": 71,
        "status": "candidate",
    }
    fresh = {"phrase": "should send", "p_mean": 0.9, "p_lcb": 0.9, "units": 4, "status": "candidate"}
    rows = [
        Standing("led", "Led", "noul", "yes", None, Leader(0.74, 38, "me", True), mine, True, True),
        Standing("open", "Open", "noul", "yes", Leader(0.5, 10, "Ada"), None, fresh, False, True),
    ]
    golf = [
        Pick("led-golf", GOLF_HIGHEST, {"phrase": "golf mine", "p_mean": 0.94}, 71, None, Leader(0.74, 38), ""),
        Pick("open-golf", GOLF_HIGHEST, {"phrase": "golf open", "p_mean": 0.9}, 20, Leader(0.4, 30, "Ada"), None, ""),
    ]
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: rows)
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: golf)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            picker = screen.query_one("#pub-picker")
            leading = screen.query_one("#leading")
            assert leading.value is False
            assert "Include me leading" in str(leading.label)
            shown = [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]
            assert any("should send" in line for line in shown)
            assert any("golf open" in line for line in shown)
            assert not any("already mine" in line for line in shown)
            assert not any("golf mine" in line for line in shown)
            await pilot.click("#leading")
            await pilot.pause()
            shown = [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]
            assert any("already mine" in line for line in shown)
            assert any("golf mine" in line for line in shown)

    asyncio.run(check())


def test_golf_and_casual_rows_mark_an_imported_estimate(monkeypatch, tmp_path):
    from jevlab.crosspost import Pick
    from jevlab.modes import CASUAL_HIGHEST, GOLF_HIGHEST

    golf = [
        Pick(
            "borrowed-golf",
            GOLF_HIGHEST,
            {"phrase": "borrowed golf line", "p_mean": 0.91, "estimated_from": "jev"},
            20,
            None,
            None,
            "",
        ),
        Pick("measured-golf", GOLF_HIGHEST, {"phrase": "measured golf line", "p_mean": 0.9}, 18, None, None, ""),
    ]
    proxy = [
        Pick(
            "borrowed-casual",
            CASUAL_HIGHEST,
            {"phrase": "borrowed casual line", "p_mean": 0.88, "estimated_from": "kev"},
            4,
            None,
            None,
            "",
        )
    ]
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: golf)
    monkeypatch.setattr("jevlab.crosspost.casual_picks", lambda *args, **kwargs: proxy)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            picker = screen.query_one("#pub-picker")
            shown = [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]
            borrowed = next(line for line in shown if "borrowed golf line" in line)
            measured = next(line for line in shown if "measured golf line" in line)
            assert borrowed.startswith("Golf Highest")
            assert not borrowed.startswith("GOLF ")
            assert "[est(jev)]" in borrowed
            assert "[est(" not in measured
            await pilot.click("#casual")
            await pilot.pause()
            shown = [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]
            casual = next(line for line in shown if "borrowed casual line" in line)
            assert casual.startswith("Casual Highest")
            assert not casual.startswith("CASUAL ")
            assert "[est(kev)]" in casual

    asyncio.run(check())


def test_casual_proxy_stays_hidden_until_asked(monkeypatch, tmp_path):
    from jevlab.crosspost import Pick
    from jevlab.modes import CASUAL_HIGHEST
    from jevlab.objective import Leader

    proxy = [
        Pick(
            "empty-board",
            CASUAL_HIGHEST,
            {"phrase": "strict highest answer", "p_mean": 0.91, "p_lcb": 0.9},
            6,
            None,
            None,
            "",
        ),
        Pick(
            "already-casual",
            CASUAL_HIGHEST,
            {"phrase": "ours already", "p_mean": 0.88, "p_lcb": 0.88},
            4,
            None,
            Leader(0.8, 5, "me", True),
            "",
        ),
    ]
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.casual_picks", lambda *args, **kwargs: proxy)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            casual = screen.query_one("#casual")
            assert casual.value is False
            assert "Include casual proxy" in str(casual.label)
            picker = screen.query_one("#pub-picker")
            assert picker.option_count == 0
            await pilot.click("#casual")
            await pilot.pause()
            shown = [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]
            assert any("strict highest answer" in line and "empty" in line for line in shown)
            assert not any("ours already" in line for line in shown)
            await pilot.click("#leading")
            await pilot.pause()
            shown = [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]
            assert any("ours already" in line for line in shown)

    asyncio.run(check())


def test_snapshot_screen_toggles_are_one_line() -> None:
    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(SnapshotScreen())

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            start = screen.query_one("#start")
            current = screen.query_one(f"#snap-ed-{EDITION}")
            other_name = next(edition for edition in EDITIONS if edition != EDITION)
            other = screen.query_one(f"#snap-ed-{other_name}")
            strict = screen.query_one("#snap-mode-strict_chain")
            casual = screen.query_one("#snap-mode-word_chain")
            assert start.region.height <= 3
            assert current.region.height == 1
            assert strict.region.height == 1
            assert casual.region.height == 1
            assert casual.value is True
            assert current.region.y == other.region.y
            assert casual.region.y == strict.region.y
            assert current.value is True
            assert other.value is True
            assert not other.disabled
            assert strict.region.y > current.region.y
            assert start.region.y < current.region.y
            mark = screen.query_one("EditionMark")
            from jevlab.tui.chrome import NAMES

            assert mark.content.plain == f"Download | {NAMES.get(EDITION, 'Jev')} Edition"

    asyncio.run(check())


def test_edition_caption_names_the_screen_edition_and_mode(monkeypatch):
    from jevlab.tui.chrome import edition_caption

    monkeypatch.setattr("jevlab.config.EDITION", "laya")
    assert (
        edition_caption("Countermeasures", "Strict Highest").plain
        == "Countermeasures | Laya Edition | Strict Highest Mode"
    )
    assert edition_caption("Download").plain == "Download | Laya Edition"
    monkeypatch.setattr("jevlab.config.EDITION", "jev")
    assert edition_caption("Scan", "Live Mode").plain == "Scan | Jev Edition | Live Mode"


def test_snapshot_switches_each_edition_then_returns(monkeypatch):
    from jevlab.tui import pickers

    monkeypatch.setattr(pickers, "EDITION", "jev")
    seen = []

    def activate(edition):
        pickers.EDITION = edition
        seen.append(("on", edition))

    def take(db, modes, log, section=""):
        seen.append(("snap", pickers.EDITION, db, modes, section))

    monkeypatch.setattr(pickers, "DB", lambda: "db")
    monkeypatch.setattr("jevlab.config.activate", activate)
    monkeypatch.setattr("jevlab.site.snapshot.take_snapshot", take)

    pickers.run_snapshots(("kev", "laya"), ("strict_chain",), lambda message: seen.append(("log", message)))

    assert [item for item in seen if item[0] != "log"] == [
        ("on", "kev"),
        ("snap", "kev", "db", ("strict_chain",), "1/2"),
        ("on", "laya"),
        ("snap", "laya", "db", ("strict_chain",), "2/2"),
        ("on", "jev"),
    ]
    rules = [item[1] for item in seen if item[0] == "log"]
    assert rules[0] == "SYNC 2 editions · Strict"
    assert "▸ 1/2 · Kev" in rules
    assert "▸ 2/2 · Laya" in rules
    assert rules[-1] == "↩ back to Jev"


def test_snapshot_log_marks_a_failed_edition_and_continues(monkeypatch):
    from jevlab.tui import pickers

    monkeypatch.setattr(pickers, "EDITION", "jev")
    seen = []

    def activate(edition):
        pickers.EDITION = edition

    def take(db, modes, log, section=""):
        if pickers.EDITION == "kev":
            raise RuntimeError("menu down")
        log(f"body {section}")

    monkeypatch.setattr(pickers, "DB", lambda: "db")
    monkeypatch.setattr("jevlab.config.activate", activate)
    monkeypatch.setattr("jevlab.site.snapshot.take_snapshot", take)
    pickers.run_snapshots(("kev", "laya"), ("golf", "word_chain"), lambda message: seen.append(message))

    assert seen[0] == "SYNC 2 editions · Golf, Casual"
    assert "✗ 1/2 · Kev · failed" in seen
    assert any("menu down" in message for message in seen)
    assert "body 2/2" in seen
    assert seen[-1] == "↩ back to Jev"


def test_snapshot_log_colors_section_rules():
    from jevlab.site.snapshot import log_rule, progress_line
    from jevlab.tui.pickers import snapshot_line

    def styles(message: str) -> str:
        return " ".join(str(span.style) for span in snapshot_line(message).spans)

    assert "bold cyan" in styles(log_rule("1/3 · Jev"))
    assert "bold green" in styles(log_rule("1/3 · Jev · done · 4s"))
    assert "bold red" in styles(log_rule("1/3 · Jev · failed"))
    assert "bold yellow" in styles(log_rule("stopped · 1/3 finished"))
    running = styles(progress_line("jev", 40, 120, 12, section="1/3"))
    assert "dim" in running
    assert "bold cyan" in running
    assert "bold green" in styles(progress_line("jev", 8, 8, 4, section="1/3"))
    assert "bold yellow" in styles(progress_line("jev", 8, 8, 4, failed=2))
    opened = snapshot_line(log_rule("snapshot · 9 editions · Strict, Golf"))
    assert "SYNC" in opened.plain
    assert "Strict" in opened.plain
    assert "Golf" in opened.plain


def _question(db: DB, slug: str, revision: str) -> None:
    db.execute(
        "INSERT INTO questions VALUES (?, 0, ?, 'Is it rude?', '', 'noul', 'yes', 0.2, 0.5, 'm', '{}', '{}')",
        (slug, revision),
    )


def _entry(slug: str, phrase: str) -> dict:
    return {
        "slug": slug,
        "mode": STRICT,
        "phrase": phrase,
        "p_mean": 0.99,
        "p_lcb": 0.9,
        "units": len(phrase.split()),
        "board": HIGH_SCORES,
        "target": "",
        "status": "queued",
    }


class _DrySite:
    def __init__(self, questions: list[dict]):
        self.questions = questions
        self.lock = threading.Lock()
        self.entered: list[str] = []
        self.second = threading.Event()
        self.overlapped = False
        self.depth = 0

    def viewer(self):
        return {"player": {"user": {"id": "me"}}}

    def menu(self):
        return {"questions": self.questions}

    def save_session(self):
        return None

    def leaderboards(self, revision_id: str, _mode: str) -> dict:
        with self.lock:
            self.depth += 1
            self.entered.append(revision_id)
            if self.depth > 1:
                self.overlapped = True
            n = len(self.entered)
        if n == 1:
            self.overlapped = self.second.wait(1)
        else:
            self.second.set()
        try:
            time.sleep(0.05)
            return {"leaderboards": {}}
        finally:
            with self.lock:
                self.depth -= 1


def test_publish_turbo_overlaps_questions_and_keeps_one_slug_in_order(tmp_path):
    db = DB(tmp_path / "t.db")
    _question(db, "alpha", "ra")
    _question(db, "beta", "rb")
    site = _DrySite(
        [
            {"slug": "alpha", "revisionId": "ra", "kind": "noul", "goal": "yes"},
            {"slug": "beta", "revisionId": "rb", "kind": "noul", "goal": "yes"},
        ]
    )
    lines: list[str] = []
    lock = threading.Lock()

    def write(message: str) -> None:
        with lock:
            lines.append(message)

    code = publish(
        dry_run=True,
        rerolls=0,
        entries=[_entry("alpha", "alpha beta"), _entry("beta", "gamma delta")],
        log=write,
        crosspost=False,
        turbo=True,
        db=db,
        client=site,
        lock_path=tmp_path / "lock",
    )
    assert code == 0
    assert site.overlapped
    text = "\n".join(lines)
    assert "[1/2]" in text and "[2/2]" in text
    assert "alpha beta" in text and "gamma delta" in text

    same = _DrySite([{"slug": "alpha", "revisionId": "ra", "kind": "noul", "goal": "yes"}])
    ordered: list[str] = []

    def write_one(message: str) -> None:
        ordered.append(message)

    publish(
        dry_run=True,
        rerolls=0,
        entries=[_entry("alpha", "alpha beta"), _entry("alpha", "gamma delta")],
        log=write_one,
        crosspost=False,
        turbo=True,
        db=db,
        client=same,
        lock_path=tmp_path / "lock-one",
    )
    joined = "\n".join(ordered)
    assert joined.find("alpha beta") < joined.find("gamma delta")
    assert not same.overlapped
    assert "[1/1]" in joined


def test_publish_runs_extra_cross_posts_after_the_numbered_batch(tmp_path, monkeypatch):
    db = DB(tmp_path / "t.db")
    site = _DrySite([{"slug": "alpha", "revisionId": "ra", "kind": "noul", "goal": "yes"}])
    monkeypatch.setattr("jevlab.publish.publish_entry", lambda *args, **kwargs: "won")
    monkeypatch.setattr("jevlab.publish.sweep_banned", lambda *args, **kwargs: [])

    def followups(_db, slug, log):
        log("  cross-posting 1 line(s) to Golf")
        return [_entry(slug, "golf line") | {"mode": GOLF, "units": 9}]

    monkeypatch.setattr("jevlab.publish.golf_followups", followups)
    lines: list[str] = []
    publish(
        rerolls=0,
        entries=[_entry("alpha", "alpha beta")],
        log=lines.append,
        turbo=True,
        db=db,
        client=site,
        lock_path=tmp_path / "lock",
    )
    text = "\n".join(lines)
    banner = text.find(XPOST_BANNER)
    assert banner > text.find("[1/1]")
    assert banner < text.find("golf line")
    golf = next(line for line in lines if "golf line" in line)
    assert "[1/1]" not in golf and "[2/" not in golf


def test_publish_turbo_defaults_off_and_the_log_starts_with_the_warning(monkeypatch, tmp_path):
    from textual.widgets.selection_list import Selection

    from jevlab.boards import Standing

    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.crosspost.picks", lambda *args, **kwargs: [])
    monkeypatch.setattr("jevlab.activity.log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("jevlab.tui.pickers.vault.set_status", lambda *args, **kwargs: None)
    pauses: list[float] = []
    shown: list[list[str]] = []
    screen_ref: list = []
    real_sleep = time.sleep

    def ticks(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip() in {"5", "4", "3", "2", "1"}]

    def log_text() -> str:
        return "\n".join(line.text for line in screen_ref[0].query_one("#pub-log").lines)

    def fake_sleep(seconds: float) -> None:
        if seconds == 1:
            pauses.append(seconds)
            if screen_ref:
                shown.append(ticks(log_text()))
            return
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    calls: list[dict] = []
    started_after: list[int] = []

    def fake_publish(**kwargs):
        started_after.append(len(pauses))
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("jevlab.publish.publish", fake_publish)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(PublishScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen_ref.append(screen)
            turbo = screen.query_one("#turbo")
            assert turbo.value is False
            assert str(turbo.label) == "Turbo"
            entry = {
                "slug": "cereal",
                "mode": STRICT,
                "phrase": "alpha beta",
                "p_mean": 0.9,
                "p_lcb": 0.8,
                "spread": 0.01,
                "n": 5,
                "units": 2,
                "status": "candidate",
            }
            standing = Standing("cereal", "Cereal", "noul", "yes", None, None, entry, False, True)

            def arm() -> None:
                screen.rows = [standing]
                picker = screen.query_one("#pub-picker")
                picker.clear_options()
                picker.add_option(Selection("cereal", standing.key, True))

            async def settle() -> None:
                for _ in range(20):
                    await pilot.pause()
                    if calls and not screen.busy:
                        break

            arm()
            screen.start_publish()
            await settle()
            assert calls[0]["turbo"] is False
            assert pauses == []
            assert ticks(log_text()) == []
            assert "Turbo is on" not in log_text()

            pauses.clear()
            shown.clear()
            calls.clear()
            started_after.clear()
            arm()
            turbo.value = True
            screen.start_publish()
            await settle()
            assert calls[0]["turbo"] is True
            assert started_after == [5]
            assert pauses == [1, 1, 1, 1, 1]
            assert shown == [
                ["5"],
                ["5", "4"],
                ["5", "4", "3"],
                ["5", "4", "3", "2"],
                ["5", "4", "3", "2", "1"],
            ]
            text = log_text()
            warning = text.rfind("Turbo is on")
            started = text.find("re-rolls", warning)
            assert warning != -1 and started > warning
            assert ticks(text[warning:started]) == ["5", "4", "3", "2", "1"]

    asyncio.run(check())
