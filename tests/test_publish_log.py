"""The submit screen lays the log out as one block per phrase."""

import asyncio

from textual.app import App, ComposeResult

from jevlab.config import EDITION, EDITIONS
from jevlab.db import DB
from jevlab.modes import HIGHEST
from jevlab.tui.pickers import PublishScreen, SnapshotScreen
from jevlab.tui.publish_log import PublishLog


def test_every_edition_banner_has_glyphs():
    from jevlab.tui.pickers import BANNER_GLYPHS

    for name in ("JEVLAB", "KEVLAB", "LAYALAB", "CLEFLAB", "LUNALAB", "DECLAB"):
        for letter in name:
            rows = BANNER_GLYPHS[letter]
            assert len(rows) == 6
            assert len({len(row) for row in rows}) == 1


def test_home_banner_stays_jevlab_and_names_the_theatre(monkeypatch):
    from jevlab.tui import pickers

    for edition in ("jev", "kev", "laya", "clef", "luna", "decider"):
        monkeypatch.setattr(pickers, "EDITION", edition)
        assert pickers._banner_word() == "JEVLAB"
        assert pickers.edition_status() == f"{pickers._theatre()} Mode Activated"


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
            assert "mode" in str(mark.content)

    asyncio.run(check())


def test_edition_caption_follows_the_open_edition(monkeypatch):
    from jevlab.tui.chrome import edition_caption

    monkeypatch.setattr("jevlab.config.EDITION", "luna")
    assert edition_caption().plain == "Luna mode"
    monkeypatch.setattr("jevlab.config.EDITION", "jev")
    assert edition_caption().plain == "Jev mode"


def test_snapshot_switches_each_edition_then_returns(monkeypatch):
    from jevlab.tui import pickers

    monkeypatch.setattr(pickers, "EDITION", "jev")
    seen = []

    def activate(edition):
        pickers.EDITION = edition
        seen.append(("on", edition))

    def take(db, modes, log):
        seen.append(("snap", pickers.EDITION, db, modes))

    monkeypatch.setattr(pickers, "DB", lambda: "db")
    monkeypatch.setattr("jevlab.config.activate", activate)
    monkeypatch.setattr("jevlab.site.snapshot.take_snapshot", take)

    pickers.run_snapshots(("kev", "laya"), ("strict_chain",), lambda message: seen.append(("log", message)))

    assert [item for item in seen if item[0] != "log"] == [
        ("on", "kev"),
        ("snap", "kev", "db", ("strict_chain",)),
        ("on", "laya"),
        ("snap", "laya", "db", ("strict_chain",)),
        ("on", "jev"),
    ]
