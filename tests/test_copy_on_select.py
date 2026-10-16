"""Drag-selecting text copies it, including the search screen's log and table."""

import asyncio

from textual.app import App, ComposeResult
from textual.events import TextSelected
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Static

from jevlab.tui.app import JevlabApp
from jevlab.tui.copyselect import SelectableDataTable, SelectableRichLog, extract_selection


class _Probe(App):
    CSS = """
    Screen { layout: vertical; }
    #line { height: 1; }
    #log { height: 6; border: round $accent; }
    #table { height: 8; border: round $primary; }
    """

    def on_text_selected(self, event: TextSelected) -> None:
        JevlabApp.on_text_selected(self, event)

    def compose(self) -> ComposeResult:
        yield Static("hello world", id="line")
        yield SelectableRichLog(id="log", markup=False, wrap=False, min_width=10)
        yield SelectableDataTable(id="table", cursor_type="row", show_row_labels=False)

    def on_mount(self) -> None:
        log = self.query_one("#log", SelectableRichLog)
        log.write("alpha beta")
        log.write("gamma delta")
        table = self.query_one("#table", SelectableDataTable)
        table.add_column("phrase")
        table.cell_padding = 0
        table.add_row("one")
        table.add_row("alphabet soup")


def _run(check) -> None:
    async def body() -> None:
        app = _Probe()
        async with app.run_test(size=(80, 24)) as pilot:
            await check(pilot, app)

    asyncio.run(body())


def test_extract_selection_spans_lines() -> None:
    selection = Selection(Offset(1, 0), Offset(4, 1))
    text = extract_selection(selection, lambda y: ("abcdef", "ghijkl")[y], 2)
    assert text == "bcdef\nghij"


def test_static_drag_copies_to_clipboard() -> None:
    async def check(pilot, app) -> None:
        await pilot.mouse_down("#line", offset=(0, 0))
        await pilot.mouse_up("#line", offset=(4, 0))
        assert app.clipboard == "hello"

    _run(check)


def test_click_does_not_copy_or_clear_clipboard() -> None:
    async def check(pilot, app) -> None:
        await pilot.mouse_down("#line", offset=(0, 0))
        await pilot.mouse_up("#line", offset=(4, 0))
        await pilot.click("#line", offset=(1, 0))
        assert app.clipboard == "hello"

    _run(check)


def test_log_drag_copies_across_lines() -> None:
    async def check(pilot, app) -> None:
        # Border occupies the first row and column; the text starts inside it.
        await pilot.mouse_down("#log", offset=(1, 1))
        await pilot.mouse_up("#log", offset=(5, 2))
        assert app.clipboard == "alpha beta\ngamma"

    _run(check)


def test_log_selection_shows_on_top_of_a_colored_line() -> None:
    from rich.text import Text

    async def check(pilot, app) -> None:
        log = app.query_one("#log", SelectableRichLog)
        log.clear()
        log.write(Text("alpha beta", style="bold green"))
        await pilot.pause()
        app.screen.selections = {log: Selection(Offset(0, 0), Offset(5, 0))}
        await pilot.pause()
        wanted = app.screen.get_component_rich_style("screen--selection").bgcolor
        strip = log.render_line(0)
        selected = next(segment for segment in strip if segment.text.startswith("alpha"))
        assert selected.style is not None and selected.style.bgcolor == wanted
        # The offset that hit-testing reads has to survive the highlight.
        assert selected.style.meta["offset"] == (0, 0)

    _run(check)


def test_table_drag_copies_cell_text_and_click_moves_the_row() -> None:
    async def check(pilot, app) -> None:
        table = app.query_one("#table", SelectableDataTable)
        await pilot.mouse_down("#table", offset=(1, 3))
        await pilot.mouse_up("#table", offset=(8, 3))
        assert app.clipboard == "alphabet"
        await pilot.click("#table", offset=(1, 2))
        assert table.cursor_coordinate.row == 0

    _run(check)
