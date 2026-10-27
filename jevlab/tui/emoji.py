"""Question picker for the emoji sidecar. It never starts the Strict engine."""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, SelectionList, Static
from textual.widgets.selection_list import Selection

from ..boards import standings
from ..db import DB
from ..emoji.search import search
from ..modes import GameMode
from .chrome import EditionMark
from .copyselect import SelectableRichLog


class EmojiSetupScreen(Screen):
    CSS = """
    #emoji-help { height: auto; padding: 0 1; }
    #picker { height: 1fr; border: round $primary; }
    #emoji-run { height: 3; }
    #budget { width: 8; }
    #emoji-log { height: 8; border: round $secondary; display: none; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("g", "go", "Go"),
    ]

    def __init__(self, game_mode: GameMode):
        super().__init__()
        self.game_mode = game_mode
        self.rows = []

    def compose(self) -> ComposeResult:
        yield EditionMark("Emoji", self.game_mode.label)
        yield Static(id="emoji-help")
        yield SelectionList[str](id="picker")
        with Horizontal(id="emoji-run"):
            yield Label("Budget")
            yield Input(value="40", id="budget")
            yield Button("GO", id="go", variant="primary")
            yield Button("Back", id="back")
        yield SelectableRichLog(id="emoji-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#picker").border_title = f"{self.game_mode.label}: questions"
        self.rows = [
            row for row in standings(DB(), mode=self.game_mode.play_mode, board=self.game_mode.board) if row.searchable
        ]
        self.rows.sort(key=lambda row: row.search_order)
        picker = self.query_one("#picker", SelectionList)
        for row in self.rows:
            lead = f"{row.leader.probability:.2f}/{row.leader.units}w" if row.leader else "empty"
            prompt = Text()
            prompt.append(f"{lead:<10} ", style="yellow")
            prompt.append(row.label[:80], style="" if row.should_search else "dim")
            picker.add_option(Selection(prompt, row.key, row.should_search and not row.unwinnable))
        self.query_one("#emoji-help", Static).update(
            "Name overlap picks the shortlist. The budget is the most oracle calls for one question."
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self.action_go()
        elif event.button.id == "back":
            self.action_back()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_go(self) -> None:
        chosen = set(self.query_one("#picker", SelectionList).selected)
        if not chosen:
            self.notify("select at least one question", severity="warning")
            return
        try:
            budget = int(self.query_one("#budget", Input).value)
        except ValueError:
            budget = 0
        if budget < 1:
            self.notify("budget must be a whole number of at least 1", severity="error")
            return
        order = [row.key for row in self.rows if row.key in chosen]
        self.query_one("#go", Button).disabled = True
        self._run(order, budget)

    @work(thread=True, exclusive=True)
    def _run(self, order: list[str], budget: int) -> None:
        log = self.query_one("#emoji-log", SelectableRichLog)

        def begin() -> None:
            log.display = True
            log.clear()

        def write(message: str) -> None:
            self.app.call_from_thread(log.write, message)

        self.app.call_from_thread(begin)
        db = DB()
        for slug in order:
            write(f"\n== {slug}")
            try:
                asyncio.run(search(db, slug, board=self.game_mode.board, budget=budget, log=write))
            except Exception as error:
                write(f"{slug}: {error}")

        def finish() -> None:
            self.query_one("#go", Button).disabled = False
            self.notify(f"emoji search finished, {len(order)} question(s)")

        self.app.call_from_thread(finish)
