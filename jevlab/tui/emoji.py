"""Question picker for the emoji sidecar. It never starts the Strict engine."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, SelectionList, Select, Static, Switch
from textual.widgets.selection_list import Selection

from ..boards import Standing, standings
from ..db import DB
from ..modes import GameMode
from .chrome import EditionMark
from .copyselect import SelectableRichLog
from .pickers import RunKeys, Setting, fmt_leader, fmt_ours


class EmojiSetupScreen(RunKeys, Screen):
    CSS = """
    #emoji-help { height: auto; padding: 0 1; }
    #emoji-body { height: 1fr; }
    #picker { height: 1fr; width: 1fr; border: round $primary; }
    #emoji-log {
        height: 1fr; width: 1fr; min-width: 36;
        border: round $secondary; margin-left: 1;
    }
    #emoji-actions { height: 3; }
    #emoji-actions Label { padding: 1 1 0 0; text-style: bold; color: $text-muted; }
    #emoji-actions Button { margin-right: 1; min-width: 8; }
    #no-choice { height: 1; margin: 1 1 0 0; }
    #target-player { width: 32; }
    #emoji-run { height: 3; }
    #budget { width: 16; min-width: 16; }
    #patience { width: 8; min-width: 8; }
    #win-mode { margin: 0 1; }
    #emoji-run Button { margin-left: 1; min-width: 8; }
    """

    BINDINGS = [
        Binding("space", "toggle_pause", "pause/resume", priority=True),
        Binding("s", "stop", "stop", priority=True),
        Binding("escape", "back", "Back"),
        Binding("g", "go", "Go"),
        Binding("a", "select_all", "all"),
        Binding("x", "select_none", "none"),
        Binding("l", "select_not_leading", "not leading"),
        Binding("e", "select_empty", "empty"),
        Binding("t", "target_player", "target player"),
    ]

    pause_message = "paused after this call"
    leave_message = "stopping after this call"

    def __init__(self, game_mode: GameMode):
        super().__init__()
        self.game_mode = game_mode
        self.exclude_choice = False
        self.runs: dict[str, dict] = {}
        self.rows = sorted(
            standings(DB(), mode=game_mode.play_mode, board=game_mode.board),
            key=lambda row: (not row.searchable, row.search_order),
        )

    def compose(self) -> ComposeResult:
        yield EditionMark("Emoji", self.game_mode.label)
        yield Static(id="emoji-help")
        with Horizontal(id="emoji-body"):
            yield SelectionList[str](*self.picker_options(), id="picker")
            yield SelectableRichLog(id="emoji-log", wrap=True, markup=False, max_lines=2000)
        with Horizontal(id="emoji-actions"):
            yield Label("select")
            yield Button("All", id="all")
            yield Button("None", id="none")
            yield Button("Not leading", id="not-leading")
            yield Button("Empty", id="empty")
            yield Setting("Exclude choice", False, id="no-choice", compact=True)
            yield Select(self.player_options(), prompt="Target player", id="target-player")
        with Horizontal(id="emoji-run"):
            yield Label("Budget")
            yield Input("40", id="budget", type="integer", max_length=6)
            yield Label("Patience")
            yield Input("6", id="patience", type="integer", max_length=3)
            yield Label("Win mode")
            yield Switch(True, id="win-mode")
            yield Button("GO", id="go", variant="primary")
            yield Button("Stop", id="stop", disabled=True)
            yield Button("Back", id="back")
        yield Footer()

    def visible_rows(self) -> list[Standing]:
        if self.exclude_choice:
            return [row for row in self.rows if row.kind != "choice"]
        return list(self.rows)

    def targetable(self, row: Standing) -> bool:
        return row.searchable and not row.we_lead and not row.unwinnable and row.leader is not None

    def player_options(self) -> list[tuple[str, str]]:
        """Players leading at least one board we could take, most boards first."""
        counts: dict[str, int] = {}
        for row in self.visible_rows():
            if self.targetable(row) and row.leader.name:
                counts[row.leader.name] = counts.get(row.leader.name, 0) + 1
        return [(f"{name} ({count})", name) for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def picker_options(self, selected: set[str] | None = None) -> list[Selection]:
        out = []
        for row in self.visible_rows():
            initial = row.should_search if selected is None else row.key in selected and row.searchable
            out.append(Selection(self.row_prompt(row), row.key, initial, id=row.key, disabled=not row.searchable))
        return out

    def row_prompt(self, row: Standing) -> Text:
        """Leader line, plus the best emoji from a search and WIN when its lower bound beats the leader."""
        prompt = Text()
        run = self.runs.get(row.key) or {}
        state = run.get("state")
        icon, icon_style = {
            "running": ("▶ ", "bold yellow"),
            "done": ("✓ ", "green"),
            "stopped": ("- ", "dim"),
            "skipped": ("- ", "dim"),
            "error": ("! ", "red"),
        }.get(state, ("", ""))
        if icon:
            prompt.append(icon, style=icon_style)
        if run.get("best"):
            prompt.append(f"{run['best']:<10}", style="bold green" if run.get("win") else "bold")
        if run.get("win"):
            prompt.append("WIN ", style="bold green")
        if run.get("phrase"):
            prompt.append(f"{run['phrase']} ", style="bold green" if run.get("win") else "cyan")
        if row.searchable and row.leader is None:
            prompt.append("EMPTY           ", style="bold cyan")
        else:
            prompt.append(f"lead {fmt_leader(row)} ", style="yellow")
            prompt.append(f"{(row.leader.name if row.leader else '')[:14]:<14}  ", style="dim yellow")
        prompt.append(f"ours {fmt_ours(row)}  ", style="green" if row.we_lead else "")
        prompt.append("LEAD " if row.we_lead else "     ", style="bold green")
        prompt.append(row.label[:80], style="" if row.searchable and not row.unwinnable else "dim strike")
        if not row.searchable:
            prompt.append(f"  ({row.kind}: not searchable yet)", style="dim")
        elif row.unwinnable:
            prompt.append("  (unwinnable: leader has 1.00 in 1 word, we could only tie)", style="dim")
        return prompt

    def apply_progress(self, info: dict) -> None:
        """Update one question's best and WIN tag without resetting the checks."""
        from textual.content import Content
        from textual.widgets.option_list import OptionDoesNotExist

        slug = info["slug"]
        current = dict(self.runs.get(slug) or {})
        current.update(info)
        self.runs[slug] = current
        picker = self.query_one("#picker", SelectionList)
        try:
            option = picker.get_option(slug)
        except OptionDoesNotExist:
            return
        row = next(row for row in self.rows if row.key == slug)
        option._set_prompt(Content.from_text(self.row_prompt(row)).split()[0])
        # The list caches each row's pixels. Without this, a WIN shows only while the row is
        # highlighted, then the old line comes back when the search moves on.
        picker._clear_caches()
        if info.get("state") == "running":
            picker.highlighted = picker.get_option_index(slug)

    def refresh_picker(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        selected = set(picker.selected)
        picker.clear_options()
        picker.add_options(self.picker_options(selected))
        self.query_one("#target-player", Select).set_options(self.player_options())

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "stop":
            return True if self.busy else None
        return super().check_action(action, parameters)

    def on_mount(self) -> None:
        order = "longest leader first" if self.game_mode.shortest else "weakest leader first"
        self.query_one("#picker").border_title = f"{self.game_mode.label}: questions, {order}"
        self.query_one("#picker").border_subtitle = "best = best emoji so far    WIN = lower bound beats the leader"
        self.query_one("#emoji-log").border_title = "emoji"
        self.query_one("#emoji-help", Static).update(
            "All, None, Not leading, Empty, and Target player choose the questions. "
            "Win mode, when on, saves a WIN for publish and moves to the next question. "
            "Patience counts climb passes that do not raise the score. The question continues until the budget is spent, or a one-glyph line rounds to 1.00. "
            "Go opens the lab: project, singles, climb, then after patience an order pass and a beam, then compact. "
            "Space pauses and s stops, after the current oracle call."
        )

    def on_checkbox_changed(self, event) -> None:
        if event.checkbox.id != "no-choice":
            return
        self.exclude_choice = bool(event.value)
        self.refresh_picker()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "target-player" or not isinstance(event.value, str):
            return
        picker = self.query_one("#picker", SelectionList)
        picker.deselect_all()
        chosen = [row for row in self.visible_rows() if self.targetable(row) and row.leader.name == event.value]
        for row in chosen:
            picker.select(row.key)
        self.notify(
            f"{len(chosen)} board{'' if len(chosen) == 1 else 's'} where {event.value} leads {self.game_mode.label}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        {
            "all": self.action_select_all,
            "none": self.action_select_none,
            "not-leading": self.action_select_not_leading,
            "empty": self.action_select_empty,
            "go": self.action_go,
            "stop": self.action_stop,
            "back": self.action_back,
        }[event.button.id]()

    def action_select_all(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        for row in self.visible_rows():
            if row.searchable:
                picker.select(row.key)

    def action_select_none(self) -> None:
        self.query_one("#picker", SelectionList).deselect_all()

    def action_select_not_leading(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        picker.deselect_all()
        for row in self.visible_rows():
            if row.should_search:
                picker.select(row.key)

    def action_select_empty(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        picker.deselect_all()
        for row in self.visible_rows():
            if row.searchable and row.leader is None and not row.unwinnable:
                picker.select(row.key)

    def action_target_player(self) -> None:
        self.query_one("#target-player", Select).focus()

    def action_back(self) -> None:
        if self.busy:
            self.notify("emoji search is still running", severity="warning")
            return
        self.app.pop_screen()

    def action_stop(self) -> None:
        if not self.busy:
            return
        self.stop_requested = True
        self.paused = False
        self.notify("stopping after this call")

    def action_go(self) -> None:
        if self.busy:
            return
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
        try:
            patience = int(self.query_one("#patience", Input).value)
        except ValueError:
            patience = 0
        if patience < 1:
            self.notify("patience must be a whole number of at least 1", severity="error")
            return
        win_mode = self.query_one("#win-mode", Switch).value
        order = [row.key for row in self.rows if row.key in chosen and row.searchable]
        from .emoji_lab import EmojiLabScreen

        self.app.push_screen(EmojiLabScreen(order, budget, win_mode, self.game_mode, patience))
