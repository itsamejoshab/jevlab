"""Defense: pick boards we lead and post decoy words onto their public word lists."""

from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Input, Label, SelectionList, Static
from textual.widgets.selection_list import Selection

from .. import activity
from ..config import DEFENSE_TOP
from ..db import DB
from ..defense import BARE, BLIND, QUIET, READY, DefenseTarget, defend, scan
from ..modes import CASUAL
from .chrome import EditionMark
from .copyselect import SelectableRichLog
from .pickers import Setting


def _mode_name(play_mode: str) -> str:
    return "Casual" if play_mode == CASUAL else "Strict"


class DefenseScreen(Screen):
    CSS = """
    #def-body { height: 1fr; }
    #def-side { width: 3fr; height: 1fr; }
    #def-actions { height: 3; margin-bottom: 1; }
    #def-actions Button { margin-right: 1; min-width: 10; }
    #defend { min-width: 20; }
    #def-options {
        height: auto; border: round $primary-darken-2; padding: 0 1; margin-bottom: 1;
    }
    .opt { height: 1; align: left middle; }
    .opt Setting { min-width: 22; }
    .opt Setting:focus, .opt Setting:hover { background: transparent; }
    .opt Label { width: 16; height: 1; text-style: bold; }
    .hint { width: 1fr; height: 1; margin-left: 1; color: $text-muted; }
    #top-n { width: 6; height: 1; margin: 0; background: $boost; }
    #top-n:focus { border: none; }
    #def-help { height: auto; padding: 0 1 1 1; }
    #def-picker { height: 1fr; border: round $primary; }
    #def-log { width: 2fr; min-width: 42; max-width: 72; height: 1fr; border: round $secondary; margin-left: 1; }
    """

    BINDINGS = [Binding("escape", "back", "back")]

    def __init__(self):
        super().__init__()
        self.targets: list[DefenseTarget] = []
        self.top_n = DEFENSE_TOP
        self.busy = False
        self.stop_requested = False

    def compose(self) -> ComposeResult:
        yield EditionMark()
        with Horizontal(id="def-body"):
            with Vertical(id="def-side"):
                with Horizontal(id="def-actions"):
                    yield Button("Refresh", id="refresh")
                    yield Button("Defend selected", id="defend", variant="warning")
                    yield Button("Stop", id="stop", disabled=True)
                    yield Button("Back", id="back")
                with Vertical(id="def-options"):
                    with Horizontal(classes="opt"):
                        yield Setting("Dry run", False, id="dry", compact=True)
                        yield Static("nothing is sent", classes="hint")
                    with Horizontal(classes="opt"):
                        yield Setting("Include casual", False, id="casual", compact=True)
                        yield Static("casual keeps its own word list", classes="hint")
                    with Horizontal(classes="opt"):
                        yield Label("Top words")
                        yield Input(str(DEFENSE_TOP), id="top-n", type="integer", compact=True)
                        yield Static("per board, from other boards", classes="hint")
                yield Static(id="def-help")
                yield SelectionList[str](id="def-picker")
            yield SelectableRichLog(id="def-log", wrap=True, markup=False, max_lines=2000)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#def-picker").border_title = "boards we lead"
        self.query_one("#def-log").border_title = "defense"
        self.reload()

    def _read_top(self) -> int | None:
        field = self.query_one("#top-n", Input)
        try:
            value = int(field.value)
        except ValueError:
            value = 0
        if value < 1:
            self.notify("Top words must be at least 1", severity="error")
            field.focus()
            return None
        return value

    def reload(self) -> None:
        top_n = self._read_top()
        if top_n is None:
            return
        self.top_n = top_n
        include_casual = self.query_one("#casual", Checkbox).value
        db = DB()
        snap = db.latest_snapshot()
        self.targets = scan(db, top_n=top_n, include_casual=include_casual) if snap and db.me() else []
        picker = self.query_one("#def-picker", SelectionList)
        picker.clear_options()
        for target in self.targets:
            picker.add_option(Selection(_row(target), target.key, target.selected))
        self.query_one("#def-help", Static).update(_help(self.targets, snap, db.me(), top_n))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "casual" and not self.busy:
            self.reload()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "top-n" or self.busy:
            return
        try:
            value = int(event.value)
        except ValueError:
            return
        if value < 1 or value == self.top_n:
            return
        self.reload()

    def note(self, message: str) -> None:
        for line in str(message).splitlines():
            if line.strip():
                activity.log("defense", line)
        log = self.query_one("#def-log", SelectableRichLog)
        follow = log.is_vertical_scroll_end
        log.write(Text(message), scroll_end=follow)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "refresh":
            self.reload()
        elif event.button.id == "defend":
            self.start()
        elif event.button.id == "stop":
            self.stop_requested = True
            self.notify("stopping after the current word")

    def action_back(self) -> None:
        if self.busy:
            self.notify("defense is still running", severity="warning")
            return
        self.app.pop_screen()

    def start(self) -> None:
        if self.busy:
            return
        top_n = self._read_top()
        if top_n is None:
            return
        self.top_n = top_n
        chosen = set(self.query_one("#def-picker", SelectionList).selected)
        jobs = [(target.slug, target.play_mode) for target in self.targets if target.key in chosen]
        if not jobs:
            self.notify("nothing selected", severity="warning")
            return
        dry = self.query_one("#dry", Checkbox).value
        self.busy = True
        self.stop_requested = False
        self.query_one("#defend", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.note(f"{'DRY RUN: ' if dry else ''}defending {len(jobs)} board(s), top {top_n}")
        self.worker(jobs, dry, top_n)

    @work(thread=True, exclusive=True)
    def worker(self, jobs: list[tuple[str, str]], dry: bool, top_n: int) -> None:
        def write(message: str) -> None:
            self.app.call_from_thread(self.note, message)

        try:
            result = defend(jobs, top_n=top_n, dry_run=dry, log=write, halt=lambda: self.stop_requested)
        except Exception as error:
            write(f"defense crashed: {error!r}")
            result = None
        if result is not None:
            if dry:
                write(f"\nfinished: would post on {result.planned}/{len(jobs)} (dry run, nothing sent)")
            else:
                write(f"\nfinished: {result.posted} word(s) on {result.boards}/{len(jobs)} boards")
        self.app.call_from_thread(self._done)

    def _done(self) -> None:
        self.busy = False
        self.query_one("#defend", Button).disabled = False
        self.query_one("#stop", Button).disabled = True
        self.reload()


def _row(target: DefenseTarget) -> Text:
    text = Text()
    tag = {QUIET: "crowded", BARE: "nothing new", BLIND: "no phrase"}.get(target.status, "")
    style = {READY: "bold green", QUIET: "dim"}.get(target.status, "dim")
    text.append(f"{_mode_name(target.play_mode):<7} ", style="bold")
    text.append(f"{'+'.join(target.held):<18} ", style=style)
    if tag:
        text.append(f"[{tag}] ", style=style)
    plan = target.plan
    if target.status == READY:
        text.append(f"{plan.ours_in_top}/{plan.top_len or plan.ours_in_top} ours  {len(target.decoys):>2} decoys  ")
    elif target.status == QUIET:
        text.append(f"{plan.word_count} words, {plan.ours_in_top}/{plan.top_len} ours  ")
    text.append(f"{target.title[:36]:<36} ", style="bold")
    if target.phrase:
        text.append(target.phrase[:48], style="dim")
    return text


def _help(targets: list[DefenseTarget], snap, me: str, top_n: int) -> Text:
    text = Text()
    text.append("Defense  ", style="bold")
    if snap is None:
        text.append("No snapshot yet. Refresh snapshot from the home screen first.", style="bold red")
        return text
    if not me:
        text.append("This snapshot has no signed-in player. Refresh it while signed in.", style="bold red")
        return text
    counts = {READY: 0, QUIET: 0, BARE: 0, BLIND: 0}
    for target in targets:
        counts[target.status] = counts.get(target.status, 0) + 1
    if not targets:
        text.append("No boards where we lead. Publish a winner, then refresh the snapshot.", style="yellow")
    else:
        text.append(
            f"{counts[READY]} ready, preselected. {counts[QUIET]} already crowded. "
            f"{counts[BARE] + counts[BLIND]} with nothing safe to add."
        )
    text.append(
        "\nThe words list is per play mode, shared by Highest and Shortest, so one Strict pass covers both. "
        f"Each board gets up to {top_n} words that rank high on other boards. "
        "Words in our leading phrase, words already listed here, and words already planted are skipped. "
        "A board with 40 or more words whose top list is mostly other people is left unchecked. "
        "A decoy that takes a lead stays on the board. Casual has its own list.",
    )
    return text
