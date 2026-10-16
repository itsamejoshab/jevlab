"""Defense: pick boards we lead and post decoy words onto their public word lists."""

from __future__ import annotations

import re

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, SelectionList, Static
from textual.widgets.selection_list import Selection

from .. import activity
from ..db import DB
from ..defense import BARE, BLIND, QUIET, READY, DefenseTarget, defend, scan
from ..modes import CASUAL, GOLF
from .chrome import EditionMark
from .copyselect import SelectableRichLog
from .pickers import Setting


def _mode_name(play_mode: str) -> str:
    if play_mode == CASUAL:
        return "Casual"
    if play_mode == GOLF:
        return "Golf"
    return "Strict"


_OPENING = re.compile(r"^(?:DRY RUN: )?defending (\d+) board")


class DefenseLog:
    """Plain defense lines, with a cyan [1/54] counter on each board header."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0

    def render(self, message: str) -> Text:
        lines = str(message).splitlines()
        if not lines:
            return Text("")
        out = Text()
        for index, raw in enumerate(lines):
            if index:
                out.append("\n")
            out.append_text(self._line(raw))
        return out

    def _line(self, raw: str) -> Text:
        line = raw.strip()
        if match := _OPENING.match(line):
            self.total = int(match.group(1))
            self.count = 0
            return Text(raw)
        if line.startswith("== "):
            self.count += 1
            mark = f"[{self.count}/{self.total}]" if self.total else f"[{self.count}]"
            text = Text()
            text.append(f"{mark} ", style="bold cyan")
            text.append(raw)
            return text
        return Text(raw)


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
    .hint { width: 1fr; height: 1; margin-left: 1; color: $text-muted; }
    #def-help { height: auto; padding: 0 1 1 1; }
    #def-picker { height: 1fr; border: round $primary; }
    #def-log { width: 2fr; min-width: 42; max-width: 72; height: 1fr; border: round $secondary; margin-left: 1; }
    """

    BINDINGS = [Binding("escape", "back", "back")]

    def __init__(self):
        super().__init__()
        self.targets: list[DefenseTarget] = []
        self.busy = False
        self.stop_requested = False
        self.transcript = DefenseLog()

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
                        yield Setting("Word by word", False, id="by-word", compact=True)
                        yield Static("one row per word on the words table", classes="hint")
                    with Horizontal(classes="opt"):
                        yield Setting("Include golf (no longer on website)", False, id="golf", compact=True)
                        yield Static("leave this off", classes="hint")
                    with Horizontal(classes="opt"):
                        yield Setting("Include casual", False, id="casual", compact=True)
                        yield Static("casual keeps its own word list", classes="hint")
                yield Static(id="def-help")
                yield SelectionList[str](id="def-picker")
            yield SelectableRichLog(id="def-log", wrap=True, markup=False, max_lines=2000)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#def-picker").border_title = "boards we lead"
        self.query_one("#def-log").border_title = "defense"
        self.reload()

    def reload(self) -> None:
        include_casual = self.query_one("#casual", Checkbox).value
        include_golf = self.query_one("#golf", Checkbox).value
        db = DB()
        snap = db.latest_snapshot()
        self.targets = scan(db, include_casual=include_casual, include_golf=include_golf) if snap and db.me() else []
        picker = self.query_one("#def-picker", SelectionList)
        picker.clear_options()
        for target in self.targets:
            picker.add_option(Selection(_row(target), target.key, target.selected))
        self.query_one("#def-help", Static).update(_help(self.targets, snap, db.me()))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id in {"casual", "golf"} and not self.busy:
            self.reload()

    def note(self, message: str) -> None:
        for line in str(message).splitlines():
            if line.strip():
                activity.log("defense", line)
        log = self.query_one("#def-log", SelectableRichLog)
        follow = log.is_vertical_scroll_end
        log.write(self.transcript.render(message), scroll_end=follow)

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
        chosen = set(self.query_one("#def-picker", SelectionList).selected)
        jobs = [(target.slug, target.play_mode) for target in self.targets if target.key in chosen]
        if not jobs:
            self.notify("nothing selected", severity="warning")
            return
        dry = self.query_one("#dry", Checkbox).value
        word_by_word = self.query_one("#by-word", Checkbox).value
        self.busy = True
        self.stop_requested = False
        self.query_one("#defend", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.note(f"{'DRY RUN: ' if dry else ''}defending {len(jobs)} board(s)")
        self.worker(jobs, dry, word_by_word)

    @work(thread=True, exclusive=True)
    def worker(self, jobs: list[tuple[str, str]], dry: bool, word_by_word: bool) -> None:
        def write(message: str) -> None:
            self.app.call_from_thread(self.note, message)

        try:
            result = defend(jobs, dry_run=dry, log=write, halt=lambda: self.stop_requested, word_by_word=word_by_word)
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
    tag = {READY: "at-risk", QUIET: "crowded", BARE: "nothing new", BLIND: "no phrase"}.get(target.status, "")
    style = {READY: "bold green", QUIET: "dim"}.get(target.status, "dim")
    text.append(f"{_mode_name(target.play_mode):<7} ", style="bold")
    text.append(f"{'+'.join(target.held):<18} ", style=style)
    if tag:
        text.append(f"[{tag}] ", style=style)
    plan = target.plan
    if target.status == READY:
        count = len(target.decoys)
        label = f"{count}/{plan.need}" if plan.need and count < plan.need else f"{count:>2}"
        text.append(f"{plan.ours_in_top}/{plan.top_len or plan.ours_in_top} {label} decoys  ")
    elif target.status == QUIET:
        text.append(f"{plan.answer_words}/{plan.word_count}  ")
    text.append(f"{target.title[:36]:<36} ", style="bold")
    if target.phrase:
        text.append(target.phrase[:48], style="dim")
    return text


def _help(targets: list[DefenseTarget], snap, me: str) -> Text:
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
        "\nThe words list is per play mode. Highest and Shortest share it, so one Strict pass covers both. "
        "Casual has its own. "
        "A board is left unselected once the list has 40 or more words and our answer is at most "
        "a quarter of the top 20 by impact, or at most a quarter of the whole list. "
        "Each pass adds a few more words than that line needs, so it still finishes if a banned word is skipped. "
        "Words in our leading phrase, words already on the list, words already planted, and stop words are skipped.",
    )
    return text
