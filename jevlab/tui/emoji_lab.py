"""Emoji lab. Same regions as the Strict lab: queue, candidates, the arm that is scoring, and the log."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static

from ..boards import split_key, standings
from ..db import DB
from ..emoji.search import search
from ..modes import EMOJI, GameMode
from ..objective import site_round
from .chrome import EditionMark
from .copyselect import SelectableDataTable, SelectableRichLog
from .pickers import RunKeys, fmt_leader

ARMS = {
    "project": "First strings, from this edition's best Strict Highest lines.",
    "singles": "One glyph at a time, from the question and from the usage probe.",
    "climb": "Swap or append one glyph on the best string.",
    "compact": "Drop glyphs while the rounded score holds.",
}


def p_style(p: float, target: float) -> str:
    if p >= target and target > 0:
        return "bold green"
    if p >= 0.9:
        return "green"
    if p >= 0.6:
        return "yellow"
    if p >= 0.3:
        return "dark_orange"
    return "red"


@dataclass
class QueueItem:
    key: str
    slug: str
    title: str
    leader: str = "  empty  "
    leader_p: float = 0.0
    goal: str = ""
    state: str = "pending"
    best: str = "-"
    win: bool = False
    phrase: str = ""
    rows: list[dict] = field(default_factory=list)


class EmojiLabScreen(RunKeys, Screen):
    CSS = """
    EmojiLabScreen { layout: vertical; }
    #lab-header { height: 6; padding: 0 1; background: $boost; }
    #body { height: 1fr; }
    #queue-box { width: 64; height: 1fr; border: round $secondary; padding: 0 1; }
    #queue { height: auto; }
    #middle { width: 3fr; }
    #table { height: 1fr; border: round $primary; }
    #feed { height: 16; border: round $accent; }
    #side { width: 2fr; min-width: 62; }
    #timeline { height: 12; border: round $secondary; }
    #insight { height: 1fr; border: round $accent; padding: 0 1; }
    #lab-log { height: 6; border: round $primary; }
    """

    BINDINGS = [
        Binding("space", "toggle_pause", "pause/resume", priority=True),
        Binding("n", "next_question", "next question"),
        Binding("q", "leave", "stop & back"),
    ]

    pause_message = "paused after this call"
    leave_message = "stopping after this call"

    def __init__(self, slugs: list[str], budget: int, win_mode: bool, game_mode: GameMode, patience: int = 6):
        super().__init__()
        self.game_mode = game_mode
        self.budget_cap = budget
        self.win_mode = win_mode
        self.patience = max(1, patience)
        self.stalls = 0
        self.db = DB()
        self.arm = "-"
        self.calls = 0
        self.shown_budget = budget
        self.skip_question = False
        self.current_key = ""
        self.index = -1
        self.scrolled_to = -1
        found = {row.key: row for row in standings(self.db, mode=EMOJI, board=game_mode.board)}
        self.queue: list[QueueItem] = []
        for key in slugs:
            slug, _target = split_key(key)
            row = found.get(key)
            title = row.label if row else slug
            leader = fmt_leader(row) if row else "  empty  "
            leader_p = row.leader.probability if row and row.leader else 0.0
            goal = row.goal if row else ""
            self.queue.append(QueueItem(key, slug, title, leader, leader_p, goal))
        self.by_key = {item.key: item for item in self.queue}

    def compose(self) -> ComposeResult:
        yield EditionMark("Emoji", self.game_mode.label)
        yield Static(id="lab-header")
        with Horizontal(id="body"):
            with VerticalScroll(id="queue-box"):
                yield Static(id="queue")
            with Vertical(id="middle"):
                yield SelectableDataTable(
                    id="table",
                    cursor_type="row",
                    zebra_stripes=True,
                    cursor_foreground_priority="renderable",
                )
                yield SelectableRichLog(id="feed", wrap=False, markup=False, max_lines=600)
            with Vertical(id="side"):
                yield SelectableRichLog(id="timeline", wrap=False, markup=False, max_lines=300)
                yield Static(id="insight")
        yield SelectableRichLog(id="lab-log", wrap=False, markup=False, max_lines=1000)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.add_columns("#", "p", "w", "arm", "", "phrase")
        table.border_title = "candidates in board order   WIN = lower bound beats leader"
        order = "longest leader first" if self.game_mode.shortest else "weakest leader first"
        self.query_one("#queue-box").border_title = f"{self.game_mode.label} queue ({order})"
        self.query_one("#queue-box").border_subtitle = "question · leader · best"
        self.query_one("#timeline").border_title = "new bests"
        self.query_one("#feed").border_title = "now trying"
        self.query_one("#insight").border_title = "what's working"
        self.query_one("#lab-log").border_title = "log"
        self.refresh_view()
        self.begin_run()
        self._run()

    def halt(self) -> bool:
        """True ends the question. Skip ends just this one; stop ends the queue."""
        while self.paused and not self.stop_requested and not self.skip_question:
            time.sleep(0.05)
        if self.skip_question:
            self.skip_question = False
            return True
        return bool(self.stop_requested)

    def action_next_question(self) -> None:
        if not self.busy:
            return
        self.skip_question = True
        self.paused = False
        self.log_line("skipping to the next question")

    def log_line(self, message: str) -> None:
        log = self.query_one("#lab-log", SelectableRichLog)
        follow = log.is_vertical_scroll_end
        raw = str(message)
        line = Text(f"{time.strftime('%H:%M:%S')} ")
        line.append(raw, style="bold green" if "WIN" in raw else "")
        log.write(line, scroll_end=follow)

    def current_item(self) -> QueueItem | None:
        return self.by_key.get(self.current_key)

    def refresh_view(self) -> None:
        self.render_header()
        self.render_queue()
        self.render_insight()

    def render_header(self) -> None:
        current = self.current_item()
        width = self.query_one("#lab-header", Static).content_size.width or 120
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append(f"{self.game_mode.label}  ", style="bold reverse")
        if current is not None and self.index >= 0:
            header.append(f"[{self.index + 1}/{len(self.queue)}] ", style="bold magenta")
            tail = Text()
            if current.goal:
                tail.append("   goal ", style="dim")
                tail.append(current.goal, style="bold cyan")
            tail.append("   ")
            empty = current.leader.strip() in {"", "-", "empty"}
            tail.append("board empty" if empty else f"leader {current.leader.strip()}", style="bold yellow")
            tail.append("   best ")
            tail.append(current.best, style="bold green" if current.win else "bold")
            if current.win:
                tail.append(" WIN", style="bold green")
            if current.phrase:
                tail.append(f"  {current.phrase}", style="cyan")
            title_room = max(16, width - header.cell_len - tail.cell_len)
            header.append(current.title[:title_room], style="bold")
            header.append_text(tail)
        else:
            header.append("emoji", style="bold")
        header.truncate(width, overflow="ellipsis")
        header.append("\n")
        header.append_text(self.calls_bar(width))
        header.append("\n")
        status = Text(no_wrap=True, overflow="ellipsis")
        if self.arm in ARMS:
            status.append(f" {self.arm} ", style="bold cyan reverse")
            status.append(f"  {ARMS[self.arm]}", style="dim")
        else:
            status.append(self.arm or "starting", style="dim")
        status.truncate(width, overflow="ellipsis")
        header.append_text(status)
        self.query_one("#lab-header", Static).update(header)

    def calls_bar(self, width: int) -> Text:
        calls = self.calls
        budget = max(self.shown_budget, 1)
        filled = round(width * min(calls / budget, 1.0))
        bar = Text()
        color = "green" if self.current_item() and self.current_item().win else "cyan"
        bar.append("█" * filled, style=color)
        bar.append("░" * (width - filled), style="grey30")
        bar.append("\n")
        state = (self.current_item().state if self.current_item() else "starting").upper()
        label = (
            f" {state}   calls {calls}/{self.shown_budget} ({calls / budget:.0%})"
            f"   patience {self.stalls}/{self.patience}   {self.arm}"
        )
        bar.append(label[:width], style="bold")
        return bar

    def render_queue(self) -> None:
        text = Text(no_wrap=True, overflow="ellipsis")
        icons = {
            "pending": ("·", "dim"),
            "running": ("▶", "bold yellow"),
            "done": ("✓", "green"),
            "stopped": ("-", "dim"),
            "skipped": ("-", "dim"),
            "error": ("!", "red"),
        }
        for item in self.queue:
            icon, style = icons.get(item.state, ("·", "dim"))
            text.append(f"{icon} ", style=style)
            text.append(f"{item.title[:28]:<28} ", style="bold" if item.state == "running" else "")
            text.append(f"{item.leader:<9} ", style="yellow")
            text.append(f"{item.best:<11}", style="bold green" if item.win else "")
            if item.win:
                text.append(" WIN", style="bold green")
            text.append("\n")
        self.query_one("#queue", Static).update(text)
        if self.index != self.scrolled_to and self.index >= 0:
            self.query_one("#queue-box", VerticalScroll).scroll_to(y=max(0, self.index - 2), animate=False)
            self.scrolled_to = self.index

    def render_insight(self) -> None:
        text = Text()
        for name, blurb in ARMS.items():
            active = name == self.arm
            text.append(f" {name} ", style="bold cyan reverse" if active else "bold")
            if active:
                text.append(" running now", style="dim")
            text.append("\n")
            text.append(f"  {blurb}\n", style="" if active else "dim")
        self.query_one("#insight", Static).update(text)

    def target_p(self, item: QueueItem) -> float:
        if self.game_mode.shortest:
            return 0.51
        return item.leader_p

    def apply_progress(self, info: dict) -> None:
        key = info.get("key") or info.get("slug")
        item = self.by_key.get(key) or self.by_key.get(info.get("slug", ""))
        if item is None:
            return
        self.current_key = item.key
        self.index = next(i for i, row in enumerate(self.queue) if row.key == item.key)
        state = info.get("state")
        if state in {"running", "done", "stopped", "skipped", "error"}:
            item.state = state
        previous = item.best
        if info.get("best"):
            item.best = info["best"]
            item.win = bool(info.get("win"))
            item.phrase = info.get("phrase") or item.phrase
        if info.get("arm"):
            self.arm = info["arm"]
        if "calls" in info:
            self.calls = int(info["calls"])
            self.shown_budget = int(info.get("budget") or self.shown_budget)
        if "stalls" in info:
            self.stalls = int(info["stalls"])
            self.patience = int(info.get("patience") or self.patience)
        if item.best != previous and item.best not in {"", "-"}:
            self.note_best(item)
        if state == "batch":
            self.add_batch(item, info)
        self.refresh_view()

    def note_best(self, item: QueueItem) -> None:
        line = Text(f"{time.strftime('%H:%M:%S')} ")
        line.append(item.best, style="bold green" if item.win else "bold")
        if item.win:
            line.append(" WIN", style="bold green")
        line.append(f"  [{self.arm}] ", style="cyan")
        line.append(item.phrase)
        self.query_one("#timeline", SelectableRichLog).write(line)

    def mark(self, key: str, state: str) -> None:
        item = self.by_key.get(key)
        if item is None:
            return
        self.current_key = key
        self.index = next(i for i, row in enumerate(self.queue) if row.key == key)
        item.state = state
        if state == "running":
            self.calls = 0
            self.stalls = 0
            item.rows.clear()
            self.query_one("#feed", SelectableRichLog).clear()
            self.query_one("#table", DataTable).clear()
        self.refresh_view()

    def mark_finished(self, key: str) -> None:
        item = self.by_key.get(key)
        if item is not None and item.state == "running":
            item.state = "stopped" if self.stop_requested else "done"
            self.refresh_view()

    def add_batch(self, item: QueueItem, info: dict) -> None:
        rows = list(info.get("scored") or [])
        if not rows:
            return
        item.rows.extend(rows)
        scores = [float(row["p"]) for row in rows]
        arm = str(info.get("arm") or "-")
        target = self.target_p(item)
        head = Text()
        head.append(f"{time.strftime('%H:%M:%S')} ", style="bold")
        head.append(f"{arm[:16]:<16} ", style="bold cyan")
        head.append(f"{len(rows):4} scored  best ")
        head.append(f"{max(scores):.3f}", style=p_style(max(scores), target))
        head.append(f"  mean {sum(scores) / len(scores):.3f}")
        hits = sum(1 for row in rows if row.get("win"))
        if hits:
            head.append(f"  {hits} WIN", style="bold green")
        self.query_one("#feed", SelectableRichLog).write(head)
        ranked = sorted(rows, key=lambda row: self.row_key(row), reverse=True)
        for row in ranked[:8]:
            line = Text("   ")
            p = float(row["p"])
            line.append(f"{p:.3f}", style=p_style(p, target))
            line.append(f" {int(row['units']):2}w ", style="dim")
            line.append(f"{(row.get('arm') or ''):<10} ", style="cyan")
            if row.get("win"):
                line.append("WIN ", style="bold green")
            else:
                line.append("    ")
            line.append(str(row["phrase"])[:110])
            self.query_one("#feed", SelectableRichLog).write(line)
        self.fill_table(item)

    def fill_table(self, item: QueueItem) -> None:
        ranked = sorted(item.rows, key=lambda row: self.row_key(row), reverse=True)
        table = self.query_one("#table", DataTable)
        table.clear()
        target = self.target_p(item)
        for i, row in enumerate(ranked, start=1):
            p = float(row["p"])
            mark = Text("WIN", style="bold green") if row.get("win") else Text("")
            table.add_row(
                str(i),
                Text(f"{p:.3f}", style=p_style(p, target)),
                str(row["units"]),
                row.get("arm") or "",
                mark,
                row["phrase"],
                key=str(i),
            )

    def row_key(self, row: dict) -> tuple:
        p = float(row["p"])
        units = int(row["units"])
        rounded = site_round(p)
        if self.game_mode.shortest:
            qual = 1 if p >= 0.51 else 0
            return (qual, -units, rounded)
        return (rounded, -units)

    @work(thread=True, exclusive=True)
    def _run(self) -> None:
        def write(message: str) -> None:
            self.app.call_from_thread(self.log_line, message)

        def on_progress(info: dict) -> None:
            self.app.call_from_thread(self.apply_progress, info)

        write(f"emoji: {len(self.queue)} question(s), budget {self.budget_cap}")
        try:
            for item in self.queue:
                if self.stop_requested:
                    self.app.call_from_thread(self.mark, item.key, "stopped")
                    break
                self.app.call_from_thread(self.mark, item.key, "running")

                def progress(info: dict, key=item.key) -> None:
                    on_progress({"key": key, **info})

                try:
                    asyncio.run(
                        search(
                            self.db,
                            item.slug,
                            board=self.game_mode.board,
                            budget=self.budget_cap,
                            log=write,
                            halt=self.halt,
                            progress=progress,
                            win_mode=self.win_mode,
                            patience=self.patience,
                        )
                    )
                except Exception as error:
                    write(f"{item.slug} failed: {type(error).__name__}: {error}")
                    self.app.call_from_thread(self.mark, item.key, "error")
                    continue
                self.app.call_from_thread(self.mark_finished, item.key)
                if self.stop_requested:
                    break
        finally:
            self.app.call_from_thread(self._finish)

    def _finish(self) -> None:
        if self.finish_run():
            return
        self.notify("emoji search finished")
