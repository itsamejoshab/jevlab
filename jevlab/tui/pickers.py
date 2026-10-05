"""Home, Search setup, and Publish screens."""

from __future__ import annotations

import time

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Input, Label, RichLog, Rule, Select, SelectionList, Static, Switch
from textual.widgets.selection_list import Selection

from .. import activity, vault
from ..boards import Standing, standings
from ..config import EDITION, EDITIONS
from ..db import DB
from ..modes import GOLF, HOME_MODES, GameMode, from_name
from ..search.engine import LEVELS
from .copyselect import SelectableRichLog
from .lab import LabScreen


def fmt_leader(s: Standing, u: str = "w") -> str:
    return f"{s.leader.probability:.2f}/{s.leader.units:>2}{u}" if s.leader else "  empty  "


def fmt_ours(s: Standing, u: str = "w") -> str:
    return f"{s.ours.probability:.2f}/{s.ours.units:>2}{u}" if s.ours else "    -    "


BANNER_GLYPHS = {
    "J": ["     ██╗", "     ██║", "     ██║", "██   ██║", "╚█████╔╝", " ╚════╝ "],
    "K": ["██╗  ██╗", "██║ ██╔╝", "█████╔╝ ", "██╔═██╗ ", "██║  ██╗", "╚═╝  ╚═╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "V": ["██╗   ██╗", "██║   ██║", "██║   ██║", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚═══╝  "],
    "L": ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "B": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██████╔╝", "╚═════╝ "],
    "Y": ["██╗   ██╗", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚██╔╝  ", "   ██║   ", "   ╚═╝   "],
    "C": [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
    "F": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "██║     ", "╚═╝     "],
}
BANNER = {"kev": "KEVLAB", "laya": "LAYALAB", "clef": "CLEFLAB"}.get(EDITION, "JEVLAB")
BANNER_FACE = ["#ffffff", "#e4eef8", "#c6d9ec", "#a8c4e0", "#8aafd4", "#6c9ac8"]
BANNER_SHADOW = "#3a4f66"
BANNER_ACCENT = "#8aafd4"
ACRONYM = ["Joint", "Experimental", "Verbal", "Leverage", "Analysis", "Bureau"]
CLASSIFICATION = "TOP SECRET // JEV-ORCON // NOFORN // EYES ONLY"


def banner() -> Text:
    text = Text(justify="center", no_wrap=True)
    for row, face in enumerate(BANNER_FACE):
        line = "   ".join(BANNER_GLYPHS[letter][row] for letter in BANNER)
        for ch in line:
            text.append(ch, style=f"bold {face}" if ch == "█" else BANNER_SHADOW)
        text.append("\n")
    text.append("\n")
    for i, word in enumerate(ACRONYM):
        text.append(("  " if i else "") + word[0], style=f"bold {BANNER_ACCENT}")
        text.append(word[1:].upper(), style="dim")
    return text


def dossier() -> Text:
    text = Text(justify="center")
    text.append(
        "Directorate of Offline Adversarial Lexicography & Stochastic Oracle Interrogation\n", style="bold italic"
    )
    theatre = {"kev": "Kev", "laya": "Laya", "clef": "Clef"}.get(EDITION, "Jev")
    text.append(
        f"Trick {theatre} Theatre of Operations  ·  Special Access Programme JEV-7/Ω  ·  Sector 12-B", style="dim"
    )
    return text


def card(value: Text | str, caption: str) -> Text:
    text = Text(justify="center")
    text.append_text(value if isinstance(value, Text) else Text(value, style="bold"))
    text.append(f"\n{caption}", style="dim")
    return text


class HomeScreen(Screen):
    CSS = """
    HomeScreen { align: center middle; }
    #home {
        width: 100; height: auto; border: double $primary; padding: 1 2; background: $surface;
        border-title-color: $error; border-title-style: bold; border-title-align: center;
        border-subtitle-color: $text-muted; border-subtitle-align: right;
    }
    #home-banner { width: 100%; text-align: center; text-wrap: nowrap; }
    #home-dossier { width: 100%; margin-top: 1; text-align: center; }
    #home-rule { color: $primary-darken-2; margin: 1 0 0 0; }
    #home-mode { height: 3; margin-top: 1; align: center middle; }
    #home-mode Label { padding: 1 1 0 0; text-style: bold; color: $accent; }
    #game-mode { width: 40; }
    #edition { width: 16; margin-left: 2; }
    #home-cards { height: 4; margin-top: 1; }
    .card { width: 1fr; height: 4; margin: 0 1; border: round $primary-darken-1; content-align: center middle; text-align: center;
            border-title-color: $accent; border-title-style: bold; border-title-align: center; }
    #home-stats { width: 100%; margin: 1 0 0 0; text-align: center; }
    #home-buttons { height: 5; align: center middle; }
    #home-buttons Button { margin: 0 2; min-width: 20; }
    #home-log { height: 12; border: round $secondary; display: none; }
    """

    BINDINGS = [
        Binding("s", "search", "Search"),
        Binding("p", "publish", "Publish"),
        Binding("r", "refresh", "Refresh snapshot"),
        Binding("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="home"):
            yield Static(banner(), id="home-banner")
            yield Static(dossier(), id="home-dossier")
            yield Rule(id="home-rule", line_style="heavy")
            with Horizontal(id="home-mode"):
                yield Label("GAME MODE")
                yield Select(
                    [(m.label, m.name) for m in HOME_MODES],
                    value=self.app.game_mode.name,
                    allow_blank=False,
                    id="game-mode",
                )
                yield Select([(e.capitalize(), e) for e in EDITIONS], value=EDITION, allow_blank=False, id="edition")
            with Horizontal(id="home-cards"):
                yield Static(id="card-lead", classes="card")
                yield Static(id="card-ready", classes="card")
                yield Static(id="card-vault", classes="card")
            yield Static(id="home-stats")
            with Horizontal(id="home-buttons"):
                yield Button("Search", id="search", variant="primary")
                yield Button("Publish", id="publish", variant="success")
                yield Button("Refresh snapshot", id="refresh")
                yield Button("Quit", id="quit", variant="error")
            yield SelectableRichLog(id="home-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        home = self.query_one("#home")
        home.border_title = f" ▌ {CLASSIFICATION} ▐ "
        self.query_one("#card-lead").border_title = "BOARDS HELD"
        self.query_one("#card-ready").border_title = "STRIKE READY"
        self.query_one("#card-vault").border_title = "VAULT"
        self.tick()
        self.set_interval(1, self.tick)
        self.update_stats()

    def tick(self) -> None:
        self.query_one("#home").border_subtitle = f" SECURE TERMINAL 7  ·  {time.strftime('%H:%M:%S')} "

    def on_screen_resume(self) -> None:
        self.update_stats()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "game-mode" and isinstance(event.value, str):
            self.app.game_mode = from_name(event.value)
            self.update_stats()
        elif event.select.id == "edition" and isinstance(event.value, str) and event.value != EDITION:
            self.app.exit(event.value)

    def update_stats(self) -> None:
        mode: GameMode = self.app.game_mode
        db = DB()
        snap = db.latest_snapshot()
        text = Text()
        lead_card = self.query_one("#card-lead", Static)
        ready_card = self.query_one("#card-ready", Static)
        vault_card = self.query_one("#card-vault", Static)
        if not mode.publishable:
            self.query_one("#card-lead").border_title = "RACE"
            self.query_one("#card-ready").border_title = "AUTO"
            self.query_one("#card-vault").border_title = "HOLD"
            lead_card.update(card("race", "site live round"))
            ready_card.update(card("auto", "fast first word"))
            vault_card.update(card("hold", "slow if dethroned"))
            text.append(
                "Search watches /round. Fast fires a word the moment a round appears, then "
                "reposts on every score gain. Slow posts only when we are not in 1st and does not spam.",
                style="bold",
            )
            self.query_one("#home-stats", Static).update(text)
            return
        self.query_one("#card-lead").border_title = "BOARDS HELD"
        self.query_one("#card-ready").border_title = "STRIKE READY"
        self.query_one("#card-vault").border_title = "VAULT"
        if not snap:
            for widget in (lead_card, ready_card, vault_card):
                widget.update(card("—", "awaiting intel"))
            text.append("No snapshot yet. Press Refresh snapshot first.", style="bold red")
            self.query_one("#home-stats", Static).update(text)
            return
        rows = standings(db, mode=mode.play_mode, board=mode.board)
        searchable = [s for s in rows if s.searchable]
        leading = sum(1 for s in searchable if s.we_lead)
        ready = sum(1 for s in rows if s.entry_beats)
        entries = [e for e in vault.all_entries(mode.board) if e["mode"] == mode.play_mode]
        published = sum(1 for e in entries if e.get("status") == "published")

        lead_value = Text()
        lead_value.append(f"{leading}", style="bold green")
        lead_value.append(f" / {len(searchable)}", style="bold")
        lead_card.update(card(lead_value, f"{mode.label}: we lead"))
        ready_card.update(
            card(Text(f"{ready}", style="bold yellow" if ready else "bold"), "vault lines beat the leader")
        )
        vault_value = Text()
        vault_value.append(f"{len(entries)}", style="bold cyan")
        vault_value.append("  ·  ", style="dim")
        vault_value.append(f"{published}", style="bold magenta")
        vault_card.update(card(vault_value, "saved  ·  published"))

        text.append("SNAPSHOT ", style="bold")
        text.append(f"{snap['taken_at']}  ·  {len(rows)} questions, {len(searchable)} searchable", style="dim")
        if not mode.searchable:
            text.append(
                "\nPublish only: fill this vault with `jevlab crosspost` from the Strict vault.", style="yellow"
            )
        self.query_one("#home-stats", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        {
            "search": self.action_search,
            "publish": self.action_publish,
            "refresh": self.action_refresh,
            "quit": self.app.exit,
        }[event.button.id]()

    def action_search(self) -> None:
        if not self.app.game_mode.publishable:
            self.app.push_screen(LiveSetupScreen())
            return
        if not self.app.game_mode.searchable:
            self.notify(
                f"{self.app.game_mode.label} is publish only; run `jevlab crosspost` to fill its vault",
                severity="warning",
            )
            return
        self.app.push_screen(SearchSetupScreen(self.app.game_mode))

    def action_publish(self) -> None:
        if not self.app.game_mode.publishable:
            self.notify("Live Mode posts from Search while the round is live", severity="warning")
            return
        self.app.push_screen(PublishScreen(self.app.game_mode))

    def action_refresh(self) -> None:
        self.app.push_screen(SnapshotScreen())


SNAPSHOT_MODES = (("Strict", "strict_chain"), ("Golf", "golf"), ("Casual", "word_chain"))


class SnapshotScreen(Screen):
    """Pick play modes for this edition. Other editions are shown disabled."""

    CSS = """
    #snap-help { height: auto; padding: 0 1; }
    #snap-editions, #snap-modes { height: 3; padding: 0 1; }
    #snap-editions Checkbox, #snap-modes Checkbox { margin-right: 2; }
    #snap-actions { height: 3; }
    #snap-actions Button { margin: 0 1; }
    #snap-log { height: 1fr; border: round $secondary; }
    """

    BINDINGS = [Binding("escape", "back", "back")]

    def compose(self) -> ComposeResult:
        yield Static(id="snap-help")
        with Horizontal(id="snap-editions"):
            for edition in EDITIONS:
                yield Checkbox(
                    edition.capitalize(),
                    value=edition == EDITION,
                    disabled=edition != EDITION,
                    id=f"snap-ed-{edition}",
                )
        with Horizontal(id="snap-modes"):
            for label, mode in SNAPSHOT_MODES:
                yield Checkbox(label, value=True, id=f"snap-mode-{mode}")
        with Horizontal(id="snap-actions"):
            yield Button("Start", id="start", variant="success")
            yield Button("Back", id="back")
        yield SelectableRichLog(id="snap-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#snap-help", Static).update(
            f"Snapshot {EDITION}. Other editions are disabled; switch to snapshot them. "
            "Strict and Golf each fill Highest and Shortest."
        )
        self.query_one("#snap-log").border_title = "snapshot"
        self.busy = False

    def action_back(self) -> None:
        if self.busy:
            self.notify("snapshot is still running", severity="warning")
            return
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "start":
            self.start()

    def start(self) -> None:
        if self.busy:
            return
        modes = tuple(mode for _label, mode in SNAPSHOT_MODES if self.query_one(f"#snap-mode-{mode}", Checkbox).value)
        if not modes:
            self.notify("pick at least one play mode", severity="warning")
            return
        self.busy = True
        self.query_one("#start", Button).disabled = True
        self.snapshot_worker(modes)

    @work(thread=True, exclusive=True)
    def snapshot_worker(self, modes: tuple[str, ...]) -> None:
        from ..site.snapshot import take_snapshot

        log = self.query_one("#snap-log", RichLog)

        def write(message: str) -> None:
            activity.log("snapshot", message)
            self.app.call_from_thread(log.write, f"{time.strftime('%H:%M:%S')} {message}")

        try:
            take_snapshot(DB(), modes=modes, log=write)
        except Exception as error:
            write(f"snapshot failed: {error!r}")

        def done() -> None:
            self.busy = False
            self.query_one("#start", Button).disabled = False

        self.app.call_from_thread(done)


class LiveSetupScreen(Screen):
    CSS = """
    #live-help { height: auto; padding: 0 1; }
    #live-question { height: auto; border: round $accent; padding: 0 1;
                     border-title-color: $accent; border-title-style: bold; }
    #live-now { height: auto; }
    #live-panel { height: auto; border: round $accent; padding: 0 1;
                  border-title-color: $accent; border-title-style: bold; }
    #params { height: auto; }
    .field { width: auto; height: auto; margin-right: 3; }
    .field Label { color: $text-muted; text-style: bold; }
    #pace { width: 32; }
    #chain { width: 34; }
    #max-level { width: 24; }
    .field Switch { margin-left: 1; }
    #actions { height: 3; margin-top: 1; }
    #actions Button { margin-right: 1; min-width: 12; }
    #actions-spacer { width: 1fr; }
    #go { min-width: 26; margin-right: 0; }
    """

    BINDINGS = [
        Binding("g", "go", "GO"),
        Binding("escape", "back", "back"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(id="live-help")
        with Vertical(id="live-question"):
            yield Static(id="live-now")
        with Vertical(id="live-panel"):
            with Horizontal(id="params"):
                with Vertical(classes="field"):
                    yield Label("PACE")
                    yield Select(
                        [
                            ("FAST  first word, every gain", "fast"),
                            ("SLOW  only when not 1st", "slow"),
                        ],
                        value="fast",
                        allow_blank=False,
                        id="pace",
                    )
                with Vertical(classes="field"):
                    yield Label("CHAIN")
                    yield Select(
                        [
                            ("CASUAL  phrase as one word", "casual"),
                            ("STRICT  one word at a time", "strict"),
                        ],
                        value="casual",
                        allow_blank=False,
                        id="chain",
                    )
                with Vertical(classes="field"):
                    yield Label("LLM IDEAS")
                    yield Switch(True, id="llm")
                with Vertical(classes="field"):
                    yield Label("POST")
                    yield Switch(True, id="post")
                with Vertical(classes="field"):
                    yield Label("ESCALATE")
                    yield Switch(True, id="escalate")
                with Vertical(classes="field"):
                    yield Label("UP TO LEVEL")
                    yield Select(
                        [(f"L{i}  {name}", i) for i, (name, *_) in enumerate(LEVELS)],
                        value=len(LEVELS) - 1,
                        allow_blank=False,
                        id="max-level",
                    )
            with Horizontal(id="actions"):
                yield Static(id="actions-spacer")
                yield Button("GO", id="go", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#live-question").border_title = "LIVE ROUND"
        self.query_one("#live-panel").border_title = "RUN PARAMETERS"
        self.update_help()
        self.refresh_round()

    def update_help(self) -> None:
        pace = self.query_one("#pace", Select).value
        text = Text()
        text.append("Live Mode  ", style="bold")
        if pace == "fast":
            text.append(
                "FAST: reads the next question immediately, posts the best saved line when this question "
                "was already scanned, otherwise fires one word, then reposts on every score gain. "
            )
        else:
            text.append("SLOW: posts a saved line or a new leader only when we are not in 1st, then holds. ")
        if self.query_one("#chain", Select).value == "strict":
            text.append("STRICT: one word at a time on a word-chain round. ")
        else:
            text.append("CASUAL: a whole phrase goes up as one hyphenated word on a word-chain round. ")
        text.append("A sentence round submits the phrase in one turn, on the mode the round names. ")
        if not self.query_one("#post", Switch).value:
            text.append("POST is off: the search runs and nothing is sent. p toggles it during the run.")
        self.query_one("#live-help", Static).update(text)

    def refresh_round(self) -> None:
        from ..live import LiveSession
        from ..site.client import SiteClient, SiteError

        now = Text()
        try:
            state = SiteClient().round_state("live")
        except SiteError as error:
            now.append(f"Could not read the live round: {error}", style="bold red")
            self.query_one("#live-now", Static).update(now)
            return
        session = LiveSession()
        session.observe(state, me="")
        if session.status != "live" or not session.question:
            now.append(
                f"No live round right now ({session.status or 'idle'}). GO still waits for the next one.",
                style="yellow",
            )
        else:
            now.append(session.question["title"], style="bold")
            site_mode = "strict" if state.get("playMode") == "strict_chain" else "casual"
            how = f"submit a phrase ({site_mode})" if state.get("sentences") else f"round is {site_mode}"
            now.append(f"\n{how}  ·  finish at {session.threshold:.0%}  ·  ", style="dim")
            lead = session.leader
            now.append(
                f"leader {lead.probability:.2f}/{lead.units}w {lead.name}" if lead else "nobody racing yet",
                style="bold yellow" if lead else "dim",
            )
        self.query_one("#live-now", Static).update(now)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("pace", "chain"):
            self.update_help()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "post":
            self.update_help()
        elif event.switch.id == "escalate":
            self.query_one("#max-level", Select).disabled = not event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self.action_go()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_go(self) -> None:
        max_level = self.query_one("#max-level", Select).value
        use_llm = self.query_one("#llm", Switch).value
        escalate = self.query_one("#escalate", Switch).value
        post = self.query_one("#post", Switch).value
        pace = self.query_one("#pace", Select).value
        chain = self.query_one("#chain", Select).value
        self.app.switch_screen(
            LabScreen(
                [],
                budget=10**9,
                use_llm=use_llm,
                max_stall=0,
                escalate=escalate,
                max_level=max_level,
                live_play=True,
                live_pace=pace if pace in ("fast", "slow") else "fast",
                live_chain="strict" if chain == "strict" else "casual",
                live_post=bool(post),
            )
        )


class SearchSetupScreen(Screen):
    CSS = """
    #setup-help { height: auto; padding: 0 1; }
    #picker { height: 1fr; border: round $primary; }
    #run-panel { height: auto; border: round $accent; padding: 0 1;
                 border-title-color: $accent; border-title-style: bold; }
    #params { height: auto; }
    .field { width: auto; height: auto; margin-right: 3; }
    .field Label { color: $text-muted; text-style: bold; }
    #budget { width: 18; }
    #stall { width: 14; }
    #max-level { width: 24; }
    #win-extra { width: 14; }
    .field Switch { margin-left: 1; }
    #actions { height: 3; margin-top: 1; }
    #actions Label { padding: 1 1 0 0; color: $text-muted; }
    #actions Button { margin-right: 1; min-width: 12; }
    #actions-spacer { width: 1fr; }
    #target-player { width: 36; }
    #go { min-width: 26; margin-right: 0; }
    """

    BINDINGS = [
        Binding("g", "go", "GO"),
        Binding("a", "select_all", "all"),
        Binding("x", "select_none", "none"),
        Binding("l", "select_not_leading", "not leading"),
        Binding("e", "select_empty", "empty"),
        Binding("t", "target_player", "target player"),
        Binding("escape", "back", "back"),
    ]

    def __init__(self, game_mode: GameMode):
        super().__init__()
        self.game_mode = game_mode
        self.rows = sorted(
            standings(DB(), board=game_mode.board, targets=True), key=lambda s: (not s.searchable, s.search_order)
        )

    def compose(self) -> ComposeResult:
        yield Static(id="setup-help")
        yield SelectionList[str](*self.picker_options(), id="picker")
        with Vertical(id="run-panel"):
            with Horizontal(id="params"):
                with Vertical(classes="field"):
                    yield Label("ORACLE CALLS / QUESTION")
                    yield Input("3000", id="budget", type="integer", max_length=7)
                with Vertical(classes="field"):
                    yield Label("PATIENCE")
                    yield Input("12", id="stall", type="integer", max_length=3)
                with Vertical(classes="field"):
                    yield Label("LLM IDEAS")
                    yield Switch(True, id="llm")
                with Vertical(classes="field"):
                    yield Label("ESCALATE")
                    yield Switch(True, id="escalate")
                with Vertical(classes="field"):
                    yield Label("UP TO LEVEL")
                    yield Select(
                        [(f"L{i}  {name}", i) for i, (name, *_) in enumerate(LEVELS)],
                        value=len(LEVELS) - 1,
                        allow_blank=False,
                        id="max-level",
                    )
                with Vertical(classes="field"):
                    yield Label("WIN MODE")
                    yield Switch(False, id="win-mode")
                with Vertical(classes="field"):
                    yield Label("CALLS AFTER LEAD")
                    yield Input("2000", id="win-extra", type="integer", max_length=6, disabled=True)
            with Horizontal(id="actions"):
                yield Label("select")
                yield Button("All", id="all")
                yield Button("None", id="none")
                yield Button("Not leading", id="not-leading")
                yield Button("Empty", id="empty")
                yield Select(self.player_options(), prompt="Target player", id="target-player")
                yield Static(id="actions-spacer")
                yield Button("GO", id="go", variant="success")
        yield Footer()

    def targetable(self, s: Standing) -> bool:
        return s.searchable and not s.we_lead and not s.unwinnable and s.leader is not None

    def player_options(self) -> list[tuple[str, str]]:
        """Players leading at least one board we could take, most boards first."""
        counts: dict[str, int] = {}
        for s in self.rows:
            if self.targetable(s) and s.leader.name:
                counts[s.leader.name] = counts.get(s.leader.name, 0) + 1
        return [(f"{name} ({n})", name) for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def picker_options(self) -> list[Selection]:
        out = []
        for s in self.rows:
            prompt = Text()
            if s.searchable and s.leader is None:
                prompt.append("EMPTY           ", style="bold cyan")
            else:
                prompt.append(f"lead {fmt_leader(s)} ", style="yellow")
                prompt.append(f"{(s.leader.name if s.leader else '')[:14]:<14}  ", style="dim yellow")
            prompt.append(f"ours {fmt_ours(s)}  ", style="green" if s.we_lead else "")
            prompt.append("LEAD " if s.we_lead else "     ", style="bold green")
            if s.best_entry:
                prompt.append(
                    f"vault {s.best_entry['p_mean']:.2f}/{s.best_entry['units']:>2}w ",
                    style="bold yellow" if s.entry_beats else "dim",
                )
            else:
                prompt.append(" " * 17)
            mem = s.memory
            if mem and mem.runs:
                prompt.append(f"runs {mem.runs:<2} ", style="cyan")
            else:
                prompt.append(" " * 8)
            prompt.append(s.label[:80], style="" if s.searchable and not s.unwinnable else "dim strike")
            if not s.searchable:
                prompt.append(f"  ({s.kind}: not searchable yet)", style="dim")
            elif s.unwinnable:
                prompt.append("  (unwinnable: leader has 1.00 in 1 word, we could only tie)", style="dim")
            elif mem and mem.runs and not mem.wins:
                stalled = f"; stalled at L{mem.level} {LEVELS[mem.level][0]}" if 0 < mem.level < len(LEVELS) else ""
                prompt.append(
                    f"  (best so far {mem.best_p:.2f}/{mem.best_units}w{stalled}; next run starts fresh)",
                    style="dim cyan",
                )
            out.append(Selection(prompt, s.key, s.should_search, disabled=not s.searchable))
        return out

    def on_mount(self) -> None:
        order = "longest leader first" if self.game_mode.shortest else "weakest leader first"
        self.query_one("#picker").border_title = f"{self.game_mode.label}: questions, {order} (this is the run order)"
        self.query_one("#run-panel").border_title = "RUN PARAMETERS"
        self.query_one("#run-panel").border_subtitle = (
            "patience = flat rounds before escalating (or moving on)   "
            "win mode = once we take the lead, stop after that many more calls"
        )
        self.update_help()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "escalate":
            self.query_one("#max-level", Select).disabled = not event.value
        elif event.switch.id == "win-mode":
            self.query_one("#win-extra", Input).disabled = not event.value

    def update_help(self) -> None:
        picked = len(self.query_one("#picker", SelectionList).selected)
        go = self.query_one("#go", Button)
        go.label = f"GO  ·  {picked} question{'' if picked == 1 else 's'}"
        go.disabled = not picked
        text = Text()
        text.append(f"Search {self.game_mode.label}  ", style="bold")
        unwinnable = sum(1 for s in self.rows if s.searchable and s.unwinnable)
        order = "longest leader" if self.game_mode.shortest else "weakest leader"
        text.append(
            f"{picked} selected. Preselected: every searchable board we are not leading "
            f"({unwinnable} unwinnable boards left out). Space toggles, "
            f"GO runs them one at a time starting with the {order}."
        )
        self.query_one("#setup-help", Static).update(text)

    def on_selection_list_selected_changed(self, _event) -> None:
        self.update_help()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        {
            "all": self.action_select_all,
            "none": self.action_select_none,
            "not-leading": self.action_select_not_leading,
            "empty": self.action_select_empty,
            "go": self.action_go,
        }[event.button.id]()

    def action_select_all(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        for s in self.rows:
            if s.searchable:
                picker.select(s.key)

    def action_select_none(self) -> None:
        self.query_one("#picker", SelectionList).deselect_all()

    def action_select_not_leading(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        picker.deselect_all()
        for s in self.rows:
            if s.should_search:
                picker.select(s.key)

    def action_select_empty(self) -> None:
        picker = self.query_one("#picker", SelectionList)
        picker.deselect_all()
        for s in self.rows:
            if s.searchable and s.leader is None and not s.unwinnable:
                picker.select(s.key)

    def action_target_player(self) -> None:
        self.query_one("#target-player", Select).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "target-player" or not isinstance(event.value, str):
            return
        picker = self.query_one("#picker", SelectionList)
        picker.deselect_all()
        chosen = [s for s in self.rows if self.targetable(s) and s.leader.name == event.value]
        for s in chosen:
            picker.select(s.key)
        self.notify(
            f"{len(chosen)} board{'' if len(chosen) == 1 else 's'} where {event.value} leads {self.game_mode.label}"
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_go(self) -> None:
        chosen = set(self.query_one("#picker", SelectionList).selected)
        if not chosen:
            self.notify("select at least one question", severity="warning")
            return
        order = [s.key for s in sorted(self.rows, key=lambda s: s.search_order) if s.key in chosen]
        budget = self.read_int("#budget", "oracle calls per question", 1)
        stall = self.read_int("#stall", "patience", 1)
        if budget is None or stall is None:
            return
        win_extra = 0
        if self.query_one("#win-mode", Switch).value:
            win_extra = self.read_int("#win-extra", "calls after lead", 1)
            if win_extra is None:
                return
        max_level = self.query_one("#max-level", Select).value
        use_llm = self.query_one("#llm", Switch).value
        escalate = self.query_one("#escalate", Switch).value
        self.app.switch_screen(
            LabScreen(
                order,
                budget=budget,
                use_llm=use_llm,
                max_stall=stall,
                escalate=escalate,
                max_level=max_level,
                board=self.game_mode.board,
                win_extra=win_extra,
            )
        )

    def read_int(self, selector: str, name: str, minimum: int) -> int | None:
        field = self.query_one(selector, Input)
        try:
            value = int(field.value)
        except ValueError:
            value = None
        if value is None or value < minimum:
            self.notify(f"{name} must be a whole number of at least {minimum}", severity="error")
            field.focus()
            return None
        return value


LONG_REROLLS = 5


class PublishScreen(Screen):
    CSS = """
    #pub-help { height: auto; padding: 0 1; }
    #pub-picker { height: 1fr; border: round $primary; }
    #pub-controls, #pub-import { height: 3; }
    #pub-controls Label, #pub-import Label { padding: 1 1 0 1; }
    #pub-controls Input { width: 8; }
    #pub-controls Button, #pub-import Button { margin: 0 1; }
    #import-source { width: 22; }
    #pub-log { height: 14; border: round $secondary; }
    """

    BINDINGS = [
        Binding("escape", "back", "back"),
    ]

    def __init__(self, game_mode: GameMode):
        super().__init__()
        self.game_mode = game_mode
        self.rows: list[Standing] = []
        self.long_rows: list[Standing] = []
        self.golf_picks: list = []
        self.busy = False
        self.stop_requested = False

    def compose(self) -> ComposeResult:
        yield Static(id="pub-help")
        yield SelectionList[str](id="pub-picker")
        with Horizontal(id="pub-controls"):
            yield Label("dry run")
            yield Switch(False, id="dry")
            yield Label("near-misses")
            yield Switch(False, id="long")
            yield Label("re-rolls")
            yield Input("3", id="rerolls", type="integer")
            yield Button("Refresh", id="refresh")
            yield Button("Publish selected", id="publish", variant="success")
            yield Button("Stop", id="stop", disabled=True)
            yield Button("Back", id="back")
        with Horizontal(id="pub-import"):
            sources = [(edition.capitalize(), edition) for edition in EDITIONS if edition != EDITION]
            yield Label("import from")
            yield Select(sources, prompt="edition", id="import-source")
            yield Button("Import", id="import-go")
        yield SelectableRichLog(id="pub-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(
            "#pub-picker"
        ).border_title = f"{self.game_mode.label}: vault lines estimated to beat the current leader"
        self.query_one("#pub-log").border_title = "publisher"
        self.reload()

    def reload(self) -> None:
        with_long = self.query_one("#long", Switch).value
        u = self.game_mode.unit_abbr
        every = standings(
            DB(), mode=self.game_mode.play_mode, long_shots=with_long, board=self.game_mode.board, targets=True
        )
        self.rows = sorted((s for s in every if s.entry_beats), key=lambda s: s.search_order)
        self.long_rows = sorted((s for s in every if s.long_shot), key=lambda s: s.search_order)
        from ..crosspost import picks as golf_picks

        self.golf_picks = golf_picks(DB())
        picker = self.query_one("#pub-picker", SelectionList)
        picker.clear_options()
        for s in self.rows:
            e = s.best_entry
            prompt = Text()
            prompt.append(f"est {e['p_mean']:.3f} (lcb {e['p_lcb']:.3f}) {e['units']:>2}{u}  ", style="bold green")
            prompt.append(f"vs lead {fmt_leader(s, u)}  ", style="yellow")
            if e.get("estimated_from"):
                prompt.append(f"[est({e['estimated_from']})] ", style="bold cyan")
            if e.get("gamble"):
                prompt.append(f"[BET {e.get('hits')} rolls] ", style="bold magenta")
            prompt.append(f"[{e['status']}] ", style="dim")
            prompt.append(f"{s.label[:48]:<48} ", style="bold")
            prompt.append(e["phrase"][:90])
            picker.add_option(Selection(prompt, s.key, True))
        for s in self.long_rows:
            e = s.long_shot
            prompt = Text()
            prompt.append("[long shot] ", style="bold magenta")
            prompt.append(
                f"est {e['p_mean']:.3f} (lcb {e['p_lcb']:.3f}) {e['units']:>2}w n={e['n']}  ", style="magenta"
            )
            prompt.append(f"vs lead {fmt_leader(s)}  ", style="yellow")
            prompt.append(f"{s.label[:48]:<48} ", style="bold")
            prompt.append(e["phrase"][:90])
            picker.add_option(Selection(prompt, f"long:{s.key}", False))
        for index, pick in enumerate(self.golf_picks):
            entry = pick.entry
            prompt = Text()
            prompt.append(f"GOLF {pick.mode.label} ", style="bold magenta")
            prompt.append(
                f"{entry['p_mean']:.3f}/{pick.units}c  ",
                style="bold magenta",
            )
            lead = "empty" if pick.leader is None else f"{pick.leader.probability:.2f}/{pick.leader.units}c"
            prompt.append(f"vs {lead}  ", style="yellow")
            prompt.append(f"{pick.slug[:40]:<40} ", style="bold")
            prompt.append(entry["phrase"][:80])
            picker.add_option(Selection(prompt, f"golf:{index}", True))
        help_text = Text()
        help_text.append(f"Publish {self.game_mode.label}  ", style="bold")
        if self.rows:
            help_text.append(
                f"{len(self.rows)} boards where our best vault line should take the top spot, including "
                "empty boards. All are preselected. Borrowed lines are posted without an oracle check. "
                "Lines this edition measured are still re-checked."
            )
        else:
            help_text.append(
                "Nothing ready on this board: no vault line beats the leader or takes an empty board.",
                style="yellow",
            )
        help_text.append(
            f"\n{len(self.golf_picks)} Golf lines the Strict vault would crown, on both Golf boards, "
            "including when you already lead Strict. Uncheck one to hold it this run.",
            style="magenta",
        )
        if with_long:
            help_text.append(
                f"\n{len(self.long_rows)} long shots: lines whose average, but not lower bound, beats "
                f"the leader. Unselected; each skips the oracle re-check and is re-rolled "
                f"the requested number of times, up to {LONG_REROLLS} while still short. "
                "A miss costs only turns, since the board keeps our best.",
                style="magenta",
            )
        self.query_one("#pub-help", Static).update(help_text)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "long" and not self.busy:
            self.reload()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "refresh":
            self.reload()
        elif event.button.id == "import-go":
            self.import_source()
        elif event.button.id == "publish":
            self.start_publish()
        elif event.button.id == "stop":
            self.stop_requested = True
            self.notify("stopping after the current phrase")

    def import_source(self) -> None:
        from ..transfer import import_from

        source = self.query_one("#import-source", Select).value
        log = self.query_one("#pub-log", RichLog)
        if not isinstance(source, str) or source not in EDITIONS:
            self.notify("pick an edition to import from", severity="warning")
            return
        try:
            import_from(DB(), source, log=log.write)
        except RuntimeError as error:
            log.write(str(error))
        self.reload()

    def action_back(self) -> None:
        if self.busy:
            self.notify("publisher is still running", severity="warning")
            return
        self.app.pop_screen()

    def start_publish(self) -> None:
        if self.busy:
            return
        chosen = set(self.query_one("#pub-picker", SelectionList).selected)
        board = self.game_mode.board
        entries = []
        for s in self.rows:
            if s.key in chosen:
                e = s.best_entry
                vault.set_status(s.slug, e["mode"], e["phrase"], "queued", board, target=s.target)
                entries.append(e | {"status": "queued", "board": board, "target": s.target})
        for s in self.long_rows:
            if f"long:{s.key}" in chosen:
                e = s.long_shot
                vault.save(
                    s.slug,
                    e["mode"],
                    e["phrase"],
                    p_mean=e["p_mean"],
                    p_lcb=e["p_lcb"],
                    spread=e["spread"],
                    n=e["n"],
                    units=e["units"],
                    leader=s.leader,
                    beats=False,
                    title=s.title,
                    origin="long shot",
                    status="queued",
                    note="long shot",
                    board=board,
                    estimated_from=e.get("estimated_from") or "",
                    jev_p=e.get("jev_p"),
                )
                vault.set_status(s.slug, e["mode"], e["phrase"], "queued", board, target=s.target)
                entries.append(e | {"status": "queued", "board": board, "target": s.target})
        from ..crosspost import queue as queue_golf

        for index, pick in enumerate(self.golf_picks):
            if f"golf:{index}" not in chosen:
                continue
            queue_golf(DB(), [pick])
            entry = pick.entry
            entries.append(
                {
                    "slug": pick.slug,
                    "mode": GOLF,
                    "phrase": entry["phrase"],
                    "p_mean": entry["p_mean"],
                    "p_lcb": entry["p_lcb"],
                    "spread": entry.get("spread", 0.0),
                    "n": entry.get("n", 0),
                    "units": pick.units,
                    "board": pick.mode.board,
                    "target": pick.target,
                    "status": "queued",
                    "estimated_from": entry.get("estimated_from") or "",
                }
            )
        if not entries:
            self.notify("nothing selected", severity="warning")
            return
        try:
            rerolls = int(self.query_one("#rerolls", Input).value or 3)
        except ValueError:
            rerolls = 3
        dry = self.query_one("#dry", Switch).value
        self.busy = True
        self.stop_requested = False
        self.query_one("#publish", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#import-go", Button).disabled = True
        self.query_one("#pub-log", RichLog).write(
            f"{'DRY RUN: ' if dry else ''}publishing {len(entries)} line(s), re-rolls {rerolls}"
        )
        self.publish_worker(entries, dry, rerolls)

    @work(thread=True, exclusive=True)
    def publish_worker(self, entries: list[dict], dry: bool, rerolls: int) -> None:
        from ..publish import publish

        log = self.query_one("#pub-log", RichLog)
        results = []

        def write(message: str) -> None:
            for line in str(message).splitlines():
                if line.strip():
                    activity.log("publish", line)
            self.app.call_from_thread(log.write, message)

        def on_result(entry: dict, result: str) -> None:
            results.append((entry, result))

        try:
            publish(
                dry_run=dry,
                rerolls=rerolls,
                entries=entries,
                log=write,
                on_result=on_result,
                long_rerolls=LONG_REROLLS,
                halt=lambda: self.stop_requested,
            )
        except Exception as error:
            write(f"publisher crashed: {error!r}")
        won = sum(1 for _, r in results if r.startswith("won") or r == "already")
        write(f"\nfinished: {won}/{len(results)} on top" + (" (dry run, nothing sent)" if dry else ""))
        if dry:
            for s_entry in entries:
                vault.set_status(
                    s_entry["slug"],
                    s_entry["mode"],
                    s_entry["phrase"],
                    "candidate",
                    s_entry["board"],
                    target=s_entry.get("target") or "",
                )

        def done() -> None:
            self.busy = False
            self.stop_requested = False
            self.query_one("#publish", Button).disabled = False
            self.query_one("#stop", Button).disabled = True
            self.query_one("#import-go", Button).disabled = False
            self.reload()

        self.app.call_from_thread(done)
