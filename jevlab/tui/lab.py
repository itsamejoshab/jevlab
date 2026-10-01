"""The live search screen: one or more questions, searched one at a time."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, RichLog, Static

from .. import vault
from ..calibrate import calibration_status
from ..db import DB
from .. import activity, netlog
from ..boards import split_key
from ..live import LiveSession, opening_word, post_phrase, saved_line, search_question
from ..rules.banned import BannedPhrase, contains as phrase_banned
from ..modes import HIGH_SCORES, LIVE, from_board
from ..objective import site_round
from ..config import TIER_LEVEL
from ..search.engine import LEVELS, Engine, Event
from ..search.strategies import LLMGenerate
from ..search.surrogate import Embedder

LEVEL_COLORS = ["green", "cyan", "blue", "magenta", "red"]  # ladder levels L0-L4: label, timeline, calls bar
LEVEL_STYLES = [f"bold {c}" for c in LEVEL_COLORS]
PLAN_COLOURS = {"explore": "green", "exploit": "yellow", "compress": "cyan"}
FEED_STYLES = {"precision": "bold cyan", "sweep": "bold blue", "boosters": "bold magenta", "beam": "bold red"}


def strategy_style(name: str) -> str:
    if name.startswith("llm"):
        return "bold green"
    return FEED_STYLES.get(name, "bold")


STRATEGY_INFO = {
    "llm": "Generator models write whole new lines from the question's archetypes, shown our best-scored lines so "
    "far so they can build on what works. Each batch goes to the next model in this price tier; the pricier "
    "tiers join as the ladder escalates (mid at L2, premium at L4).",
    "chain": "Long chains: assembles very long lines out of fragments of our strong chains using the {recipe} "
    "recipe, screens them with a probe claim that tells apart lines rounding to the same score, and sends "
    "only the most promising to the oracle.",
    "chain_ablate": "Drops each fragment of the best chains once. The score lost is that fragment's measured "
    "worth, which tells the chain builders and the compactor what to keep.",
    "chain_compact": "Cuts a chain that holds the ceiling (or already wins) down while it keeps that standing: whole "
    "fragments first, weakest first, then windows of 32, 16, 8, 4 and 2 words.",
    "local_edit": "Single edits around a top line: substitute, delete or insert a word, swap two words, or move a "
    "clause. Cheap and steady once a good line exists.",
    "compress": "Removes each word of the best line in turn to measure how much it matters (the word colours "
    "above), then deletes as many words as it can while the rounded score holds. Only runs when shorter "
    "helps: level with the leader's score, or longer than their line.",
    "genetic": "Genetic algorithm seeded from diverse good lines: picks parents by tournament, splices them at "
    "clause boundaries, mutates the children, and now and then has an LLM rewrite one.",
    "surrogate_bo": "Generates thousands of edits locally and asks the surrogate (a small model trained on our own "
    "scores) which look best, with a bonus for the ones it is unsure about. Only the top picks cost "
    "oracle calls.",
    "gcg": "Uses the word surrogate's gradient to guess which single-word swaps raise the score most (HotFlip "
    "style), then checks the best guesses with the oracle.",
    "precision": "L1 averaging. The oracle rounds to 0.01 but its noise wobbles, so the mean of several samples "
    "tells lines apart more finely than the rounding. Hill-climbs on those means, keeping a step only "
    "when it beats the noise.",
    "sweep": "L2 sweep. Tries every word slot and every gap against a large vocabulary, then pairs of the best "
    "single edits. Stops at a line that no single change improves.",
    "boosters": "L3 boosters. Short word sequences that raised the score across other questions, applied to our top "
    "lines as a prefix, a suffix or an inserted clause.",
    "beam": "L4 reframe. Builds lines from nothing one word at a time, keeping the best few at each length. Every "
    "prefix is itself a scoreable line, so it finds framings the edit strategies can't reach.",
    "single_word": "Shortest yes: most boards are led by one word, so this scores single words and pairs directly: "
    "LLM ideas, casings and pairs of our best singles, other players' words, then the next slice of "
    "the big vocabulary.",
    "extend": "Plateau mode. Grows the best line toward the leader's length a clause at a time, using pieces of our "
    "other strong lines, boosters and a cheap model's supporting clauses. Lines that tie are raced on "
    "averaged scores.",
    "grow": "Plateau mode. Builds a fresh line one appended word at a time, up to one word under the leader's "
    "length, with a narrow beam. The beam is saved, so each run keeps growing the same lines.",
    "probe_climb": "Plateau mode, for boards where every good line reads the same rounded score. Appends a short "
    "claim for the other answer to each line, which pulls it down to mid-range where lines nearer "
    "the next level resist more, then climbs on that hidden ranking: edits that hold the level are "
    "probed once, the best re-probed, and the walk moves when a step clears the noise.",
    "scenario_join": "Plateau mode. A writer model sets high-stakes scenes where the wrong answer would be absurd, "
    "and each scene is joined before and after our best tied lines. This is what broke two hard "
    "boards past 0.98. The scenes whose joins do best steer the next batch; after two rounds with "
    "nothing above the level it steps aside.",
    "drift": "Plateau mode, for lines stuck at the top rounded score. Walks toward the line whose edits most often "
    "keep that score, since such lines were far likelier to have an edit that scores 0.01 higher. Its "
    "position is saved across runs.",
}


def strategy_info(name: str) -> str:
    if name.startswith("llm"):
        return STRATEGY_INFO["llm"]
    if name.startswith("chain:"):
        return STRATEGY_INFO["chain"].format(recipe=name.split(":", 1)[1].replace("_", " "))
    return STRATEGY_INFO.get(name, "No description yet.")


@dataclass
class QueueItem:
    slug: str
    title: str
    target: str = ""  # one answer of a choice question
    state: str = "pending"  # pending | running | done | skipped | error
    leader: str = "-"
    best: str = "-"
    win: bool = False
    reason: str = ""
    calls: int = 0


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


class LabScreen(Screen):
    CSS = """
    LabScreen { layout: vertical; }
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
        Binding("space", "toggle_pause", "pause/resume"),
        Binding("n", "next_question", "next question"),
        Binding("c", "compress", "compress row"),
        Binding("r", "resample", "resample row"),
        Binding("s", "save", "save row"),
        Binding("v", "queue_row", "queue row"),
        Binding("p", "toggle_post", "post on/off"),
        Binding("q", "leave", "stop & back"),
    ]

    def __init__(
        self,
        slugs: list[str],
        budget: int = 3000,
        use_llm: bool = True,
        max_stall: int = 12,
        single: bool = False,
        seed: int | None = None,
        escalate: bool = True,
        max_level: int = 4,
        board: str = HIGH_SCORES,
        win_extra: int = 0,
        question: dict | None = None,
        live_play: bool = False,
        live_pace: str = "slow",
        live_chain: str = "casual",
        live_post: bool = True,
    ):
        super().__init__()
        self.db = DB()
        self.win_extra = win_extra
        self.single = single
        self.board = board
        self.live_play = live_play or bool(
            question and question.get("raw", {}).get("live") and question.get("revision_id")
        )
        self.live_pace = "fast" if live_pace == "fast" else "slow"
        self.live_chain = "strict" if live_chain == "strict" else "casual"
        self.live_post = live_post
        self.live_question = question
        self.live_session: LiveSession | None = None
        self.live_client = None
        self.posting = False
        self.skipped_phrases: set[str] = set()
        self.live_gen = 0
        self.opened_round = ""
        self.gap_logged = False
        self.game_mode = (
            LIVE if self.live_play or (question and question.get("raw", {}).get("live")) else from_board(board)
        )
        self.budget = budget
        self.use_llm = use_llm
        self.max_stall = 0 if single or self.live_play else max_stall
        self.seed = seed
        self.escalate = escalate
        self.max_level = max_level
        self.scrolled_to = -1
        self.level_marks: list[tuple[int, int]] = []  # (oracle calls, ladder level) at each level change
        self.win_marks: list[int] = []  # oracle calls when a line beating the leader was saved
        titles = {r["slug"]: r["title"] for r in self.db.all("SELECT slug, title FROM questions")}
        self.queue = []
        if self.live_play and not slugs:
            self.queue.append(QueueItem("live", "waiting for live round"))
        for key in slugs:
            slug, target = split_key(key)
            title = (question or {}).get("title") or titles.get(slug, slug)
            self.queue.append(QueueItem(slug, f"{title} -> {target}" if target else title, target))
        self.engine: Engine | None = None
        self.embedder: Embedder | None = None
        self.index = -1
        self.status = "starting"
        self.current = ""
        self.surrogate = "untrained"
        self.plan = ""
        self.plan_mode = ""
        self.plan_focus: list[str] = []
        self.plan_bans: list[str] = []
        self.plan_at = 0.0
        self.plans = 0
        self.calibration = calibration_status()
        self.recent_p: deque[float] = deque(maxlen=800)
        self.timeline: deque[Text] = deque(maxlen=300)  # newest first
        self.timeline_dirty = False
        self.strategy_top: dict[str, float] = {}
        self.strategy_hits: dict[str, int] = {}
        self.bests = 0
        self.last_best_calls = 0
        self.last_best_time = 0.0
        self.batches = 0
        self.dirty = True
        self.cancelled = False
        self.started_at = time.monotonic()

    def compose(self) -> ComposeResult:
        yield Static(id="lab-header")
        with Horizontal(id="body"):
            with VerticalScroll(id="queue-box"):
                yield Static(id="queue")
            with Vertical(id="middle"):
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="feed", wrap=False, markup=False, max_lines=600)
            with Vertical(id="side"):
                yield RichLog(id="timeline", wrap=True, markup=False, max_lines=400)
                yield Static(id="insight")
        yield RichLog(id="lab-log", wrap=True, markup=False, max_lines=1000)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.add_columns("#", "p", "±", "n", "w", "origin", "", "phrase")
        table.border_title = "candidates in board order   WIN = lower bound beats leader   * = in vault"
        if self.game_mode.shortest:
            table.border_title += "   BET k/n = shorter than leader, k rolls cleared the yes line (re-roll on publish)"
        if self.live_play:
            self.query_one("#queue-box").border_title = "Live Mode: /play?round=live"
            self.query_one("#queue-box").border_subtitle = "question · live leader · our best"
        else:
            self.query_one("#queue-box").border_title = f"{self.game_mode.label} queue (" + (
                "longest leader first)" if self.game_mode.shortest else "weakest leader first)"
            )
            self.query_one("#queue-box").border_subtitle = "question · leader · our best"
        self.query_one("#timeline").border_title = "new bests"
        self.query_one("#feed").border_title = "now trying"
        self.query_one("#insight").border_title = "what's working"
        self.query_one("#lab-log").border_title = "log"
        self.set_interval(1.0, self.refresh_view)
        self.run_worker(self.run_queue(), exclusive=True, name="queue")

    # Queue runner.

    async def run_queue(self) -> None:
        self.log_line(f"loading embedder and starting {len(self.queue)} question(s)")
        self.embedder = await asyncio.to_thread(Embedder)
        if self.live_play:
            await self.run_live()
            return
        for index, item in enumerate(self.queue):
            if self.cancelled:
                item.state = "skipped"
                continue
            self.index = index
            item.state = "running"
            self.recent_p.clear()
            self.level_marks = [(0, 0)]
            self.win_marks = []
            self.strategy_top.clear()
            self.strategy_hits.clear()
            self.bests = 0
            self.last_best_calls = 0
            self.last_best_time = time.monotonic()
            self.plan, self.plan_mode, self.plan_focus, self.plan_bans, self.plans = "", "", [], [], 0
            self.query_one("#feed", RichLog).clear()
            self.timeline.clear()
            self.timeline_dirty = True
            try:
                engine = Engine(
                    self.db,
                    item.slug,
                    budget=self.budget,
                    use_llm=self.use_llm,
                    seed=self.seed,
                    on_event=self.on_engine_event,
                    idle_when_done=self.single,
                    embedder=self.embedder,
                    max_stall=self.max_stall,
                    escalate=self.escalate,
                    max_level=self.max_level,
                    board=self.board,
                    target=item.target,
                    win_extra=self.win_extra,
                    question=self.live_question,
                )
            except ValueError as error:
                item.state, item.reason = "error", str(error)
                self.log_line(f"skip {item.slug}: {error}")
                continue
            self.engine = engine
            self.show_yes_line()
            lead = engine.leader
            item.leader = f"{lead.probability:.2f}/{lead.units}w" if lead else "empty"
            self.log_line(f"== [{index + 1}/{len(self.queue)}] {item.title}  leader {item.leader}")
            await engine.run()
            self.update_item(item, engine)
            item.reason = engine.end_reason
            item.state = "done"
            self.log_line(f"done {item.title}: best {item.best} {'WIN' if item.win else 'no win'} ({item.reason})")
            self.dirty = True
        self.status = "all done"
        wins = sum(1 for q in self.queue if q.win)
        self.log_line(
            f"queue finished: {wins}/{len(self.queue)} boards with a winning line. "
            "Publish from the home screen.  q = back"
        )

    async def run_live(self) -> None:
        from ..site.client import SiteClient

        self.log_line(
            f"Live Mode ({self.live_pace}, {self.live_chain}): watching /play?round=live. "
            + (
                "Posting is off; the search still runs. p toggles it."
                if not self.live_post
                else (
                    "Fire a word immediately, then repost on every score gain. p toggles posting."
                    if self.live_pace == "fast"
                    else "Post when not 1st, then hold. p toggles posting."
                )
            )
        )
        self.live_session = LiveSession(pace=self.live_pace, chain=self.live_chain, post=self.live_post)
        client = SiteClient()
        self.live_client = client
        me = self.db.me()
        try:
            viewer = await asyncio.to_thread(client.viewer)
            me = ((viewer.get("player") or {}).get("user") or {}).get("id") or me
        except Exception as error:
            self.log_line(f"viewer failed ({error}); using snapshot identity {me or 'anonymous'}")
        item = self.queue[0] if self.queue else QueueItem("live", "waiting for live round")
        if not self.queue:
            self.queue.append(item)
        while not self.cancelled:
            try:
                state = await asyncio.to_thread(client.round_state, "live")
            except Exception as error:
                self.log_line(f"live poll failed: {error}")
                await asyncio.sleep(self.live_session.poll_interval())
                continue
            self.live_session.observe(state, me=me)
            session = self.live_session
            question = search_question(state, session.question)
            if question is None:
                item.title = f"no live round ({session.status or 'idle'})"
                item.state = "pending"
                item.leader = "-"
                self.status = "waiting"
                self.dirty = True
                self.gap_logged = False
                if self.engine:
                    self.engine.stop()
                    self.engine = None
                await asyncio.sleep(session.poll_interval())
                continue
            item.title = question["title"]
            live_now = session.status == "live"
            lead = session.leader
            if live_now and session.we_lead:
                item.leader = "we lead"
                item.win = True
            elif live_now:
                item.leader = f"{lead.probability:.2f}/{lead.units}w {lead.name}" if lead else "empty"
                item.win = False
            else:
                item.leader = "between rounds"
                item.win = False
            self.dirty = True
            same = self.engine is not None and self.engine.slug == question["slug"]
            if not same:
                if self.engine:
                    self.engine.stop()
                self.reset_live_view()
                self.live_question = question
                self.live_gen += 1
                gen = self.live_gen
                item.state = "running"
                self.index = 0
                where = "live" if live_now else "between rounds"
                self.log_line(
                    f"== {where} [{session.play_mode}] {question['title']}  "
                    f"finish {session.threshold:.2f}  leader {item.leader}"
                )
                try:
                    engine = Engine(
                        self.db,
                        question["slug"],
                        budget=self.budget,
                        use_llm=self.use_llm,
                        seed=self.seed,
                        on_event=self.on_engine_event,
                        idle_when_done=True,
                        embedder=self.embedder,
                        max_stall=0,
                        escalate=self.escalate,
                        max_level=self.max_level,
                        question=question,
                    )
                except ValueError as error:
                    item.state, item.reason = "error", str(error)
                    self.log_line(f"live question failed: {error}")
                    await asyncio.sleep(session.poll_interval())
                    continue
                engine.leader = session.leader
                self.engine = engine
                self.show_yes_line()
                self.run_worker(self._run_live_engine(engine, gen), name=f"live-engine-{gen}")
                self.gap_logged = not live_now
            elif self.engine:
                self.engine.leader = session.leader
            if live_now and session.round_id != self.opened_round:
                self.opened_round = session.round_id
                self.gap_logged = False
                self.arm_live_round()
            elif live_now:
                self.maybe_post_live()
            else:
                self.status = "between rounds"
                if not self.gap_logged:
                    self.gap_logged = True
                    self.log_line(f"[live] between rounds; still searching {question['title']}")
            await asyncio.sleep(session.poll_interval())
        if self.engine:
            self.engine.stop()
        self.status = "all done"
        self.log_line("Live Mode stopped.  q = back")

    def reset_live_view(self) -> None:
        self.recent_p.clear()
        self.level_marks = [(0, 0)]
        self.win_marks = []
        self.strategy_top.clear()
        self.strategy_hits.clear()
        self.bests = 0
        self.last_best_calls = 0
        self.last_best_time = time.monotonic()
        self.plan, self.plan_mode, self.plan_focus, self.plan_bans, self.plans = "", "", [], [], 0
        self.query_one("#feed", RichLog).clear()
        self.timeline.clear()
        self.timeline_dirty = True

    async def _run_live_engine(self, engine: Engine, gen: int) -> None:
        try:
            await engine.run()
        except Exception as error:
            if gen == self.live_gen:
                self.log_line(f"live engine stopped: {error!r}")

    def arm_live_round(self) -> None:
        """Round just went live. Vault line first, then a line this search already found, then one word."""
        self.fire_opening_shot(word=False)
        if not self.posting:
            self.maybe_post_live()
        if not self.posting:
            self.fire_opening_shot(word=True)

    def fire_opening_shot(self, word: bool = True) -> None:
        session = self.live_session
        if self.posting or not session or not session.question:
            return
        saved = saved_line(session.question)
        if (
            saved
            and saved["phrase"] not in self.skipped_phrases
            and not phrase_banned(saved["phrase"])
            and saved["p"] >= session.threshold - 1e-9
            and session.should_post(saved["p"], saved["units"], saved["phrase"])
        ):
            if self.start_live_post(saved["phrase"], saved["p"], saved["units"]):
                self.log_line(f"[live] vault shot {saved['p']:.3f}/{saved['units']}w {saved['phrase']}")
            return
        if not word or session.pace != "fast" or session.posted:
            return
        word = opening_word(session.question)
        if not session.should_post(0.0, 1, word):
            return
        if self.start_live_post(word, 0.0, 1):
            self.log_line(f"[live] opening shot {word}")

    def maybe_post_live(self) -> None:
        session = self.live_session
        engine = self.engine
        if self.posting or not session or not engine or not len(engine.archive):
            return
        cand = next(
            (
                item
                for item in engine.archive.top_by_board(12)
                if item.phrase not in self.skipped_phrases and not phrase_banned(item.phrase)
            ),
            None,
        )
        if cand is None or not session.should_post(cand.p, cand.units, cand.phrase):
            return
        if self.start_live_post(cand.phrase, cand.p, cand.units):
            self.log_line(f"[live] posting {cand.p:.3f}/{cand.units}w {cand.phrase}")

    def start_live_post(self, phrase: str, p: float, units: int) -> bool:
        """Schedule a site post. Never build the coroutine once the screen is leaving or posting is off."""
        if self.cancelled or not self.is_running:
            return False
        if self.live_session and not self.live_session.post_enabled:
            return False
        self.posting = True
        work = self.post_live_worker(phrase, p, units)
        try:
            self.run_worker(work, name="live-post", exit_on_error=False)
        except Exception:
            work.close()
            self.posting = False
            raise
        return True

    async def post_live_worker(self, phrase: str, p: float, units: int) -> None:
        session = self.live_session
        client = self.live_client
        question = session.question if session else None
        try:
            if not session or not client or not question:
                return
            scored = await asyncio.to_thread(
                post_phrase, client, question, session.play_mode, phrase, self.log_line, session.threshold
            )
            if scored >= session.threshold - 1e-9:
                session.finished = True
                session.mark_posted(phrase, p=scored, units=units)
                self.log_line(
                    f"[live] crossed the finish line at {scored:.2f} (finish {session.threshold:.2f}); holding the chain"
                )
            else:
                session.mark_posted(phrase, p=p, units=units)
                self.log_line(f"[live] site scored {scored:.2f} ({p:.3f} oracle / {units}w): {phrase}")
        except BannedPhrase as error:
            self.skipped_phrases.add(phrase)
            if error.named:
                removed = vault.drop_banned()
                self.log_line(
                    f"[live] banned {error.named!r}; removed {len(removed)} vault line(s); its words stay usable"
                )
            else:
                self.log_line("[live] banned, span not named; skipping this line")
        except Exception as error:
            self.log_line(f"[live] post failed: {error}")
        finally:
            self.posting = False
        if self.cancelled:
            return
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            return
        self.maybe_post_live()

    # Engine events: cheap, just record and mark dirty.

    def on_engine_event(self, event: Event) -> None:
        d = event.data
        kind = event.kind
        if kind == "log":
            self.log_line(d["message"])
        elif kind == "scored":
            self.on_scored(d)
        elif kind == "best":
            self.on_best(d)
        elif kind == "round":
            self.current = d["strategy"]
        elif kind == "vault":
            tag = (f"BET {d['gamble']} rolls" if d.get("gamble") else "WIN") if d["beats"] else "saved"
            if d["beats"] and self.engine and self.engine.oracle:
                self.win_marks.append(self.engine.oracle.calls)
            self.log_line(f"[vault {tag}] {d['p']:.3f} (lcb {d['lcb']:.3f}) {d['units']}w {d['phrase']}")
        elif kind == "plan":
            self.plan = " ".join(str(d["directive"]).split())
            self.plan_mode = d.get("mode") or ""
            self.plan_focus = list(d.get("focus") or [])
            self.plan_bans = list(d.get("ban") or [])
            self.plan_at = time.monotonic()
            self.plans += 1
            self.log_line(f"[planner] {self.plan_mode or '-'}: {self.plan}")
        elif kind == "surrogate":
            self.surrogate = (
                f"{d['embedder']} ρ {d['phrase_rho']:.2f} phrase / {d['word_rho']:.2f} word, {d['n']:,} lines"
            )
        elif kind == "status":
            self.status = d["status"]
        elif kind == "level":
            oracle = self.engine.oracle if self.engine else None
            self.level_marks.append((oracle.calls if oracle else 0, d["level"]))
            line = Text(f"{time.strftime('%H:%M:%S')} ")
            line.append(f"── L{d['level']} {d['name']} ──", style=LEVEL_STYLES[min(d["level"], len(LEVEL_STYLES) - 1)])
            line.append(f" {d['reason']}", style="dim")
            self.timeline.appendleft(line)
            self.timeline_dirty = True
            activity.log("level", f"L{d['level']} {d['name']}: {d['reason']}")
        self.dirty = True
        if self.live_play and kind in ("scored", "best", "vault"):
            self.maybe_post_live()

    def show_yes_line(self) -> None:
        if not self.game_mode.shortest or not self.engine:
            return
        bar = f"{self.engine.objective.threshold:.2f}"
        self.query_one("#table", DataTable).border_title = (
            "candidates in board order   WIN = lower bound beats leader   * = in vault"
            f"   BET k/n = shorter than leader, k rolls cleared {bar} (re-roll on publish)"
        )

    def target(self) -> float:
        """The score that counts: the leader's rounded score, or on Shortest yes the yes threshold."""
        if self.engine and self.engine.objective.shortest:
            return self.engine.objective.threshold
        lead = self.engine.leader if self.engine else None
        return site_round(lead.probability) if lead else 0.0

    def on_scored(self, d: dict) -> None:
        items = d.get("items") or []
        if not items:
            return
        self.batches += 1
        items = [it for it in items if it[1] == it[1]]  # NaN != NaN
        if not items:
            return
        fresh = [it for it in items if it[3] <= 1]
        for it in fresh:
            self.recent_p.append(it[1])
        target = self.target()
        feed = self.query_one("#feed", RichLog)
        ps = [it[1] for it in items]
        head = Text()
        strategy = d.get("strategy") or "-"
        head.append(f"{time.strftime('%H:%M:%S')} ", style="bold")
        head.append(f"{strategy[:16]:<16} ", style=strategy_style(strategy))
        head.append(f"{len(items):4} scored  best ")
        head.append(f"{max(ps):.3f}", style=p_style(max(ps), target))
        head.append(f"  mean {sum(ps) / len(ps):.3f}")
        hits = sum(1 for p in ps if target and site_round(p) >= target)
        self.strategy_top[strategy] = max(self.strategy_top.get(strategy, 0.0), max(ps))
        self.strategy_hits[strategy] = self.strategy_hits.get(strategy, 0) + hits
        if hits:
            label = "qualify" if self.game_mode.shortest else "at leader score"
            head.append(f"  {hits} {label}", style="bold green")
        feed.write(head)
        activity.log(
            "batch",
            f"{strategy} {len(items)} scored best {max(ps):.3f} mean {sum(ps) / len(ps):.3f}"
            + (f" {hits} at target" if hits else ""),
        )
        ranked = sorted(items, key=lambda it: -it[1])
        shown = ranked[:3]
        rest = ranked[3:]
        if rest:
            step = max(1, len(rest) // 3)
            shown += rest[::step][:3]
        for phrase, p, origin, _n in shown:
            line = Text("   ")
            line.append(f"{p:.3f}", style=p_style(p, target))
            line.append(f" {len(phrase.split()):2}w ", style="dim")
            line.append(f"{origin[:12]:<12} ", style="cyan")
            line.append(phrase[:110])
            feed.write(line)

    def on_best(self, d: dict) -> None:
        activity.log("best", f"{d['p']:.3f} {d['units']}w [{d['origin']}] {d['phrase']}")
        line = Text(f"{time.strftime('%H:%M:%S')} ")
        line.append(f"{d['p']:.3f}", style=p_style(d["p"], self.target()))
        line.append(f" {d['units']}w ", style="bold")
        line.append(f"[{d['origin']}] ", style="cyan")
        line.append(d["phrase"])
        self.timeline.appendleft(line)
        self.timeline_dirty = True
        self.bests += 1
        self.last_best_calls = self.engine.oracle.calls if self.engine and self.engine.oracle else 0
        self.last_best_time = time.monotonic()

    def log_line(self, message: str) -> None:
        activity.log("lab", message)
        try:
            self.query_one("#lab-log", RichLog).write(f"{time.strftime('%H:%M:%S')} {message}")
        except Exception:
            pass

    # Rendering.

    def refresh_view(self) -> None:
        self.render_header()
        self.render_queue()
        if self.timeline_dirty:
            self.timeline_dirty = False
            timeline = self.query_one("#timeline", RichLog)
            timeline.clear()
            for line in self.timeline:
                timeline.write(line, scroll_end=False)
            timeline.scroll_home(animate=False)
        if not self.dirty or self.engine is None:
            return
        self.dirty = False
        self.render_table()
        self.render_insight()

    def render_header(self) -> None:
        engine = self.engine
        header = Text()
        if engine is None:
            header.append(f"{self.status}…")
            self.query_one("#lab-header", Static).update(header)
            return
        q = engine.question
        lead = engine.leader
        header.append(f"{self.game_mode.label}  ", style="bold reverse")
        header.append(f"[{self.index + 1}/{len(self.queue)}] ", style="bold magenta")
        header.append(q["title"], style="bold")
        header.append("   goal ", style="dim")
        header.append(f"{engine.target or engine.objective.goal}   ", style="bold cyan" if engine.target else "")
        header.append(
            f"leader {lead.probability:.2f}/{lead.units}w {lead.name}" if lead else "board empty", style="bold yellow"
        )
        if engine.our_best:
            header.append(f"   ours on board {engine.our_best.probability:.2f}/{engine.our_best.units}w")
        if len(engine.archive):
            best = engine.archive.top_by_board(1)[0]
            wins = engine.objective.wins(best.score, best.units, lead)
            header.append("   best found ")
            header.append(f"{best.p:.3f}/{best.units}w", style="bold green" if wins else "bold")
            if wins:
                header.append(" WIN", style="bold green")
        if getattr(engine, "plateau", False):
            root = engine.ctx.climb_roots(1)[0] if len(engine.archive) else None
            header.append(
                "   plateau: climbing on dithered means"
                + (f" (best {root.p:.4f} n={root.score.n} {root.units}w)" if root else ""),
                style="magenta",
            )
            grow, drift = engine.strategies["grow"], engine.strategies["drift"]
            if grow.beam and not isinstance(grow.beam[0], str):
                header.append(f"   grow {grow.beam[0].p:.3f}/{grow.depth}w", style="magenta")
            walk = engine.archive.items.get(drift.line) if drift.line else None
            if walk is not None:
                rate, n = drift.hold_rate(engine.ctx, walk.phrase, site_round(walk.p))
                header.append(f"   drift holds {rate:.2f} over {n} edits ({walk.units}w)", style="magenta")
        if engine.long:
            phase = engine.phase()
            header.append(
                f"   long chains: {phase.upper()} to {engine.objective.ceiling:.2f}",
                style="bold magenta" if phase == "compact" else "magenta",
            )
            mark = engine.long_mark()
            if phase == "build" and mark[2]:
                header.append(f"  probe {mark[2]:.3f}", style="magenta")
        width = self.query_one("#lab-header", Static).content_size.width or 120
        header.truncate(width, overflow="ellipsis")
        header.append("\n")
        bar, overflow = self.calls_bar(engine)
        header.append_text(bar)
        header.append("\n")
        header.append_text(self.models_line(engine, overflow, width))
        header.append("\n")
        header.append_text(self.planner_line(width * 2))
        self.query_one("#lab-header", Static).update(header)

    def models_line(self, engine: Engine, overflow: list[tuple[str, str]], width: int) -> Text:
        """Run stats that did not fit on the bar line, then what the search is steering by."""
        parts = [
            *overflow,
            ("calibration", self.calibration.removeprefix("cal ")),
            ("surrogate", self.surrogate),
            ("bans", f"{len(engine.ctx.banned)}"),
            ("pins", ", ".join(sorted(engine.ctx.pinned)) or "none"),
        ]
        line = Text(no_wrap=True, overflow="ellipsis")
        for i, (label, value) in enumerate(parts):
            if i:
                line.append("  │  ", style="grey42")
            line.append(f"{label} ", style="dim")
            line.append(value)
        line.truncate(width, overflow="ellipsis")
        return line

    def planner_line(self, room: int) -> Text:
        """The planner's latest directive: a badge with its mode and age, then the directive over two lines."""
        line = Text()
        if not self.plan:
            line.append(" PLANNER ", style="bold black on grey62")
            line.append(
                "  waiting for its first read of the search (runs once there are 5+ scored lines)", style="dim italic"
            )
            return line
        colour = PLAN_COLOURS.get(self.plan_mode, "white")
        line.append(" PLANNER ", style="bold black on grey62")
        line.append(f" {(self.plan_mode or 'note').upper()} ", style=f"bold black on {colour}")
        age = int(time.monotonic() - self.plan_at)
        line.append(f" #{self.plans} · {age // 60}m{age % 60:02d}s ago  ", style="dim")
        tail = Text()
        if self.plan_focus:
            tail.append("  focus ", style="dim")
            tail.append(", ".join(self.plan_focus), style=colour)
        if self.plan_bans:
            tail.append("  banned ", style="dim")
            tail.append(", ".join(self.plan_bans), style="red")
        space = max(room - len(line) - len(tail), 20)
        directive = self.plan if len(self.plan) <= space else self.plan[: space - 1].rstrip() + "…"
        line.append(directive, style=f"italic {colour}")
        line.append_text(tail)
        return line

    def calls_bar(self, engine: Engine) -> tuple[Text, list[tuple[str, str]]]:
        """A full-width bar of oracle calls against the budget in the ladder level's colour, with the run stats
        on the line below it. Stats that do not fit are returned to go on the next line."""
        oracle = engine.oracle
        calls = oracle.calls if oracle else 0
        elapsed = int(time.monotonic() - self.started_at)
        stall = f"{engine.stall}/{engine.patience()}" if engine.escalate or engine.max_stall else str(engine.stall)
        inside = [
            f" {self.status.upper()} ",
            f"run {engine.memory.runs + 1}" + (" · FRESH START" if engine.fresh else ""),
            self.level_label(engine),
            f"calls {calls:,}/{engine.budget:,} ({calls / max(engine.budget, 1):.0%})",
            f"round {engine.rounds} [{self.current}]",
            f"stall {stall}",
            f"{elapsed // 60}m{elapsed % 60:02d}s",
        ]
        if self.live_play and self.live_session and not self.live_session.post_enabled:
            inside.append("POST OFF")
        if netlog.counts:
            worst = [
                f"{svc} {status.replace('HTTP ', '')}x{n} {where.split('.')[0]}"
                for (svc, where, status), n in netlog.counts.most_common(2)
            ]
            conc = f" conc {int(oracle.limiter.limit)}" if oracle else ""
            inside.append("NET ERR " + ", ".join(worst) + conc)
        if engine.win_extra:
            inside.insert(
                4,
                f"WIN MODE led @{engine.won_at:,}" if engine.won_at is not None else f"win mode +{engine.win_extra:,}",
            )
        spare = [
            ("speed", f"{oracle.rate:.0f}/s p50 {oracle.p50_ms:.0f}ms") if oracle else None,
            (
                "oracle via",
                ", ".join(f"{b.name} {b.calls:,}{' RESTING' if b.resting else ''}" for b in oracle.limiter.backends),
            )
            if oracle
            else None,
            ("archive", f"{len(engine.archive):,} (restored {engine.restored:,})"),
        ]
        width = self.query_one("#lab-header", Static).content_size.width or 120
        label = "  ·  ".join(inside)
        overflow = []
        for item in filter(None, spare):
            text = " ".join(item)
            if len(label) + len(text) + 5 <= width:
                label += "  ·  " + text
            else:
                overflow.append(item)
        filled = round(width * min(calls / max(engine.budget, 1), 1.0))
        marks = self.level_marks or [(0, engine.level)]
        wins = {min(filled - 1, int(c / max(engine.budget, 1) * width)) for c in self.win_marks}
        bar = Text()
        for col in range(filled):
            at = col / width * engine.budget  # the call this column stands for
            level = next((lvl for start, lvl in reversed(marks) if start <= at), marks[0][1])
            colour = LEVEL_COLORS[min(level, len(LEVEL_COLORS) - 1)]
            if col in wins:
                bar.append("✓", style=f"bold black on {colour}")
            else:
                bar.append("█", style=colour)
        bar.append("░" * (width - filled), style="grey30")
        bar.append("\n")
        bar.append(label[:width], style="bold")
        return bar, overflow

    @staticmethod
    def level_label(engine: Engine) -> str:
        name, lead, _ = LEVELS[engine.level]
        label = f"L{engine.level} {name}"
        if lead == "sweep":
            label += f" {engine.strategies['sweep'].progress:.0%}"
        if not engine.escalate:
            label += " (escalation off)"
        elif engine.level == engine.max_level:
            label += " (top)"
        return label

    @staticmethod
    def update_item(item: QueueItem, engine: Engine) -> None:
        if len(engine.archive):
            best = engine.archive.top_by_board(1)[0]
            item.best = f"{best.p:.3f}/{best.units}w"
            item.win = engine.objective.wins(best.score, best.units, engine.leader)
        item.calls = engine.oracle.calls if engine.oracle else 0

    def render_queue(self) -> None:
        if self.engine:
            for item in self.queue:
                if item.state == "running" and (item.slug, item.target) == (self.engine.slug, self.engine.target):
                    self.update_item(item, self.engine)
        text = Text(no_wrap=True, overflow="ellipsis")
        icons = {
            "pending": ("·", "dim"),
            "running": ("▶", "bold yellow"),
            "done": ("✓", "green"),
            "skipped": ("-", "dim"),
            "error": ("!", "red"),
        }
        for item in self.queue:
            icon, style = icons[item.state]
            text.append(f"{icon} ", style=style)
            text.append(f"{item.title[:28]:<28} ", style="bold" if item.state == "running" else "")
            text.append(f"{item.leader:<9} ", style="yellow")
            best = item.best
            if item.state == "running" and self.engine and len(self.engine.archive):
                c = self.engine.archive.top_by_board(1)[0]
                best = f"{c.p:.3f}/{c.units}w"
            text.append(f"{best:<11}", style="bold green" if item.win else "")
            if item.win:
                text.append(" WIN", style="bold green")
            text.append("\n")
        self.query_one("#queue", Static).update(text)
        if self.index != self.scrolled_to and self.index >= 0:
            # Only follow the running item when it changes, so manual scrolling isn't undone every second.
            self.query_one("#queue-box", VerticalScroll).scroll_to(y=max(0, self.index - 2), animate=False)
            self.scrolled_to = self.index

    def render_table(self) -> None:
        engine = self.engine
        table = self.query_one("#table", DataTable)
        selected = self.selected_phrase()
        table.clear()
        target = self.target()
        for i, cand in enumerate(engine.archive.top_by_board(40), start=1):
            objective = engine.objective
            if objective.wins(cand.score, cand.units, engine.leader):
                gamble = objective.gamble(cand.score, cand.units, engine.leader)
                mark = (
                    Text(f"BET {objective.hits(cand.score)}/{cand.score.n}", style="bold magenta")
                    if gamble
                    else Text("WIN", style="bold green")
                )
            elif objective.beats(cand.p, cand.units, engine.leader):
                mark = Text("tie?", style="yellow")
            elif (
                objective.reachable(cand.score)
                and cand.score.n < 5
                and (engine.leader is None or cand.units < engine.leader.units)
            ):
                mark = Text("bet?", style="magenta")
            else:
                mark = Text("")
            saved = "*" if cand.phrase in engine.vaulted else ""
            table.add_row(
                str(i),
                Text(f"{cand.p:.3f}", style=p_style(cand.p, target)),
                f"{cand.score.spread:.3f}",
                str(cand.score.n),
                str(cand.units),
                cand.origin[:12],
                mark,
                saved + cand.phrase,
                key=cand.phrase,
            )
        if selected:
            try:
                table.move_cursor(row=table.get_row_index(selected))
            except Exception:
                pass

    def render_insight(self) -> None:
        engine = self.engine
        text = Text()
        if len(engine.archive):
            best = engine.archive.top_by_board(1)[0]
            ablation = engine.archive.ablation.get(best.phrase)
            text.append("best line, colored by how much each word matters:\n", style="dim")
            for i, word in enumerate(best.words):
                if ablation and i < len(ablation):
                    delta = ablation[i]
                    style = (
                        "bold white on red"
                        if delta >= 0.1
                        else "bold red"
                        if delta >= 0.03
                        else "yellow"
                        if delta >= 0.005
                        else "dim"
                    )
                else:
                    style = "bold"
                text.append(word, style=style)
                text.append(" ")
            if not ablation:
                text.append("(ablation runs during compress)", style="dim italic")
            text.append("\n\n")
        text.append_text(self.run_stats(engine))
        shares = engine.scheduler.shares()
        wins = engine.origin_wins
        novel = engine.novel_by_strategy
        text.append(
            f"{'strategy':<14}{'share':>5} {'bests':>5} {'ideas':>5} {'top':>6} {'hits':>5} {'reward':>6}\n",
            style="dim",
        )
        lead = LEVELS[engine.level][1]
        for name, arm in engine.scheduler.arms.items():
            strategy = engine.strategies[name]
            if isinstance(strategy, LLMGenerate) and not strategy.unlocked(engine.ctx):
                text.append(
                    f"{name[:13]:<14}{'':5} {'':5} {'':5} {'':6} {'':5} joins at L{TIER_LEVEL.get(strategy.tier, 0)}\n",
                    style="dim",
                )
                continue
            style = "bold reverse" if name == self.current else strategy_style(name) if name == lead else ""
            won = (
                wins.get(name, 0)
                + (wins.get("ablate", 0) + wins.get("resample", 0) if name == "compress" else 0)
                + (wins.get("llm_rewrite", 0) if name == "genetic" else 0)
            )
            top = self.strategy_top.get(name)
            text.append(
                f"{name[:13]:<14}{shares.get(name, 0):5.0%} {won:5} {novel.get(name, 0):5} "
                f"{f'{top:.3f}' if top is not None else '-':>6} {self.strategy_hits.get(name, 0) or '-':>5} "
                f"{arm.rate():6.2f}\n",
                style=style,
            )
        other = {k: v for k, v in wins.items() if k in ("planner", "manual", "site", "seed", "history", "triage")}
        if other:
            text.append("also found bests: " + ", ".join(f"{k} {v}" for k, v in other.items()) + "\n", style="dim")
        text.append(
            "top = its best score here · hits = its lines at the "
            + ("yes threshold" if self.game_mode.shortest else "leader's score")
            + "\n",
            style="dim",
        )
        name = self.current
        if name:
            text.append("\n")
            text.append(f" {name} ", style=f"{strategy_style(name)} reverse")
            text.append(" running now\n", style="dim")
            text.append(strategy_info(name))
        self.query_one("#insight", Static).update(text)

    def run_stats(self, engine: Engine) -> Text:
        """How the search is going on this question: hit rate of recent tries, and how long since the last best."""
        text = Text()
        calls = engine.oracle.calls if engine.oracle else 0
        if self.recent_p:
            target = self.target()
            share = sum(1 for p in self.recent_p if target and site_round(p) >= target) / len(self.recent_p)
            reach = "clear yes" if self.game_mode.shortest else "hit leader"
            text.append(f"last {len(self.recent_p)} tries ", style="dim")
            text.append(f"{share:.1%}", style="bold green" if share else "bold")
            text.append(f" {reach} · mean ", style="dim")
            text.append(f"{sum(self.recent_p) / len(self.recent_p):.3f}")
            text.append(" · top ", style="dim")
            text.append(f"{max(self.recent_p):.3f}\n", style=p_style(max(self.recent_p), target))
        quiet = int(time.monotonic() - self.last_best_time)
        text.append(f"{self.bests} new bests", style="bold")
        text.append(
            f" · last {quiet // 60}m{quiet % 60:02d}s / {calls - self.last_best_calls:,} calls ago", style="dim"
        )
        if self.bests and calls >= 100:
            text.append(f" · {self.bests / calls * 1000:.1f}/1k calls", style="dim")
        text.append("\n\n")
        return text

    # Selection and actions.

    def selected_phrase(self) -> str | None:
        table = self.query_one("#table", DataTable)
        if table.row_count == 0:
            return None
        try:
            return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            return None

    def selected(self):
        phrase = self.selected_phrase()
        return self.engine.archive.items.get(phrase) if phrase and self.engine else None

    def action_toggle_post(self) -> None:
        if not self.live_play or not self.live_session:
            return
        self.live_session.post_enabled = not self.live_session.post_enabled
        self.log_line("[live] posting on" if self.live_session.post_enabled else "[live] posting off")
        self.dirty = True

    def action_toggle_pause(self) -> None:
        if not self.engine:
            return
        if self.engine.paused.is_set():
            self.engine.pause()
            self.log_line("paused after the current batch")
        else:
            self.engine.resume()
            self.log_line("resumed")

    def action_next_question(self) -> None:
        if self.engine and self.engine.status != "stopped":
            self.log_line("skipping to the next question")
            self.engine.end_reason = "skipped by you"
            self.engine.stop()

    def action_compress(self) -> None:
        cand = self.selected()
        if cand:
            self.log_line(f"compress queued: {cand.phrase}")
            self.engine.submit(lambda: self.engine.compress_phrase(cand.phrase))

    def action_resample(self, n: int = 10) -> None:
        cand = self.selected()
        if cand:
            self.log_line(f"resample x{n} queued: {cand.phrase}")
            self.engine.submit(lambda: self.engine.resample(cand.phrase, n))

    def action_save(self) -> None:
        cand = self.selected()
        if cand:
            self.engine.save(cand)

    def action_queue_row(self) -> None:
        cand = self.selected()
        if not cand:
            return
        if self.live_play:
            if self.live_session and not self.live_session.post_enabled:
                self.log_line("[live] posting is off")
                return
            self.maybe_post_live()
            return
        self.engine.save(cand)
        vault.set_status(
            self.engine.slug, self.engine.mode, cand.phrase, "queued", self.engine.board, target=self.engine.target
        )
        self.log_line(f"queued for publishing: {cand.phrase}")

    def action_leave(self) -> None:
        self.cancelled = True
        if self.engine:
            self.engine.stop()
        if getattr(self.app, "has_home", False):
            self.app.pop_screen()
        else:
            self.app.exit()
