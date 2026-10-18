"""A nudge is one generator sweep, then the search carries on.

Pressing M arms it. The next break between rounds asks every generator model and covers every tactic,
the way a hard deal does, without entering a hard visit. The indicator is lit only while the sweep is waiting.
"""

import asyncio
import re

from textual.app import App, ComposeResult

from jevlab.search.hardboard import CATALOG, nudge_batches
from jevlab.search.probe import Lock
from jevlab.tui.chrome import EditionMark, edition_caption
from jevlab.tui.lab import LabScreen

from test_hard_board import FakeOracle, Writer, engine


def test_a_nudge_assigns_every_model_and_every_tactic():
    few = ["m0", "m1", "m2", "m3"]
    batches = nudge_batches(few)
    frames = [frame for batch in batches for frame in batch.frames]
    assert [batch.model for batch in batches] == [few[i % len(few)] for i in range(len(batches))]
    assert frames == list(CATALOG)
    assert all(1 <= len(batch.frames) <= 3 for batch in batches)

    many = [f"m{i}" for i in range(40)]
    wide = nudge_batches(many)
    assert [batch.model for batch in wide] == many
    flat = [frame for batch in wide for frame in batch.frames]
    assert flat[: len(CATALOG)] == list(CATALOG)
    assert len(set(flat[: len(CATALOG)])) == len(CATALOG)
    assert nudge_batches([]) == []


class _Oracle(FakeOracle):
    async def close(self) -> None:
        return None


class _Rounds:
    """One ordinary round, then the nudge is armed twice, then one more ordinary round."""

    name = "sweep"

    def __init__(self, writer: Writer):
        self.writer = writer
        self.prompts_at_run: list[int] = []

    def available(self, ctx) -> bool:
        return True

    async def run(self, ctx) -> None:
        self.prompts_at_run.append(len(self.writer.prompts))
        if len(self.prompts_at_run) == 1:
            ctx.engine.request_nudge()
            ctx.engine.request_nudge()
            return
        if len(self.prompts_at_run) == 2:
            return
        ctx.engine.stopping = True


def test_a_nudge_waits_for_the_next_break_then_search_continues(tmp_path, monkeypatch):
    models = ["alpha/one", "beta/two"]
    monkeypatch.setattr("jevlab.config.GEN_MODELS", models)
    eng = engine(tmp_path, "easy", leader=(0.40, 8))
    assert not eng.on_hard_board()
    eng.level = 2
    eng.probe_lock = Lock(model="locked-writer", batches=3)
    writer = Writer()
    writer.armed: list[bool] = []

    async def close() -> None:
        return None

    writer.close = close

    async def json(model, system, user, temperature=1.0, max_tokens=4000, **kwargs):
        writer.armed.append(eng.nudge_armed)
        return await Writer.json(writer, model, system, user, temperature, max_tokens, **kwargs)

    writer.json = json
    eng.llm = writer
    rounds = _Rounds(writer)
    eng.strategies["sweep"] = rounds
    eng.force("sweep")

    async def start():
        eng.oracle = _Oracle()

    eng.start = start
    asyncio.run(eng.run())

    assert rounds.prompts_at_run == [0, len(writer.prompts), len(writer.prompts)]
    assert writer.prompts
    assert writer.armed and not any(writer.armed)
    called = [model for model, _ in writer.prompts]
    assert called == [models[i % len(models)] for i in range(len(called))]
    assert set(called) == set(models)
    found: list[str] = []
    for _, user in writer.prompts:
        found.extend(re.findall(r"^- \[(\w+)\]", user, re.M))
        assert "about 60 words" not in user
        assert "mostly 6 to 18" in user
    assert found == list(CATALOG)
    assert eng.level == 2
    assert eng.probe_lock.model == "locked-writer" and eng.probe_lock.batches == 3
    assert eng.schedule is None
    assert not eng.on_hard_visit()
    assert not eng.nudge_armed
    assert eng.rounds == 2


def test_m_lights_nudge_in_the_top_bar_until_the_sweep_starts(monkeypatch, tmp_path):
    from jevlab.db import DB

    keys = [(binding.key, binding.action) for binding in LabScreen.BINDINGS]
    assert ("m", "nudge") in keys
    assert ("n", "next_question") in keys

    monkeypatch.setattr("jevlab.tui.lab.DB", lambda: DB(tmp_path / "t.db"))

    async def no_queue(self):
        return None

    monkeypatch.setattr(LabScreen, "run_queue", no_queue)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(LabScreen([]))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            eng = engine(tmp_path, "bar", leader=(0.40, 8))
            screen.engine = eng
            mark = screen.query_one(EditionMark)
            assert "NUDGE" not in mark.content.plain
            await pilot.press("m")
            await pilot.pause()
            assert eng.nudge_armed
            assert mark.content.plain.endswith("NUDGE")
            lit = edition_caption("Lab", "Strict Highest", nudge=True)
            assert "yellow" in str(lit.spans[-1].style)
            eng.nudge_armed = False
            screen.refresh_view()
            assert "NUDGE" not in screen.query_one(EditionMark).content.plain

    asyncio.run(check())
