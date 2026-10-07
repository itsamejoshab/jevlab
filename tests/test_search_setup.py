"""Search setup can hide choice questions."""

import asyncio

from textual.app import App, ComposeResult

from jevlab.boards import Standing
from jevlab.db import DB
from jevlab.modes import HIGHEST
from jevlab.objective import Leader
from jevlab.tui.pickers import SearchSetupScreen


def _prompt(option) -> str:
    prompt = option.prompt
    return prompt.plain if hasattr(prompt, "plain") else str(prompt)


def test_exclude_choice_hides_choice_questions(monkeypatch, tmp_path):
    rows = [
        Standing("yes-q", "Is cereal a soup", "noul", "yes", Leader(0.4, 3, "Ada"), None, None, False, False),
        Standing(
            "flavor", "Best flavor", "choice", "yes", Leader(0.5, 4, "Ada"), None, None, False, False, target="Vanilla"
        ),
    ]
    monkeypatch.setattr("jevlab.tui.pickers.DB", lambda: DB(tmp_path / "t.db"))
    monkeypatch.setattr("jevlab.tui.pickers.standings", lambda *args, **kwargs: rows)

    class _App(App):
        def on_mount(self) -> None:
            self.push_screen(SearchSetupScreen(HIGHEST))

        def compose(self) -> ComposeResult:
            yield from ()

    async def check() -> None:
        app = _App()
        async with app.run_test(size=(140, 36)) as pilot:
            await pilot.pause()
            screen = app.screen
            picker = screen.query_one("#picker")

            def shown() -> list[str]:
                return [_prompt(picker.get_option_at_index(i)) for i in range(picker.option_count)]

            assert any("Is cereal a soup" in line for line in shown())
            assert any("Best flavor" in line for line in shown())
            await pilot.click("#no-choice")
            await pilot.pause()
            assert any("Is cereal a soup" in line for line in shown())
            assert not any("Best flavor" in line for line in shown())
            await pilot.click("#all")
            await pilot.pause()
            assert not any("Best flavor" in line for line in shown())
            await pilot.click("#no-choice")
            await pilot.pause()
            assert any("Best flavor" in line for line in shown())

    asyncio.run(check())
