"""jevlab TUI: home (game mode, Search / Publish), question pickers, and the live lab."""

from __future__ import annotations

from textual.app import App

from ..config import EDITION
from ..modes import HIGHEST, GameMode, from_board
from .lab import LabScreen
from .pickers import HomeScreen


class JevlabApp(App):
    TITLE = f"jevlab · {EDITION}"

    def __init__(self, lab_slug: str | None = None, budget: int = 20000, use_llm: bool = True,
                 seed: int | None = None, escalate: bool = True, max_level: int = 4,
                 game_mode: GameMode = HIGHEST):
        super().__init__()
        self.lab_slug = lab_slug
        self.game_mode = game_mode
        self.lab_args = {"budget": budget, "use_llm": use_llm, "seed": seed, "escalate": escalate,
                         "max_level": max_level}
        self.has_home = lab_slug is None

    def on_mount(self) -> None:
        if self.lab_slug:
            self.push_screen(LabScreen([self.lab_slug], single=True, board=self.game_mode.board, **self.lab_args))
        else:
            self.push_screen(HomeScreen())


def run_home() -> int:
    app = JevlabApp()
    switch_to = app.run()
    if isinstance(switch_to, str):
        # The edition is fixed at import, so switching restarts the process with the other one.
        import os
        import sys

        os.execv(sys.executable, [sys.executable, "-m", "jevlab.cli", "--edition", switch_to])
    return 0


def run_lab(slug: str, budget: int = 20000, use_llm: bool = True, seed: int | None = None,
            escalate: bool = True, max_level: int = 4, board: str = HIGHEST.board) -> int:
    JevlabApp(slug, budget, use_llm, seed, escalate, max_level, from_board(board)).run()
    return 0
