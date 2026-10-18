"""A quiet line at the top right: the screen, the edition, and the mode when one applies."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

NAMES = {
    "kev": "Kev",
    "laya": "Laya",
    "clef": "Clef",
    "clef-flash": "Clef-flash",
    "clef-omni": "Clef-omni",
    "luna": "Luna",
    "decider": "Decider",
    "microsoft": "Microsoft",
}


def edition_name() -> str:
    from ..config import EDITION

    return NAMES.get(EDITION, "Jev")


def mode_phrase(label: str) -> str:
    """`Strict Highest` becomes `Strict Highest Mode`. A label that already says Mode stays as it is."""
    text = label.strip()
    if text.lower().endswith(" mode"):
        return text
    return f"{text} Mode"


def edition_caption(screen: str, mode: str | None = None, *, nudge: bool = False) -> Text:
    parts = [screen, f"{edition_name()} Edition"]
    if mode:
        parts.append(mode_phrase(mode))
    text = Text(" | ".join(parts), style="dim", justify="right")
    if nudge:
        text.append("  ")
        text.append("NUDGE", style="bold black on yellow")
    return text


class EditionMark(Static):
    """One muted line at the top right of every screen."""

    DEFAULT_CSS = """
    EditionMark {
        dock: top;
        height: 1;
        width: 100%;
        content-align: right middle;
        padding: 0 1;
        color: $text-muted;
        background: $background;
    }
    """

    def __init__(self, screen: str, mode: str | None = None) -> None:
        self._screen = screen
        self._mode = mode
        self._nudge = False
        super().__init__(edition_caption(screen, mode))

    def set_mode(self, mode: str | None) -> None:
        self._mode = mode
        self.update(edition_caption(self._screen, mode, nudge=self._nudge))

    def set_nudge(self, on: bool) -> None:
        self._nudge = on
        self.update(edition_caption(self._screen, self._mode, nudge=on))
