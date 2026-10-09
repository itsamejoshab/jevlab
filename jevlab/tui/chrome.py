"""A quiet edition name, present on every screen."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

_NAMES = {
    "kev": "Kev",
    "laya": "Laya",
    "clef": "Clef",
    "luna": "Luna",
    "decider": "Decider",
}


def edition_caption() -> Text:
    from ..config import EDITION

    return Text(f"{_NAMES.get(EDITION, 'Jev')} mode", style="dim", justify="right")


class EditionMark(Static):
    """One muted line at the top right. Stays up on home, search, publish, and the lab."""

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

    def __init__(self) -> None:
        super().__init__(edition_caption())
