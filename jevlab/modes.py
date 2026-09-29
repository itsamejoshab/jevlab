"""Game modes: a site play mode plus the board we compete on. Each keeps its own run memory, vault file, and
publish records; oracle samples are shared because a phrase scores the same whichever board it goes to.

Golf modes are publish-only: their vault lines are cross-posted from Strict, never searched."""

from __future__ import annotations

from dataclasses import dataclass

HIGH_SCORES = "highScores"
SHORTEST_YES = "shortestYes"
BOARDS = (HIGH_SCORES, SHORTEST_YES)
STRICT = "strict_chain"
GOLF = "golf"
SEARCHABLE_KINDS = ("noul", "choice")


def is_searchable(kind: str | None, ranked: bool = False) -> bool:
    """Yes/no questions and unranked choice questions (scored on Jev's top option). Ranked choice questions
    judge the answer's position rather than one option, so they stay out."""
    return kind in SEARCHABLE_KINDS and not (kind == "choice" and ranked)


@dataclass(frozen=True)
class GameMode:
    name: str
    label: str
    play_mode: str = STRICT
    board: str = HIGH_SCORES

    @property
    def shortest(self) -> bool:
        return self.board == SHORTEST_YES

    @property
    def golf(self) -> bool:
        return self.play_mode == GOLF

    @property
    def searchable(self) -> bool:
        return self.play_mode == STRICT

    @property
    def unit(self) -> str:
        return "char" if self.golf else "word"

    @property
    def unit_abbr(self) -> str:
        return "c" if self.golf else "w"


HIGHEST = GameMode("highest", "Strict Highest")
SHORTEST = GameMode("shortest", "Strict Shortest yes", board=SHORTEST_YES)
GAME_MODES = (HIGHEST, SHORTEST)
GOLF_HIGHEST = GameMode("golf-highest", "Golf Highest", play_mode=GOLF)
GOLF_SHORTEST = GameMode("golf-shortest", "Golf Shortest yes", play_mode=GOLF, board=SHORTEST_YES)
PUBLISH_MODES = GAME_MODES + (GOLF_HIGHEST, GOLF_SHORTEST)


def from_name(name: str) -> GameMode:
    for mode in PUBLISH_MODES:
        if name in (mode.name, mode.board):
            return mode
    raise ValueError(f"unknown game mode {name!r}; pick one of {', '.join(m.name for m in PUBLISH_MODES)}")


def from_board(board: str | None, play_mode: str | None = STRICT) -> GameMode:
    for mode in PUBLISH_MODES:
        if mode.board == (board or HIGH_SCORES) and mode.play_mode == (play_mode or STRICT):
            return mode
    return SHORTEST if board == SHORTEST_YES else HIGHEST
