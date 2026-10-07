"""Turn publisher transcript lines into a colored log for the submit screen.

The publisher still speaks in the same plain lines the CLI prints. This only
changes how the submit screen shows them: one block per phrase, with the
result on its own line.
"""

from __future__ import annotations

import re

from rich.text import Text

_START = re.compile(r"^(?P<dry>DRY RUN: )?publishing (?P<n>\d+) line\(s\), re-rolls (?P<rolls>\d+)$")
_HEADER = re.compile(r"^== (?P<slug>\S+) \[(?P<label>.+)\] (?P<score>\S+): (?P<phrase>.*)$")
_SKIPPED = re.compile(r"^== (?P<slug>\S+): skipped (?P<phrase>.*)$")
_LEADER = re.compile(r"^\[(?P<where>.+)\] live leader (?P<rest>.*)$")
_REROLL = re.compile(r"^re-roll (?P<n>\d+): (?P<p>[0-9.]+) \(best (?P<best>[0-9.]+)\)$")
_POSTED = re.compile(r"^= (?P<units>\d+c) (?P<p>[0-9.]+)$")
_CHAIN = re.compile(r"^(?P<op>[+-]) (?P<word>.+?) +(?P<p>[0-9.]+)$")
_ORACLE = re.compile(r"^oracle re-check (?P<fresh>[0-9.]+) \(vault said (?P<vault>[0-9.]+)\)$")
_RESCORE = re.compile(
    r"^(?P<how>re-scoring the phrase|re-rolling the last word) (?P<n>\d+) time\(s\); "
    r"the board keeps the best score(?P<extra>.*)$"
)
_BORROWED = re.compile(r"^estimated from (?P<edition>\S+): posting the borrowed score without an oracle check$")
_RESULT = re.compile(r"^-> (?P<body>.*)$")
_FINISHED = re.compile(r"^finished: (?P<won>\d+)/(?P<total>\d+) on top(?P<tail>.*)$")
_FALL = re.compile(r"^falling through to the next vault line: (?P<phrase>.*)$")


class PublishLog:
    """Stateful formatter: a re-roll is marked when it beats the score so far."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.best: float | None = None
        self.slug = ""

    def render(self, message: str) -> Text:
        rendered = Text()
        first = True
        for raw in str(message).splitlines():
            piece = self._one(raw.strip())
            if piece is None:
                continue
            if not first:
                rendered.append("\n")
            first = False
            rendered.append_text(piece)
        return rendered

    def _one(self, line: str) -> Text | None:
        if not line:
            return Text("")
        matched = self._match(line)
        if matched is not None:
            return matched
        text = Text("  " + line)
        if line.startswith("stopped") or "rate limited" in line or line.startswith("busy"):
            text.stylize("yellow")
        elif any(word in line for word in ("fail", "banned", "rejected", "crashed")):
            text.stylize("red")
        return text

    def _match(self, line: str) -> Text | None:
        if found := _START.match(line):
            self.total = int(found["n"])
            self.count = 0
            text = Text()
            if found["dry"]:
                text.append("dry run  ", style="bold yellow")
            text.append(f"{found['n']} lines · {found['rolls']} re-rolls", style="bold")
            return text
        if found := _HEADER.match(line):
            return self._header(found)
        if found := _SKIPPED.match(line):
            text = Text()
            text.append("skipped  ", style="bold yellow")
            text.append(found["slug"], style="bold")
            text.append("\n  " + found["phrase"], style="dim")
            return text
        if found := _LEADER.match(line):
            return self._leader(found["rest"])
        if found := _RESCORE.match(line):
            kind = "re-score" if found["how"].startswith("re-scoring") else "re-roll"
            extra = found["extra"].strip(" ,")
            text = Text(f"  {kind} ×{found['n']}", style="dim")
            text.append("  board keeps the best", style="dim")
            if extra:
                text.append(f"  {extra}", style="dim")
            return text
        if found := _POSTED.match(line):
            self._note(float(found["p"]))
            text = Text("  posted  ", style="dim")
            text.append(f"{found['units']}  {found['p']}", style="bold")
            return text
        if found := _REROLL.match(line):
            return self._reroll(found)
        if found := _CHAIN.match(line):
            self._note(float(found["p"]))
            text = Text(f"  {found['op']} {found['word'].strip()}", style="cyan")
            text.append(f"  {found['p']}", style="bold")
            return text
        if found := _ORACLE.match(line):
            text = Text("  oracle  ", style="dim")
            text.append(found["fresh"], style="bold")
            text.append(f"  vault {found['vault']}", style="dim")
            return text
        if found := _BORROWED.match(line):
            return Text(f"  borrowed from {found['edition']}, no oracle check", style="cyan")
        if found := _RESULT.match(line):
            return self._result(found["body"])
        if found := _FINISHED.match(line):
            return self._finished(found)
        if found := _FALL.match(line):
            text = Text("  next  ", style="bold magenta")
            text.append(found["phrase"])
            return text
        return None

    def _header(self, found: re.Match[str]) -> Text:
        self.count += 1
        self.best = None
        self.slug = found["slug"]
        label = found["label"]
        mode, _, target = label.partition(" -> ")
        mark = f"{self.count}/{self.total}" if self.total and self.count <= self.total else str(self.count)
        text = Text()
        text.append(f"▸ {mark}  ", style="bold cyan")
        text.append(mode, style="bold cyan")
        if target:
            text.append("  →  ", style="dim")
            text.append(target, style="bold")
        text.append(f"\n  {found['slug']}", style="bold")
        text.append(f"  {found['score']}", style="bold")
        phrase = found["phrase"].strip()
        if phrase:
            text.append("\n  " + phrase)
        return text

    def _leader(self, rest: str) -> Text:
        lead, _, ours = rest.partition("; ours ")
        text = Text("  leader  ", style="dim")
        text.append(lead.strip(), style="yellow")
        if ours:
            text.append("   ours  ", style="dim")
            text.append(ours.strip(), style="bold")
        return text

    def _reroll(self, found: re.Match[str]) -> Text:
        best = float(found["best"])
        improved = self.best is not None and best > self.best + 1e-9
        self._note(best)
        text = Text(f"  {int(found['n']):>2}  ", style="dim")
        text.append(found["p"], style="bold green" if improved else "dim")
        text.append(f"   best {found['best']}", style="bold green" if improved else "dim")
        return text

    def _result(self, body: str) -> Text:
        text = Text()
        if body.startswith("won") or body == "already":
            mark, style = "✓ ", "bold green"
        elif body.startswith("dry-run") or body.startswith("dry run"):
            mark, style = "· ", "bold yellow"
        else:
            mark, style = "✗ ", "bold red"
        text.append("  " + mark + body, style=style)
        if self.slug:
            text.append("   " + self.slug, style="dim")
        return text

    def _finished(self, found: re.Match[str]) -> Text:
        won = int(found["won"])
        total = int(found["total"])
        if total and won == total:
            style = "bold green"
        elif won:
            style = "bold yellow"
        else:
            style = "bold red"
        text = Text(f"{won}/{total} on top", style=style)
        if "dry run" in found["tail"]:
            text.append("  · nothing sent", style="yellow")
        return text

    def _note(self, score: float) -> None:
        if self.best is None or score > self.best:
            self.best = score
