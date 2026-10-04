"""Copy mouse-selected text to the clipboard.

The terminal's copy-on-select never sees a drag while this app owns the mouse.
When the button is released, the selection is written with OSC 52 so the
terminal can apply its own clipboard rules.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.cells import get_character_cell_size
from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.app import App
from textual.selection import Selection
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import DataTable, RichLog


def copy_selection_to_clipboard(app: App) -> None:
    """Copy the active screen's selection, if it has any text."""
    raw = app.screen.get_selected_text()
    if not raw:
        return
    text = "\n".join(line.rstrip() for line in raw.split("\n")).strip("\n")
    if not text:
        return
    app.copy_to_clipboard(text)


def char_index_at_cell(strip: Strip, cell: int) -> int:
    """Character index of the glyph that starts at or after ``cell``."""
    if cell <= 0:
        return 0
    pos = 0
    chars = 0
    for segment in strip:
        for character in segment.text:
            if pos >= cell:
                return chars
            pos += get_character_cell_size(character)
            chars += 1
    return chars


def extract_selection(selection: Selection, line_at: Callable[[int], str], line_count: int) -> str:
    """Pull the selected span out of a widget's logical lines."""
    if line_count <= 0:
        return ""
    if selection.start is None and selection.end is None:
        ys = range(line_count)
    elif selection.end is None:
        ys = range(max(0, selection.start.y), line_count)
    elif selection.start is None:
        ys = range(0, min(line_count, selection.end.y + 1))
    else:
        y0, y1 = selection.start.y, selection.end.y
        if y1 < y0:
            y0, y1 = y1, y0
        ys = range(max(0, y0), min(line_count, y1 + 1))
    parts: list[str] = []
    for y in ys:
        span = selection.get_span(y)
        if span is None:
            continue
        start, end = span
        text = line_at(y)
        parts.append(text[start:] if end < 0 else text[start:end])
    return "\n".join(parts)


def _paint_span(strip: Strip, start: int, end: int, style: RichStyle) -> Strip:
    width = strip.cell_length
    if end < 0 or end > width:
        end = width
    start = max(0, min(start, width))
    end = max(start, min(end, width))
    if start >= end:
        return strip
    segments: list[Segment] = []
    if start:
        segments.extend(strip.crop(0, start))
    segments.extend(strip.crop(start, end).apply_style(style))
    if end < width:
        segments.extend(strip.crop(end, width))
    return Strip(segments, width)


def paint_selection(widget: Widget, strip: Strip, y: int, x_origin: int) -> Strip:
    """Highlight ``strip`` where ``y`` is selected. ``x_origin`` is its first character."""
    selection = widget.text_selection
    if selection is None:
        return strip
    span = selection.get_span(y)
    if span is None:
        return strip
    start, end = span
    start -= x_origin
    if end >= 0:
        end -= x_origin
    return _paint_span(strip, start, end, widget.selection_style)


class SelectableRichLog(RichLog):
    """RichLog whose lines can be drag-selected and copied."""

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        _scroll_x, scroll_y = self.scroll_offset
        content_y = scroll_y + y
        if 0 <= content_y < len(self.lines):
            origin = char_index_at_cell(self.lines[content_y], _scroll_x)
        else:
            origin = 0
        return paint_selection(self, strip.apply_offsets(origin, content_y), content_y, origin)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        def line_at(index: int) -> str:
            if 0 <= index < len(self.lines):
                return self.lines[index].text
            return ""

        text = extract_selection(selection, line_at, len(self.lines))
        if not text:
            return None
        return text, "\n"


class SelectableDataTable(DataTable):
    """DataTable that allows drag-selecting cell text.

    Textual turns selection off on DataTable. The row cursor still follows clicks.
    """

    ALLOW_SELECT = True

    def _fixed_rows_height(self) -> int:
        height = sum(self.get_row_height(self._row_locations.get_key(index)) for index in range(self.fixed_rows))
        if self.show_header:
            height += self.get_row_height(self._header_row_key)
        return height

    def _content_y(self, viewport_y: int) -> int:
        if viewport_y >= self._fixed_rows_height():
            return viewport_y + self.scroll_offset.y
        return viewport_y

    def render_line(self, y: int) -> Strip:
        content_y = self._content_y(y)
        strip = super().render_line(y).apply_offsets(0, content_y)
        return paint_selection(self, strip, content_y, 0)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        width = self.size.width
        scroll_x = self.scroll_offset.x

        def line_at(content_y: int) -> str:
            return self._render_line(content_y, scroll_x, scroll_x + width, self.rich_style).text

        text = extract_selection(selection, line_at, self.virtual_size.height)
        if not text:
            return None
        return text, "\n"
