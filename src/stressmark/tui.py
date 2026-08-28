"""Interactive terminal viewer for analyzed stressmark documents."""

from dataclasses import dataclass
from typing import ClassVar

from rich.cells import cell_len
from rich.style import Style as RichStyle
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.content import Content, Span
from textual.style import Style as TextualStyle
from textual.widgets import Footer, Header, Static

from stressmark import render
from stressmark.model import APP_NAME

_DETAIL_META_STYLE = render.rich_style(render.VisualRole.LEGEND_DESCRIPTION)
_DETAIL_WORD_STYLE = "bold cyan"


@dataclass(frozen=True)
class _WordLocation:
    """A selectable word's token index and wrapped display position."""

    token_index: int
    row: int
    column: int


_WORD_LINK_PREFIX = "stressmark-word:"


def _wrapped_word_locations(text, word_ranges, width, console):
    """Map selectable words to Textual's wrapped display rows and columns."""
    content = Content.from_rich_text(text, console=console)
    content = content.add_spans(
        [
            Span(
                start,
                end,
                TextualStyle(link=f"{_WORD_LINK_PREFIX}{word_index}"),
            )
            for word_index, (_, start, end) in enumerate(word_ranges)
        ]
    )

    locations = [None] * len(word_ranges)
    for row, line in enumerate(content.wrap(max(width, 1))):
        for span in line.spans:
            style = span.style
            if not isinstance(style, TextualStyle) or not style.link:
                continue
            if not style.link.startswith(_WORD_LINK_PREFIX):
                continue
            word_index = int(style.link.removeprefix(_WORD_LINK_PREFIX))
            if locations[word_index] is None:
                token_index = word_ranges[word_index][0]
                locations[word_index] = _WordLocation(
                    token_index,
                    row,
                    cell_len(line.plain[:span.start]),
                )

    if any(location is None for location in locations):
        raise RuntimeError("Unable to locate every selectable word after wrapping")
    return locations


class StressmarkApp(App[None]):
    """Browse a stress-marked document one source word at a time."""

    TITLE = APP_NAME
    SUB_TITLE = "interactive stress viewer"

    CSS = f"""
    Screen {{
        background: {render.DARK_THEME.background};
        color: {render.DARK_THEME.foreground};
    }}

    #document {{
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
        scrollbar-color: {render.DARK_THEME.primary};
        scrollbar-color-hover: {render.DARK_THEME.nuclear};
        scrollbar-background: {render.DARK_THEME.chrome};
    }}

    #word-detail {{
        height: 5;
        padding: 0 2;
        border-top: solid {render.DARK_THEME.rule};
        background: {render.DARK_THEME.surface};
        color: {render.DARK_THEME.foreground};
    }}

    Header, Footer {{
        background: {render.DARK_THEME.chrome};
        color: {render.DARK_THEME.foreground};
    }}
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left,h", "previous_word", "←/h word", show=True),
        Binding("right,l", "next_word", "→/l word", show=True),
        Binding("up,k", "previous_line", "↑/k line", show=True),
        Binding("down,j", "next_line", "↓/j line", show=True),
        Binding("home", "first_word", "Home first", show=False),
        Binding("end", "last_word", "End last", show=False),
        Binding("q", "quit", "q quit", show=True),
    ]

    _SELECTION_STYLE = RichStyle(
        color="white",
        bgcolor=render.DARK_THEME.selection,
        bold=True,
        reverse=False,
    )

    def __init__(
        self,
        raw_tokens,
        display_results,
        lexical_results,
        *,
        show_rules=False,
        flag_heteronyms=False,
        source_name=None,
    ):
        super().__init__()
        self.raw_tokens = raw_tokens
        self.display_results = display_results
        self.lexical_results = lexical_results
        self.word_ranges = []
        self.base_text = render._render_rich_text(
            raw_tokens,
            display_results,
            show_rules,
            flag_heteronyms,
            word_ranges=self.word_ranges,
        )
        self.selected_index = 0 if self.word_ranges else -1
        self._preferred_column = None
        self._word_location_cache = {}
        if source_name:
            self.sub_title = source_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="document")
        yield Static(id="word-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_selection()

    def action_previous_word(self) -> None:
        self._move_word(-1)

    def action_next_word(self) -> None:
        self._move_word(1)

    def action_previous_line(self) -> None:
        self._move_line(-1)

    def action_next_line(self) -> None:
        self._move_line(1)

    def action_first_word(self) -> None:
        if self.word_ranges:
            self.selected_index = 0
            self._preferred_column = None
            self._refresh_selection()

    def action_last_word(self) -> None:
        if self.word_ranges:
            self.selected_index = len(self.word_ranges) - 1
            self._preferred_column = None
            self._refresh_selection()

    def _move_word(self, delta: int) -> None:
        if not self.word_ranges:
            return
        target = max(0, min(self.selected_index + delta, len(self.word_ranges) - 1))
        if target != self.selected_index:
            self.selected_index = target
            self._preferred_column = None
            self._refresh_selection()

    def _move_line(self, direction: int) -> None:
        if not self.word_ranges:
            return

        document = self.query_one("#document", Static)
        word_locations = self._get_word_locations(document)
        current = word_locations[self.selected_index]
        if self._preferred_column is None:
            self._preferred_column = current.column

        candidate_rows = {
            location.row
            for location in word_locations
            if (location.row - current.row) * direction > 0
        }
        if not candidate_rows:
            return

        target_row = min(candidate_rows) if direction > 0 else max(candidate_rows)
        candidates = [
            (index, location)
            for index, location in enumerate(word_locations)
            if location.row == target_row
        ]
        self.selected_index = min(
            candidates,
            key=lambda item: (abs(item[1].column - self._preferred_column), item[0]),
        )[0]
        self._refresh_selection()

    def _get_word_locations(self, document):
        """Return wrapped locations, recomputing them only after width changes."""
        width = max(document.content_region.width, 1)
        if width not in self._word_location_cache:
            self._word_location_cache[width] = _wrapped_word_locations(
                self.base_text,
                self.word_ranges,
                width,
                self.console,
            )
        return self._word_location_cache[width]

    def _refresh_selection(self) -> None:
        document = self.query_one("#document", Static)
        detail = self.query_one("#word-detail", Static)

        if self.selected_index < 0:
            document.update(self.base_text)
            detail.update(
                Text("No selectable words in this document.", style=_DETAIL_META_STYLE)
            )
            return

        token_index, start, end = self.word_ranges[self.selected_index]
        selected_text = self.base_text.copy()
        selected_text.stylize(self._SELECTION_STYLE, start, end)
        document.update(selected_text)

        original = self.raw_tokens[token_index][1]
        result = self.lexical_results[token_index]
        raw_stress = render.render_word(result)
        detail_text = Text()
        detail_text.append(
            f"Word {self.selected_index + 1}/{len(self.word_ranges)}  ",
            style=_DETAIL_META_STYLE,
        )
        detail_text.append(original, style=_DETAIL_WORD_STYLE)
        detail_text.append("\nRaw stressmark: ", style=_DETAIL_META_STYLE)
        detail_text.append_text(raw_stress)
        detail_text.append(
            f"   POS: {result.tag or 'unknown'}   "
            f"source: {render.confidence_label(result.confidence)}",
            style=_DETAIL_META_STYLE,
        )
        detail.update(detail_text)

        word_locations = self._get_word_locations(document)
        display_row = word_locations[self.selected_index].row
        document.scroll_to(y=max(0, display_row - 2), immediate=True, force=True)


def run_tui(
    raw_tokens,
    display_results,
    lexical_results,
    *,
    show_rules=False,
    flag_heteronyms=False,
    source_name=None,
):
    """Run the interactive stress viewer until the user quits."""
    StressmarkApp(
        raw_tokens,
        display_results,
        lexical_results,
        show_rules=show_rules,
        flag_heteronyms=flag_heteronyms,
        source_name=source_name,
    ).run()
