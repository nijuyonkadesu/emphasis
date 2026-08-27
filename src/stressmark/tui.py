"""Interactive terminal viewer for analyzed stressmark documents."""

from dataclasses import dataclass
from typing import ClassVar

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from stressmark import render


@dataclass(frozen=True)
class _WordLocation:
    """A selectable word's token index and original source position."""

    token_index: int
    line: int
    column: int


def _word_locations(raw_tokens):
    """Map word tokens to zero-based source lines and columns."""
    locations = []
    line = 0
    column = 0
    for token_index, (is_word, token_text) in enumerate(raw_tokens):
        if is_word:
            locations.append(_WordLocation(token_index, line, column))

        pieces = token_text.split("\n")
        if len(pieces) == 1:
            column += len(token_text)
        else:
            line += len(pieces) - 1
            column = len(pieces[-1])
    return locations


class StressmarkApp(App[None]):
    """Browse a stress-marked document one source word at a time."""

    TITLE = "stressmark"
    SUB_TITLE = "interactive stress viewer"

    CSS = """
    Screen {
        background: #15140f;
        color: #ede8db;
    }

    #document {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
        scrollbar-color: #e7a23b;
        scrollbar-color-hover: #ffc25c;
        scrollbar-background: #252217;
    }

    #word-detail {
        height: 5;
        padding: 0 2;
        border-top: solid #6fa8a0;
        background: #201e16;
        color: #ede8db;
    }

    Header, Footer {
        background: #252217;
        color: #ede8db;
    }
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

    _SELECTION_STYLE = Style(
        color="white",
        bgcolor="#006b78",
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
        self.word_locations = _word_locations(raw_tokens)
        self.selected_index = 0 if self.word_ranges else -1
        self._preferred_column = None
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
        if not self.word_locations:
            return

        current = self.word_locations[self.selected_index]
        if self._preferred_column is None:
            self._preferred_column = current.column

        candidate_lines = {
            location.line
            for location in self.word_locations
            if (location.line - current.line) * direction > 0
        }
        if not candidate_lines:
            return

        target_line = min(candidate_lines) if direction > 0 else max(candidate_lines)
        candidates = [
            (index, location)
            for index, location in enumerate(self.word_locations)
            if location.line == target_line
        ]
        self.selected_index = min(
            candidates,
            key=lambda item: (abs(item[1].column - self._preferred_column), item[0]),
        )[0]
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        document = self.query_one("#document", Static)
        detail = self.query_one("#word-detail", Static)

        if self.selected_index < 0:
            document.update(self.base_text)
            detail.update(Text("No selectable words in this document.", style="dim"))
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
            style="dim",
        )
        detail_text.append(original, style="bold cyan")
        detail_text.append("\nRaw stressmark: ", style="dim")
        detail_text.append_text(raw_stress)
        detail_text.append(
            f"   POS: {result.tag or 'unknown'}   source: {result.confidence}",
            style="dim",
        )
        detail.update(detail_text)

        # Approximate the selected word's wrapped display row. Rich/Textual
        # may differ by a column for wide Unicode markers, but this keeps long
        # paragraphs reliably near the cursor without altering their layout.
        width = max(document.content_size.width, 1)
        prefix_lines = self.base_text.plain[:start].split("\n")
        display_row = sum(
            max(1, (len(line) // width) + 1) for line in prefix_lines[:-1]
        )
        display_row += len(prefix_lines[-1]) // width
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
