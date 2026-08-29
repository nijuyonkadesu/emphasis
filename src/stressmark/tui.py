"""Interactive terminal viewer for analyzed stressmark documents."""

from dataclasses import dataclass
from typing import ClassVar

from rich.cells import cell_len
from rich.style import Style as RichStyle
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content, Span
from textual.screen import ModalScreen
from textual.style import Style as TextualStyle
from textual.widgets import Footer, Header, SelectionList, Static
from textual.widgets.selection_list import Selection

from stressmark import render
from stressmark.engine import analyze
from stressmark.model import APP_NAME
from stressmark.viewer import analyze_for_viewer

_DETAIL_META_STYLE = render.rich_style(render.VisualRole.LEGEND_DESCRIPTION)
_DETAIL_WORD_STYLE = "bold cyan"


@dataclass(frozen=True)
class _WordLocation:
    """A selectable word's token index and wrapped display position."""

    token_index: int
    row: int
    column: int


_WORD_LINK_PREFIX = "stressmark-word:"


@dataclass(frozen=True)
class _ModeDefinition:
    attribute: str
    label: str
    description: str

    @property
    def prompt(self):
        return f"{self.label} — {self.description}"


_MODE_DEFINITIONS = (
    _ModeDefinition(
        "nuclear_only",
        "Nuclear only",
        "retain only each phrase's nuclear peak",
    ),
    _ModeDefinition(
        "show_rules",
        "Explain",
        "show stress-rule annotations",
    ),
    _ModeDefinition(
        "flag_heteronyms",
        "Flag heteronyms",
        "mark POS-dependent stress words",
    ),
)


class ModeScreen(ModalScreen[frozenset[str] | None]):
    """Popup for editing the rendering modes of the current TUI session."""

    CSS = f"""
    ModeScreen {{
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }}

    #mode-dialog {{
        width: 68;
        max-width: 92%;
        height: 12;
        padding: 1 2;
        border: round {render.DARK_THEME.primary};
        background: {render.DARK_THEME.surface};
        color: {render.DARK_THEME.foreground};
    }}

    #mode-title {{
        height: 1;
        text-style: bold;
        color: {render.DARK_THEME.nuclear};
    }}

    #mode-list {{
        height: 5;
        background: {render.DARK_THEME.surface};
        border: none;
    }}

    #mode-status {{
        height: 1;
        margin-top: 1;
        color: {render.DARK_THEME.pre_nuclear};
    }}

    #mode-help {{
        height: 1;
        margin-top: 1;
        color: {render.DARK_THEME.unstressed};
    }}
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("j", "cursor_down", show=False, priority=True),
        Binding("k", "cursor_up", show=False, priority=True),
        Binding("enter", "apply", show=False, priority=True),
        Binding("escape,q", "cancel", show=False, priority=True),
    ]

    def __init__(self, enabled_modes):
        super().__init__()
        self.enabled_modes = frozenset(enabled_modes)

    def compose(self) -> ComposeResult:
        selections = [
            Selection(
                mode.prompt,
                mode.attribute,
                mode.attribute in self.enabled_modes,
            )
            for mode in _MODE_DEFINITIONS
        ]
        with Vertical(id="mode-dialog"):
            yield Static("Viewer modes", id="mode-title")
            yield Static(id="mode-status")
            yield SelectionList(*selections, id="mode-list")
            yield Static(
                "↑/k ↓/j move   Space toggle   Enter apply   Esc cancel",
                id="mode-help",
            )

    def on_mount(self) -> None:
        self.query_one("#mode-list", SelectionList).focus()
        self._refresh_status()

    def on_selection_list_selected_changed(
        self,
        _event: SelectionList.SelectedChanged,
    ) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        selected = set(self.query_one("#mode-list", SelectionList).selected)
        labels = [
            mode.label
            for mode in _MODE_DEFINITIONS
            if mode.attribute in selected
        ]
        summary = ", ".join(labels) if labels else "none"
        self.query_one("#mode-status", Static).update(f"Enabled: {summary}")

    def action_cursor_down(self) -> None:
        self.query_one("#mode-list", SelectionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#mode-list", SelectionList).action_cursor_up()

    def action_apply(self) -> None:
        selected = self.query_one("#mode-list", SelectionList).selected
        self.dismiss(frozenset(selected))

    def action_cancel(self) -> None:
        self.dismiss(None)


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
        scrollbar-color: {render.DARK_THEME.primary};
        scrollbar-color-hover: {render.DARK_THEME.nuclear};
        scrollbar-background: {render.DARK_THEME.chrome};
    }}

    #document-content {{
        width: 1fr;
        height: auto;
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
        Binding("left,h,b", "previous_word", "word", key_display="←/h/b"),
        Binding("right,l,w", "next_word", "word", key_display="→/l/w"),
        Binding("up,k", "previous_line", "line", key_display="↑/k"),
        Binding("down,j", "next_line", "line", key_display="↓/j"),
        Binding("home", "first_word", "Home first", show=False),
        Binding("end", "last_word", "End last", show=False),
        Binding("m", "show_modes", "modes", show=True),
        Binding("escape", "clear_document", "clear", show=True),
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
        initial_text="",
        *,
        analyzer=analyze,
        nuclear_only=False,
        show_rules=False,
        flag_heteronyms=False,
        source_name=None,
    ):
        super().__init__()
        self._analyzer = analyzer
        self.nuclear_only = nuclear_only
        self.show_rules = show_rules
        self.flag_heteronyms = flag_heteronyms
        self.source_text = ""
        self.raw_tokens = []
        self.display_results = []
        self.lexical_results = []
        self.word_ranges = []
        self.base_text = Text()
        self.selected_index = -1
        self._preferred_column = None
        self._word_location_cache = {}
        self._set_document(initial_text)
        if source_name:
            self.sub_title = source_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(
            Static(id="document-content"),
            id="document",
            can_focus=False,
        )
        yield Static(id="word-detail")
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        self._refresh_selection()

    def on_paste(self, event: events.Paste) -> None:
        """Replace the current document with one bracketed-paste payload."""
        event.stop()
        if event.text:
            self._replace_document(event.text, subtitle="pasted text")

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

    def action_clear_document(self) -> None:
        """Clear the viewer while keeping it ready for another paste."""
        self._replace_document("", subtitle=self.SUB_TITLE)

    def action_show_modes(self) -> None:
        """Open the session-mode popup."""
        enabled_modes = frozenset(
            mode.attribute
            for mode in _MODE_DEFINITIONS
            if getattr(self, mode.attribute)
        )
        self.push_screen(ModeScreen(enabled_modes), self._apply_modes)

    def _apply_modes(self, enabled_modes) -> None:
        """Apply popup choices through the normal document-rendering path."""
        if enabled_modes is None:
            return

        previous_modes = {
            mode.attribute: getattr(self, mode.attribute)
            for mode in _MODE_DEFINITIONS
        }
        next_modes = {
            mode.attribute: mode.attribute in enabled_modes
            for mode in _MODE_DEFINITIONS
        }
        if next_modes == previous_modes:
            return

        selected_index = self.selected_index
        for attribute, enabled in next_modes.items():
            setattr(self, attribute, enabled)
        try:
            self._set_document(self.source_text)
        except Exception:
            for attribute, enabled in previous_modes.items():
                setattr(self, attribute, enabled)
            raise

        if self.word_ranges and selected_index >= 0:
            self.selected_index = min(selected_index, len(self.word_ranges) - 1)
        self._refresh_selection()

    def _analyze_document(self, text):
        """Return display and lexical analyses for a TUI document."""
        analysis = analyze_for_viewer(
            text,
            nuclear_only=self.nuclear_only,
            analyzer=self._analyzer,
        )
        return (
            analysis.raw_tokens,
            analysis.display_results,
            analysis.lexical_results,
        )

    def _set_document(self, text):
        """Build and install document state without touching mounted widgets."""
        if text:
            raw_tokens, display_results, lexical_results = self._analyze_document(text)
        else:
            raw_tokens, display_results, lexical_results = [], [], []

        word_ranges = []
        base_text = render._render_rich_text(
            raw_tokens,
            display_results,
            self.show_rules,
            self.flag_heteronyms,
            word_ranges=word_ranges,
        )

        self.source_text = text
        self.raw_tokens = raw_tokens
        self.display_results = display_results
        self.lexical_results = lexical_results
        self.word_ranges = word_ranges
        self.base_text = base_text
        self.selected_index = 0 if word_ranges else -1
        self._preferred_column = None
        self._word_location_cache.clear()

    def _replace_document(self, text, *, subtitle):
        """Analyze replacement text and reset the mounted viewer atomically."""
        self._set_document(text)
        self.sub_title = subtitle
        self._refresh_selection()
        self.query_one("#document", VerticalScroll).scroll_home(
            immediate=True,
            force=True,
        )

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

        content = self.query_one("#document-content", Static)
        word_locations = self._get_word_locations(content)
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

    def _get_word_locations(self, content):
        """Return wrapped locations, recomputing them only after width changes."""
        width = max(content.content_region.width, 1)
        if width not in self._word_location_cache:
            self._word_location_cache[width] = _wrapped_word_locations(
                self.base_text,
                self.word_ranges,
                width,
                self.console,
            )
        return self._word_location_cache[width]

    def _refresh_selection(self) -> None:
        document = self.query_one("#document", VerticalScroll)
        content = self.query_one("#document-content", Static)
        detail = self.query_one("#word-detail", Static)

        if self.selected_index < 0:
            content.update(self.base_text)
            message = (
                "No selectable words in this document. Esc clears the viewer."
                if self.source_text
                else "Paste text anywhere to analyze it. Esc clears the viewer."
            )
            detail.update(Text(message, style=_DETAIL_META_STYLE))
            return

        token_index, start, end = self.word_ranges[self.selected_index]
        selected_text = self.base_text.copy()
        selected_text.stylize(self._SELECTION_STYLE, start, end)
        content.update(selected_text)

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

        word_locations = self._get_word_locations(content)
        display_row = word_locations[self.selected_index].row
        document.scroll_to(y=max(0, display_row - 2), immediate=True, force=True)


def run_tui(
    initial_text="",
    *,
    nuclear_only=False,
    show_rules=False,
    flag_heteronyms=False,
    source_name=None,
):
    """Run the interactive stress viewer until the user quits."""
    StressmarkApp(
        initial_text,
        nuclear_only=nuclear_only,
        show_rules=show_rules,
        flag_heteronyms=flag_heteronyms,
        source_name=source_name,
    ).run()
