"""
stressmark.render
Four output formats for analyzed text: terminal (rich), PDF, HTML, JSON.
"""
import html as htmlmod
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from stressmark.model import (
    APP_NAME,
    Confidence,
    DiscourseKeyType,
    ProminenceTier,
    StressLevel,
    WordClass,
)


class VisualRole(StrEnum):
    NUCLEAR = ProminenceTier.NUCLEAR
    PROMINENT = ProminenceTier.PROMINENT
    PRE_NUCLEAR = ProminenceTier.PRE_NUCLEAR
    SECONDARY = StressLevel.SECONDARY
    BACKGROUNDED = "backgrounded"
    PREDICTED_MARKER = "predicted-marker"
    AMBIGUOUS_MARKER = "ambiguous-marker"
    PRIMARY = StressLevel.PRIMARY
    PREDICTED_PRIMARY = "predicted-primary"
    AMBIGUOUS_PRIMARY = "ambiguous-primary"
    UNSTRESSED = "unstressed"
    RULE = "rule"
    HETERONYM = DiscourseKeyType.HETERONYM
    LEGEND_DESCRIPTION = "legend-description"
    DETAIL_WORD = "detail-word"


@dataclass(frozen=True)
class ThemePalette:
    background: str
    foreground: str
    primary: str
    pre_nuclear: str
    nuclear: str
    nuclear_glow: str
    unstressed: str
    backgrounded: str
    ambiguous: str
    rule: str
    heteronym: str
    separator: str
    surface: str
    chrome: str
    selection: str


DARK_THEME = ThemePalette(
    background="#15140F",
    foreground="#EDE8DB",
    primary="#E7A23B",
    pre_nuclear="#D4A843",
    nuclear="#FFC25C",
    nuclear_glow="rgba(255,194,92,.45)",
    unstressed="#8C8674",
    backgrounded="#5C5848",
    ambiguous="#FF8A5C",
    rule="#6FA8A0",
    heteronym="#D783FF",
    separator="#36321F",
    surface="#201E16",
    chrome="#252217",
    selection="#006B78",
)


@dataclass(frozen=True)
class VisualDefinition:
    """Presentation metadata for one semantic role in every styled medium."""

    name: VisualRole
    rich_style: str
    html_style: str = ""
    legend_sample: str | None = None
    legend_meaning: str | None = None
    marker: str | None = None

    @property
    def legend_text(self):
        return self.legend_sample or self.marker


# Tuple order is legend order. All presentation metadata for a semantic role
# is declared once here; lookup tables and legends are derived below.
_VISUAL_DEFINITIONS = (
    VisualDefinition(
        VisualRole.NUCLEAR,
        "bold reverse yellow",
        f"color:{DARK_THEME.nuclear}; font-weight:700; "
        f"border-bottom:3px solid {DARK_THEME.nuclear}; "
        f"text-shadow:0 0 14px {DARK_THEME.nuclear_glow}",
        "REVERSE",
        ProminenceTier.NUCLEAR,
    ),
    VisualDefinition(
        VisualRole.PROMINENT,
        "bold yellow",
        f"color:{DARK_THEME.primary}; font-weight:700; "
        f"border-bottom:2px solid {DARK_THEME.primary}",
        "CAPS",
        ProminenceTier.PROMINENT,
    ),
    VisualDefinition(
        VisualRole.PRE_NUCLEAR,
        "bold yellow3",
        f"color:{DARK_THEME.pre_nuclear}; font-weight:600; "
        f"border-bottom:2px solid {DARK_THEME.pre_nuclear}; "
        "opacity:.9",
        "CAPS",
        ProminenceTier.PRE_NUCLEAR,
    ),
    VisualDefinition(
        VisualRole.SECONDARY,
        "underline yellow3",
        f"color:{DARK_THEME.primary}; opacity:.75; "
        f"border-bottom:1px dotted {DARK_THEME.primary}",
        "underline",
        StressLevel.SECONDARY,
    ),
    VisualDefinition(
        VisualRole.BACKGROUNDED,
        "dim",
        f"color:{DARK_THEME.backgrounded}",
        "dim",
        "given/reduced/suppressed",
    ),
    VisualDefinition(
        VisualRole.PREDICTED_MARKER,
        "yellow3",
        f"color:{DARK_THEME.pre_nuclear}; font-size:.8em; opacity:.85",
        legend_meaning=Confidence.PREDICTED,
        marker="≈",
    ),
    VisualDefinition(
        VisualRole.AMBIGUOUS_MARKER,
        "bold red",
        f"color:{DARK_THEME.ambiguous}; font-size:.8em; font-weight:700",
        legend_meaning="ambiguous",
        marker="⚠",
    ),
    VisualDefinition(
        VisualRole.PRIMARY,
        "bold yellow",
        f"color:{DARK_THEME.primary}; font-weight:600; "
        f"border-bottom:2px solid {DARK_THEME.primary}",
    ),
    # In HTML these two are modifiers layered on the tier/primary class, which
    # preserves its existing visual composition. In Rich they are the complete
    # untiered primary style.
    VisualDefinition(
        VisualRole.PREDICTED_PRIMARY,
        "bold italic yellow3",
        "border-bottom-style:dashed !important; opacity:.85",
    ),
    VisualDefinition(
        VisualRole.AMBIGUOUS_PRIMARY,
        "bold orange3",
        f"color:{DARK_THEME.ambiguous} !important",
    ),
    VisualDefinition(
        VisualRole.UNSTRESSED,
        "grey62",
        f"color:{DARK_THEME.unstressed}",
    ),
    VisualDefinition(
        VisualRole.RULE,
        "dim cyan",
        f"font-size:.7em; color:{DARK_THEME.rule}; vertical-align:super",
    ),
    VisualDefinition(
        VisualRole.HETERONYM,
        "bold magenta",
        f"color:{DARK_THEME.heteronym}; font-size:.8em; font-weight:700",
        marker="⚠HET",
    ),
    VisualDefinition(VisualRole.LEGEND_DESCRIPTION, "dim"),
    VisualDefinition(VisualRole.DETAIL_WORD, "bold cyan"),
)

_VISUALS = {visual.name: visual for visual in _VISUAL_DEFINITIONS}
if len(_VISUALS) != len(_VISUAL_DEFINITIONS) or set(_VISUALS) != set(VisualRole):
    raise ValueError("Visual definitions must cover each visual role exactly once")

_LEGEND_VISUALS = tuple(
    visual
    for visual in _VISUAL_DEFINITIONS
    if visual.legend_text is not None and visual.legend_meaning is not None
)

_TIER_PRIMARY_VISUALS = {
    ProminenceTier.NUCLEAR: VisualRole.NUCLEAR,
    ProminenceTier.PROMINENT: VisualRole.PROMINENT,
    ProminenceTier.PRE_NUCLEAR: VisualRole.PRE_NUCLEAR,
}


@dataclass(frozen=True)
class ConfidenceDefinition:
    """Human label and visual roles for one engine confidence value."""

    label: str
    primary_visual: VisualRole = VisualRole.PRIMARY
    marker_visual: VisualRole | None = None


_CONFIDENCE_DEFINITIONS = {
    Confidence.DICTIONARY: ConfidenceDefinition("dictionary"),
    Confidence.POS_RESOLVED: ConfidenceDefinition(
        "dictionary (resolved by part-of-speech)"
    ),
    Confidence.AMBIGUOUS: ConfidenceDefinition(
        "dictionary -- ambiguous, unresolved",
        VisualRole.AMBIGUOUS_PRIMARY,
        VisualRole.AMBIGUOUS_MARKER,
    ),
    Confidence.PREDICTED: ConfidenceDefinition(
        "predicted (word not in dictionary)",
        VisualRole.PREDICTED_PRIMARY,
        VisualRole.PREDICTED_MARKER,
    ),
    Confidence.COMPOUND_ADJECTIVE_RULE: ConfidenceDefinition(
        "compound-adjective rule"
    ),
    Confidence.REDUCIBLE: ConfidenceDefinition("function word (reduced)"),
}
if set(_CONFIDENCE_DEFINITIONS) != set(Confidence):
    raise ValueError("Confidence presentation definitions are incomplete")

# Backward-compatible public view, derived rather than maintained separately.
CONF_LABELS = {
    confidence.value: definition.label
    for confidence, definition in _CONFIDENCE_DEFINITIONS.items()
}

for confidence, definition in _CONFIDENCE_DEFINITIONS.items():
    referenced_visuals = (definition.primary_visual, definition.marker_visual)
    for visual_name in referenced_visuals:
        if visual_name is not None and visual_name not in _VISUALS:
            raise ValueError(
                f"Confidence {confidence!r} references unknown visual {visual_name!r}"
            )

_OUTPUT_TITLE = f"{APP_NAME} output"


def confidence_label(confidence):
    definition = _CONFIDENCE_DEFINITIONS.get(confidence)
    return confidence if definition is None else definition.label


def rich_style(visual_name):
    return _VISUALS[visual_name].rich_style


def _confidence_definition(confidence):
    return _CONFIDENCE_DEFINITIONS.get(
        confidence,
        ConfidenceDefinition(str(confidence)),
    )


def _primary_rich_style(tier, confidence):
    visual_name = _TIER_PRIMARY_VISUALS.get(tier)
    if visual_name is None:
        visual_name = _confidence_definition(confidence).primary_visual
    return rich_style(visual_name)


def _marker_visual(visual_name):
    visual = _VISUALS[visual_name]
    if visual.marker is None:
        raise ValueError(f"Visual {visual_name!r} has no marker")
    return visual


def _confidence_marker_visual(confidence):
    visual_name = _confidence_definition(confidence).marker_visual
    return None if visual_name is None else _marker_visual(visual_name)


def _append_confidence_marker(text, confidence):
    visual = _confidence_marker_visual(confidence)
    if visual is not None:
        text.append(visual.marker, style=visual.rich_style)


def _rule_marker(rule):
    return f"[R{rule}]"


# ---------------------------------------------------------------------------
# Terminal (rich)
# ---------------------------------------------------------------------------

def _render_rich_text(
    raw_tokens,
    results,
    show_rules=False,
    flag_heteronyms=False,
    word_ranges=None,
):
    """Build the styled document shared by the terminal, PDF, and TUI.

    When ``word_ranges`` is supplied, append ``(token_index, start, end)``
    entries for every word. This small hook lets interactive callers overlay
    a cursor without reimplementing any of the renderer's styling rules.
    """
    from rich.text import Text

    text = Text()

    for token_index, (tok, res) in enumerate(zip(raw_tokens, results)):
        isw, txt = tok
        if not isw:
            text.append(txt)
            continue

        start = len(text)

        if res.cls == WordClass.REDUCIBLE:
            text.append(txt.lower(), style=rich_style(VisualRole.BACKGROUNDED))
            if word_ranges is not None:
                word_ranges.append((token_index, start, len(text)))
            continue
        if res.tier in (ProminenceTier.GIVEN, ProminenceTier.SUPPRESSED):
            text.append(txt.lower(), style=rich_style(VisualRole.BACKGROUNDED))
            if word_ranges is not None:
                word_ranges.append((token_index, start, len(text)))
            continue

        sylls = res.syllables
        conf = res.confidence
        for i, s in enumerate(sylls):
            if s == "-":
                text.append("-")
                continue
            if i == res.primary:
                text.append(s.upper(), style=_primary_rich_style(res.tier, conf))
            elif i in res.secondary:
                text.append(s.lower(), style=rich_style(VisualRole.SECONDARY))
            else:
                text.append(s.lower(), style=rich_style(VisualRole.UNSTRESSED))
        _append_confidence_marker(text, conf)
        if show_rules and res.rule:
            text.append(_rule_marker(res.rule), style=rich_style(VisualRole.RULE))
        if flag_heteronyms and getattr(res, "is_heteronym", False):
            visual = _marker_visual(VisualRole.HETERONYM)
            text.append(visual.marker, style=visual.rich_style)
        if word_ranges is not None:
            word_ranges.append((token_index, start, len(text)))

    return text


def _render_legend_text():
    """Render the legend from the semantic Rich visual registry."""
    from rich.text import Text

    legend = Text()
    for position, visual in enumerate(_LEGEND_VISUALS):
        legend_text = visual.legend_text
        if legend_text is None or visual.legend_meaning is None:
            raise ValueError(f"Rich visual {visual.name!r} has no legend metadata")
        if position:
            legend.append("  ", style=rich_style(VisualRole.LEGEND_DESCRIPTION))
        legend.append(legend_text, style=visual.rich_style)
        legend.append(
            f"={visual.legend_meaning}",
            style=rich_style(VisualRole.LEGEND_DESCRIPTION),
        )
    return legend


def _render_rich_document(
    raw_tokens,
    results,
    show_rules=False,
    flag_heteronyms=False,
    include_legend=False,
):
    """Compose a complete Rich document for terminal-like output media."""
    text = _render_rich_text(raw_tokens, results, show_rules, flag_heteronyms)
    if include_legend:
        if text:
            if not text.plain.endswith("\n"):
                text.append("\n")
            text.append("\n")
        text.append_text(_render_legend_text())
    return text


def render_terminal(raw_tokens, results, show_rules=False, flag_heteronyms=False, console=None):
    from rich.console import Console

    owns_console = console is None
    if console is None:
        console = Console()
    text = _render_rich_document(
        raw_tokens,
        results,
        show_rules,
        flag_heteronyms,
        include_legend=owns_console,
    )
    console.print(text)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = 100
PDF_COLUMNS = OUTPUT_COLUMNS
PDF_FONT_SIZE = 12
PDF_LINE_HEIGHT = 18
PDF_MARGIN = 36


@lru_cache(maxsize=1)
def _pdf_fonts():
    """Register an embeddable monospace family and return its four faces.

    DejaVu Sans Mono is preferred because it contains the ≈ and ⚠ glyphs used
    by the terminal renderer. Common system locations are checked on Linux,
    macOS, and Windows; PDF's built-in Courier family remains a last-resort
    fallback for unusually minimal systems.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    families = (
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf"),
        ),
        (
            Path("/usr/share/fonts/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSansMono-Oblique.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSansMono-BoldOblique.ttf"),
        ),
        (
            Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
            Path("/System/Library/Fonts/Supplemental/Courier New Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Courier New Italic.ttf"),
            Path("/System/Library/Fonts/Supplemental/Courier New Bold Italic.ttf"),
        ),
        (
            windows_fonts / "consola.ttf",
            windows_fonts / "consolab.ttf",
            windows_fonts / "consolai.ttf",
            windows_fonts / "consolaz.ttf",
        ),
    )

    for paths in families:
        if all(path.is_file() for path in paths):
            names = (
                "StressmarkMono",
                "StressmarkMono-Bold",
                "StressmarkMono-Italic",
                "StressmarkMono-BoldItalic",
            )
            for name, path in zip(names, paths):
                if name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(name, str(path)))
            return names

    return ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique")


def _pdf_color(color, theme, default):
    if color is None:
        triplet = default
    else:
        triplet = color.get_truecolor(theme)
    return tuple(channel / 255 for channel in triplet)


def render_pdf(raw_tokens, results, show_rules=False, flag_heteronyms=False):
    """Return a PDF containing the same styled text as ``render_terminal``.

    The PDF uses a fixed 100-column terminal layout on landscape A4 pages and
    embeds the selected monospace font, so its appearance does not depend on
    fonts installed on the machine used to open it.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas
    from rich.console import Console
    from rich.terminal_theme import MONOKAI

    text = _render_rich_document(
        raw_tokens,
        results,
        show_rules,
        flag_heteronyms,
        include_legend=True,
    )
    console = Console(
        width=PDF_COLUMNS,
        color_system="truecolor",
        force_terminal=True,
        legacy_windows=False,
    )
    lines = console.render_lines(
        text,
        console.options.update(width=PDF_COLUMNS),
        pad=False,
    )

    buffer = BytesIO()
    page_size = landscape(A4)
    pdf = canvas.Canvas(buffer, pagesize=page_size, pageCompression=1)
    pdf.setTitle(_OUTPUT_TITLE)
    regular, bold, italic, bold_italic = _pdf_fonts()
    page_width, page_height = page_size
    background = _pdf_color(None, MONOKAI, MONOKAI.background_color)

    def begin_page():
        pdf.setFillColorRGB(*background)
        pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

    begin_page()
    y = page_height - PDF_MARGIN - PDF_FONT_SIZE

    for line_number, line in enumerate(lines):
        if line_number and y < PDF_MARGIN:
            pdf.showPage()
            begin_page()
            y = page_height - PDF_MARGIN - PDF_FONT_SIZE

        x = PDF_MARGIN
        rendered_segments = []
        for segment in line:
            style = segment.style
            is_bold = bool(style and style.bold)
            is_italic = bool(style and style.italic)
            font_name = (
                bold_italic if is_bold and is_italic else
                bold if is_bold else
                italic if is_italic else
                regular
            )
            width = pdfmetrics.stringWidth(segment.text, font_name, PDF_FONT_SIZE)

            text_color = _pdf_color(style.color if style else None, MONOKAI, MONOKAI.foreground_color)
            fill_color = _pdf_color(style.bgcolor if style else None, MONOKAI, MONOKAI.background_color)
            if style and style.reverse:
                text_color, fill_color = fill_color, text_color
            if style and style.dim:
                text_color = tuple((fg + bg) / 2 for fg, bg in zip(text_color, background))

            rendered_segments.append((segment.text, font_name, text_color))
            if style and (style.bgcolor is not None or style.reverse):
                pdf.setFillColorRGB(*fill_color)
                pdf.rect(x, y - 3, width, PDF_LINE_HEIGHT, fill=1, stroke=0)

            if style and style.underline:
                pdf.setStrokeColorRGB(*text_color)
                pdf.setLineWidth(0.7)
                pdf.line(x, y - 2, x + width, y - 2)
            x += width

        pdf_text = pdf.beginText(PDF_MARGIN, y)
        for segment_text, font_name, text_color in rendered_segments:
            pdf_text.setFont(font_name, PDF_FONT_SIZE)
            pdf_text.setFillColorRGB(*text_color)
            pdf_text.textOut(segment_text)
        pdf.drawText(pdf_text)

        y -= PDF_LINE_HEIGHT

    pdf.save()
    return buffer.getvalue()


def render_word(result):
    """Render one WordResult's stress-highlighted syllables as a Rich Text
    object. Unlike render_terminal (which operates on a whole analyzed
    document with a sentence-level nuclear-stress tiering pass), a
    WordResult from resolve_word_by_pos() never has .tier set -- tiering
    only happens in analyze()'s sentence-level pass -- so only
    confidence-based styling applies here, not the tier-based styles
    (nuclear/prominent/pre-nuclear)."""
    from rich.text import Text

    text = Text()
    if result.cls == WordClass.REDUCIBLE:
        text.append(
            result.raw.lower(),
            style=rich_style(VisualRole.BACKGROUNDED),
        )
        return text

    sylls = result.syllables
    conf = result.confidence
    for i, s in enumerate(sylls):
        if s == "-":
            text.append("-")
            continue
        if i == result.primary:
            text.append(s.upper(), style=_primary_rich_style(None, conf))
        elif i in result.secondary:
            text.append(s.lower(), style=rich_style(VisualRole.SECONDARY))
        else:
            text.append(s.lower(), style=rich_style(VisualRole.UNSTRESSED))

    _append_confidence_marker(text, conf)

    return text


def render_word_detail(result, original, word_number, total_words):
    """Render the shared cursor-detail panel used by interactive viewers."""
    from rich.text import Text

    detail = Text()
    meta_style = rich_style(VisualRole.LEGEND_DESCRIPTION)
    detail.append(f"Word {word_number}/{total_words}  ", style=meta_style)
    detail.append(original, style=rich_style(VisualRole.DETAIL_WORD))
    detail.append("\nRaw stressmark: ", style=meta_style)
    detail.append_text(render_word(result))
    detail.append(
        f"   POS: {result.tag or 'unknown'}   "
        f"source: {confidence_label(result.confidence)}",
        style=meta_style,
    )
    return detail


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _html_visual_css():
    """Generate semantic CSS directly from the visual registry."""
    return "\n".join(
        f"  .{visual.name} {{{visual.html_style};}}"
        for visual in _VISUAL_DEFINITIONS
        if visual.html_style
    )


def _html_span(text, *visual_names):
    classes = " ".join(htmlmod.escape(name, quote=True) for name in visual_names)
    return f'<span class="{classes}">{htmlmod.escape(text)}</span>'


def _html_primary_visuals(tier, confidence):
    """Return HTML's base prominence role plus any confidence modifier."""
    base_visual = _TIER_PRIMARY_VISUALS.get(tier, VisualRole.PRIMARY)
    confidence_visual = _confidence_definition(confidence).primary_visual
    if confidence_visual == VisualRole.PRIMARY:
        return (base_visual,)
    return (base_visual, confidence_visual)


def _render_html_legend():
    """Render HTML legend content from the same ordered definitions as Rich."""
    items = []
    for visual in _LEGEND_VISUALS:
        legend_text = visual.legend_text
        if legend_text is None or visual.legend_meaning is None:
            raise ValueError(f"HTML visual {visual.name!r} has no legend metadata")
        sample = _html_span(legend_text, visual.name)
        meaning = htmlmod.escape(visual.legend_meaning)
        items.append(f'<span class="legend-item">{sample}={meaning}</span>')
    return "\n  ".join(items)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{title}</title>
<style>
  body{{ background:{background}; color:{foreground}; font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace;
        font-size:16px; line-height:2.1; padding:32px; max-width:840px; margin:0 auto; }}
{visual_css}
  .legend{{ margin-top:28px; padding-top:16px; border-top:1px solid {separator}; color:{unstressed}; font-size:13px; }}
  .legend-item{{ display:inline-block; margin-right:18px; }}
</style></head><body>
<div class="output">{body}</div>
<div class="legend">
  {legend}
</div>
</body></html>
"""


def render_html(raw_tokens, results, show_rules=False, flag_heteronyms=False):
    parts = []
    for tok, res in zip(raw_tokens, results):
        isw, txt = tok
        if not isw:
            parts.append(htmlmod.escape(txt))
            continue
        if res.cls == WordClass.REDUCIBLE or res.tier in (
            ProminenceTier.GIVEN,
            ProminenceTier.SUPPRESSED,
        ):
            parts.append(_html_span(txt.lower(), VisualRole.BACKGROUNDED))
            continue
        sylls = res.syllables
        conf = res.confidence
        word_html = ""
        for i, s in enumerate(sylls):
            if s == "-":
                word_html += "-"
                continue
            if i == res.primary:
                word_html += _html_span(
                    s.upper(),
                    *_html_primary_visuals(res.tier, conf),
                )
            elif i in res.secondary:
                word_html += _html_span(s.lower(), VisualRole.SECONDARY)
            else:
                word_html += _html_span(s.lower(), VisualRole.UNSTRESSED)
        marker_visual = _confidence_marker_visual(conf)
        if marker_visual is not None:
            word_html += _html_span(marker_visual.marker, marker_visual.name)
        if show_rules and res.rule:
            word_html += _html_span(_rule_marker(res.rule), VisualRole.RULE)
        if flag_heteronyms and getattr(res, "is_heteronym", False):
            visual = _marker_visual(VisualRole.HETERONYM)
            word_html += _html_span(visual.marker, visual.name)
        parts.append(word_html)
    return HTML_TEMPLATE.format(
        title=htmlmod.escape(_OUTPUT_TITLE),
        background=DARK_THEME.background,
        foreground=DARK_THEME.foreground,
        separator=DARK_THEME.separator,
        unstressed=DARK_THEME.unstressed,
        body="".join(parts),
        visual_css=_html_visual_css(),
        legend=_render_html_legend(),
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def render_json(raw_tokens, results):
    out = []
    for tok, res in zip(raw_tokens, results):
        isw, txt = tok
        if not isw:
            out.append({"type": "sep", "text": txt})
            continue
        entry = {
            "type": "word",
            "text": txt,
            "pos": res.tag,
            "class": res.cls,
            "tier": res.tier,
            "confidence": res.confidence,
            "confidence_label": confidence_label(res.confidence),
        }
        if res.cls != WordClass.REDUCIBLE:
            entry["syllables"] = [
                {
                    "text": s,
                    "stress": (
                        StressLevel.PRIMARY
                        if i == res.primary
                        else StressLevel.SECONDARY
                        if i in res.secondary
                        else StressLevel.NONE
                    ),
                }
                for i, s in enumerate(res.syllables) if s != "-"
            ]
            if res.rule:
                entry["rule"] = res.rule
        out.append(entry)
    return json.dumps(out, indent=2, ensure_ascii=False)
