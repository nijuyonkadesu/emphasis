"""
stressmark.render
Four output formats for analyzed text: terminal (rich), PDF, HTML, JSON.
"""
from functools import lru_cache
from io import BytesIO
import json
import html as htmlmod
import os
from pathlib import Path

CONF_LABELS = {
    "dict": "dictionary",
    "dict-pos-resolved": "dictionary (resolved by part-of-speech)",
    "dict-flagged": "dictionary -- ambiguous, unresolved",
    "predicted": "predicted (word not in dictionary)",
    "rule-9": "compound-adjective rule",
    "reducible": "function word (reduced)",
}


def _confidence_marker(conf):
    return {
        "dict": "",
        "dict-pos-resolved": "",
        "dict-flagged": "\u26a0",   # ⚠
        "predicted": "\u2248",      # ≈ (approximately)
        "rule-9": "",
    }.get(conf, "")


# ---------------------------------------------------------------------------
# Terminal (rich)
# ---------------------------------------------------------------------------

def _render_rich_text(raw_tokens, results, show_rules=False, flag_heteronyms=False):
    """Build the styled document shared by the terminal and PDF renderers."""
    from rich.text import Text

    text = Text()

    for tok, res in zip(raw_tokens, results):
        isw, txt = tok
        if not isw:
            text.append(txt)
            continue

        if res.cls == "reducible":
            text.append(txt.lower(), style="dim")
            continue
        if res.tier in ("given", "suppressed"):
            text.append(txt.lower(), style="dim")
            continue

        sylls = res.syllables
        conf = res.confidence
        for i, s in enumerate(sylls):
            if s == "-":
                text.append("-")
                continue
            if i == res.primary:
                if res.tier == "nuclear":
                    style = "bold reverse yellow"
                elif res.tier == "prominent":
                    style = "bold yellow"
                elif res.tier == "pre-nuclear":
                    style = "bold yellow3"
                elif conf == "predicted":
                    style = "bold italic yellow3"
                elif conf == "dict-flagged":
                    style = "bold orange3"
                else:
                    style = "bold yellow"
                text.append(s.upper(), style=style)
            elif i in res.secondary:
                text.append(s.lower(), style="underline yellow3")
            else:
                text.append(s.lower(), style="grey62")
        marker = _confidence_marker(conf)
        if marker:
            text.append(marker, style="bold red" if conf == "dict-flagged" else "yellow3")
        if show_rules and res.rule:
            text.append(f"[R{res.rule}]", style="dim cyan")
        if flag_heteronyms and txt.lower() in _HETERONYM_WORDS:
            text.append("\u26a0HET", style="bold magenta")

    return text


def render_terminal(raw_tokens, results, show_rules=False, flag_heteronyms=False, console=None):
    from rich.console import Console

    owns_console = console is None
    if console is None:
        console = Console()
    text = _render_rich_text(raw_tokens, results, show_rules, flag_heteronyms)

    console.print(text)
    if owns_console:
        console.print()
        console.print(
            "[bold reverse yellow]REVERSE[/bold reverse yellow]=nuclear  "
            "[bold yellow]CAPS[/bold yellow]=prominent  "
            "[bold yellow3]CAPS[/bold yellow3]=pre-nuclear  "
            "[underline]underline[/underline]=secondary  "
            "[dim]dim[/dim]=given/reduced  "
            "[yellow3]\u2248[/yellow3]=predicted  [red]\u26a0[/red]=ambiguous",
            style="dim",
        )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

PDF_COLUMNS = 100
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

    text = _render_rich_text(raw_tokens, results, show_rules, flag_heteronyms)
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
    pdf.setTitle("stressmark output")
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
    if result.cls == "reducible":
        text.append(result.raw.lower(), style="dim")
        return text

    sylls = result.syllables
    conf = result.confidence
    for i, s in enumerate(sylls):
        if s == "-":
            text.append("-")
            continue
        if i == result.primary:
            if conf == "predicted":
                style = "bold italic yellow3"
            elif conf == "dict-flagged":
                style = "bold orange3"
            else:
                style = "bold yellow"
            text.append(s.upper(), style=style)
        elif i in result.secondary:
            text.append(s.lower(), style="underline yellow3")
        else:
            text.append(s.lower(), style="grey62")

    marker = _confidence_marker(conf)
    if marker:
        text.append(marker, style="bold red" if conf == "dict-flagged" else "yellow3")

    return text


_HETERONYM_WORDS = set()


def set_heteronym_words(words):
    global _HETERONYM_WORDS
    _HETERONYM_WORDS = set(words)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>stressmark output</title>
<style>
  body{{ background:#15140F; color:#EDE8DB; font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace;
        font-size:16px; line-height:2.1; padding:32px; max-width:840px; margin:0 auto; }}
  .unstressed{{ color:#8C8674; }}
  .reduced{{ color:#5C5848; }}
  .primary{{ color:#E7A23B; font-weight:600; border-bottom:2px solid #E7A23B; }}
  .prominent{{ color:#E7A23B; font-weight:700; border-bottom:2px solid #E7A23B; }}
  .pre-nuclear{{ color:#D4A843; font-weight:600; border-bottom:2px solid #D4A843; opacity:0.9; }}
  .secondary{{ color:#E7A23B; opacity:0.75; border-bottom:1px dotted #E7A23B; }}
  .nuclear{{ color:#FFC25C; font-weight:700; border-bottom:3px solid #FFC25C;
            text-shadow:0 0 14px rgba(255,194,92,.45); }}
  .predicted{{ border-bottom-style:dashed !important; opacity:0.85; }}
  .flagged{{ color:#FF8A5C !important; }}
  .mark{{ font-size:.8em; opacity:.7; }}
  .rule{{ font-size:.7em; color:#6FA8A0; vertical-align:super; }}
  .legend{{ margin-top:28px; padding-top:16px; border-top:1px solid #36321F; color:#8C8674; font-size:13px; }}
  .legend span{{ margin-right:18px; }}
</style></head><body>
<div class="output">{body}</div>
<div class="legend">
  <span class="nuclear">CAPS</span>nuclear (IP focus) &nbsp;
  <span class="prominent">CAPS</span>prominent (contrast/wh/early) &nbsp;
  <span class="pre-nuclear">CAPS</span>pre-nuclear (new info) &nbsp;
  <span class="secondary">low</span>secondary &nbsp;
  <span class="reduced">low</span>given/reduced &nbsp;
  <span class="predicted">dashed</span>predicted &nbsp;
  <span class="flagged">orange</span>ambiguous
</div>
</body></html>
"""


def render_html(raw_tokens, results, show_rules=False):
    parts = []
    for tok, res in zip(raw_tokens, results):
        isw, txt = tok
        if not isw:
            parts.append(htmlmod.escape(txt))
            continue
        if res.cls == "reducible" or res.tier in ("given", "suppressed"):
            parts.append(f'<span class="reduced">{htmlmod.escape(txt.lower())}</span>')
            continue
        sylls = res.syllables
        conf = res.confidence
        word_html = ""
        for i, s in enumerate(sylls):
            if s == "-":
                word_html += "-"
                continue
            if i == res.primary:
                if res.tier == "nuclear":
                    cls = "nuclear"
                elif res.tier == "prominent":
                    cls = "prominent"
                elif res.tier == "pre-nuclear":
                    cls = "pre-nuclear"
                else:
                    cls = "primary"
                if conf == "predicted":
                    cls += " predicted"
                if conf == "dict-flagged":
                    cls += " flagged"
                word_html += f'<span class="{cls}">{htmlmod.escape(s.upper())}</span>'
            elif i in res.secondary:
                word_html += f'<span class="secondary">{htmlmod.escape(s.lower())}</span>'
            else:
                word_html += f'<span class="unstressed">{htmlmod.escape(s.lower())}</span>'
        marker = _confidence_marker(conf)
        if marker:
            word_html += f'<span class="mark">{marker}</span>'
        if show_rules and res.rule:
            word_html += f'<span class="rule">R{res.rule}</span>'
        parts.append(word_html)
    return HTML_TEMPLATE.format(body="".join(parts))


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
            "confidence_label": CONF_LABELS.get(res.confidence, res.confidence),
        }
        if res.cls != "reducible":
            entry["syllables"] = [
                {
                    "text": s,
                    "stress": ("primary" if i == res.primary else
                               "secondary" if i in res.secondary else "none"),
                }
                for i, s in enumerate(res.syllables) if s != "-"
            ]
            if res.rule:
                entry["rule"] = res.rule
        out.append(entry)
    return json.dumps(out, indent=2, ensure_ascii=False)
