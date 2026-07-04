"""
stressmark.render
Three output formats for analyzed text: terminal (rich), HTML, JSON.
"""
import json
import html as htmlmod

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

def render_terminal(raw_tokens, results, show_rules=False, flag_heteronyms=False, console=None):
    from rich.console import Console
    from rich.text import Text

    owns_console = console is None
    if console is None:
        console = Console()
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
