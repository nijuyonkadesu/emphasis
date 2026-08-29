"""Analysis helpers shared by interactive stressmark viewers."""

from dataclasses import dataclass

from stressmark.engine import analyze


@dataclass(frozen=True)
class ViewerAnalysis:
    raw_tokens: list
    display_results: list
    lexical_results: list


def analyze_for_viewer(text, *, nuclear_only=False, analyzer=analyze):
    """Analyze text for a styled viewer and its lexical-detail display."""
    raw_tokens, display_results = analyzer(text, nuclear_only=nuclear_only)
    lexical_results = display_results
    if nuclear_only:
        lexical_tokens, lexical_results = analyzer(text, nuclear_only=False)
        if lexical_tokens != raw_tokens:
            raise RuntimeError("Viewer analyses produced different token streams")
    return ViewerAnalysis(raw_tokens, display_results, lexical_results)
