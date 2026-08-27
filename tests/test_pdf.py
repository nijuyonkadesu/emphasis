import sys
from types import SimpleNamespace

from stressmark import render
from stressmark import cli


def _result(**overrides):
    values = {
        "cls": "content",
        "tier": None,
        "syllables": ["hel", "lo"],
        "primary": 0,
        "secondary": set(),
        "confidence": "dict",
        "rule": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pdf_uses_the_exact_terminal_text_and_style_source():
    raw_tokens = [(True, "Hello"), (False, " "), (True, "again")]
    results = [
        _result(tier="nuclear"),
        _result(),
        _result(syllables=["a", "gain"], primary=1, secondary={0}),
    ]

    text = render._render_rich_text(raw_tokens, results)

    assert text.plain == "HELlo aGAIN"
    assert [span.style for span in text.spans] == [
        "bold reverse yellow",
        "grey62",
        "underline yellow3",
        "bold yellow",
    ]


def test_render_pdf_returns_a_complete_pdf_with_an_embedded_font():
    raw_tokens = [(True, "Unknown")]
    results = [
        _result(
            syllables=["un", "known"],
            primary=1,
            confidence="predicted",
        )
    ]

    output = render.render_pdf(raw_tokens, results)

    assert output.startswith(b"%PDF-")
    assert output.rstrip().endswith(b"%%EOF")
    if render._pdf_fonts()[0] == "StressmarkMono":
        assert b"/FontFile2" in output


def test_cli_writes_pdf_through_the_existing_stressmark_interface(monkeypatch, tmp_path):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.pdf"
    input_path.write_text("Hello", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "analyze",
        lambda text, nuclear_only=False: (
            [(True, text)],
            [_result(tier="nuclear")],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["stressmark", str(input_path), "--format", "pdf", "-o", str(output_path)],
    )

    cli.main()

    assert output_path.read_bytes().startswith(b"%PDF-")
