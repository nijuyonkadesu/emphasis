import asyncio
import sys
from types import SimpleNamespace

import pytest

from stressmark import cli
from stressmark.tui import StressmarkApp


def _result(raw, *, primary=0, tier=None, secondary=None):
    return SimpleNamespace(
        raw=raw,
        cls="content",
        tier=tier,
        syllables=[raw],
        primary=primary,
        secondary=set() if secondary is None else secondary,
        confidence="dict",
        rule=None,
        tag="NN",
    )


def _separator(raw):
    return SimpleNamespace(
        raw=raw,
        cls="punct",
        tier=None,
        syllables=[],
        primary=-1,
        secondary=set(),
        confidence="dict",
        rule=None,
        tag=None,
    )


def test_tui_vim_and_arrow_navigation_and_raw_stress_detail():
    raw_tokens = [
        (True, "Alpha"),
        (False, " "),
        (True, "beta"),
        (False, "\n"),
        (True, "One"),
        (False, " "),
        (True, "gamma"),
    ]
    display_results = [
        _result("Alpha", primary=-1, secondary={0}),
        _separator(" "),
        _result("beta", primary=0, tier="nuclear"),
        _separator("\n"),
        _result("One", primary=-1, secondary={0}),
        _separator(" "),
        _result("gamma", primary=0, tier="nuclear"),
    ]
    lexical_results = [
        _result("Alpha"),
        _separator(" "),
        _result("beta"),
        _separator("\n"),
        _result("One"),
        _separator(" "),
        _result("gamma"),
    ]
    app = StressmarkApp(raw_tokens, display_results, lexical_results)

    async def exercise():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            detail = app.query_one("#word-detail")
            assert "Raw stressmark: ALPHA" in detail.render().plain

            await pilot.press("l")
            assert app.selected_index == 1

            # From beta at column 6, j/down selects the nearest column on the
            # following source line: gamma at column 4, not One at column 0.
            await pilot.press("j")
            assert app.selected_index == 3
            assert "gamma" in detail.render().plain

            await pilot.press("left")
            assert app.selected_index == 2

            await pilot.press("up")
            assert app.selected_index == 0

            await pilot.press("end")
            assert app.selected_index == 3
            await pilot.press("home")
            assert app.selected_index == 0

    asyncio.run(exercise())


def test_tui_line_navigation_moves_between_wrapped_visual_rows():
    words = ["Alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    raw_tokens = []
    results = []
    for index, word in enumerate(words):
        if index:
            raw_tokens.append((False, " "))
            results.append(_separator(" "))
        raw_tokens.append((True, word))
        results.append(_result(word))

    app = StressmarkApp(raw_tokens, results, results)

    async def exercise():
        async with app.run_test(size=(24, 16)) as pilot:
            await pilot.pause()

            # The document has 20 content columns after horizontal padding:
            #   Alpha beta gamma
            #   delta epsilon zeta
            #   eta theta
            # j/k should traverse those displayed rows even though the source
            # contains no newline, retaining the preferred visual column.
            await pilot.press("j")
            assert app.selected_index == 3

            await pilot.press("down")
            assert app.selected_index == 6

            await pilot.press("k")
            assert app.selected_index == 3

            await pilot.press("up")
            assert app.selected_index == 0

    asyncio.run(exercise())


def test_tui_handles_a_document_with_no_words():
    app = StressmarkApp(
        [(False, "...\n")], [_separator("...\n")], [_separator("...\n")]
    )

    async def exercise():
        async with app.run_test(size=(60, 16)) as pilot:
            await pilot.press("h", "j", "right", "up")
            assert app.selected_index == -1
            assert "No selectable words" in app.query_one("#word-detail").render().plain

    asyncio.run(exercise())


def test_cli_tui_keeps_nuclear_display_and_raw_lexical_analyses_separate(
    monkeypatch, tmp_path
):
    input_path = tmp_path / "input.txt"
    input_path.write_text("Alpha", encoding="utf-8")
    raw_tokens = [(True, "Alpha")]
    calls = []

    def fake_analyze(text, nuclear_only=False):
        calls.append((text, nuclear_only))
        primary = -1 if nuclear_only else 0
        return raw_tokens, [_result("Alpha", primary=primary)]

    captured = {}

    def fake_run_tui(raw, display, lexical, **options):
        captured.update(
            raw=raw,
            display=display,
            lexical=lexical,
            options=options,
        )

    monkeypatch.setattr(cli, "analyze", fake_analyze)
    monkeypatch.setattr("stressmark.tui.run_tui", fake_run_tui)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stressmark", str(input_path), "--tui", "--nuclear-only", "--explain"],
    )

    cli.main()

    assert calls == [("Alpha", True), ("Alpha", False)]
    assert captured["display"][0].primary == -1
    assert captured["lexical"][0].primary == 0
    assert captured["options"]["show_rules"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--tui"],
        ["input.txt", "--tui", "-o", "output.txt"],
        ["input.txt", "--tui", "--format", "pdf"],
    ],
)
def test_cli_rejects_incompatible_tui_inputs(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["stressmark", *arguments])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
