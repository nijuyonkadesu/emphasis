import asyncio
import sys
from types import SimpleNamespace

import pytest
from textual import events
from textual.widgets import SelectionList

from stressmark import cli
from stressmark.tui import ModeScreen, StressmarkApp


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


def _analysis_for_words(text, nuclear_only=False):
    """Small deterministic analyzer used by TUI interaction tests."""
    raw_tokens = []
    results = []
    for index, word in enumerate(text.split()):
        if index:
            raw_tokens.append((False, " "))
            results.append(_separator(" "))
        raw_tokens.append((True, word))
        results.append(_result(word, primary=-1 if nuclear_only else 0))
    return raw_tokens, results


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
    def analyzer(text, nuclear_only=False):
        return raw_tokens, display_results if nuclear_only else lexical_results

    app = StressmarkApp(
        "Alpha beta\nOne gamma",
        analyzer=analyzer,
        nuclear_only=True,
    )

    async def exercise():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            detail = app.query_one("#word-detail")
            assert "Raw stressmark: ALPHA" in detail.render().plain

            await pilot.press("w")
            assert app.selected_index == 1

            await pilot.press("b")
            assert app.selected_index == 0

            await pilot.press("right")
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

    app = StressmarkApp(" ".join(words), analyzer=lambda _text, **_options: (raw_tokens, results))

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


def test_tui_starts_empty_and_supports_repeated_paste_and_escape_cycles():
    app = StressmarkApp("", analyzer=_analysis_for_words)

    async def exercise():
        async with app.run_test(size=(60, 16)) as pilot:
            content = app.query_one("#document-content")
            detail = app.query_one("#word-detail")

            assert app.selected_index == -1
            assert content.render().plain == ""
            assert "Paste text" in detail.render().plain

            app.post_message(events.Paste("Alpha beta"))
            await pilot.pause()
            assert app.source_text == "Alpha beta"
            assert content.render().plain == "ALPHA BETA"
            assert app.selected_index == 0

            await pilot.press("escape")
            assert app.source_text == ""
            assert content.render().plain == ""
            assert app.selected_index == -1

            app.post_message(events.Paste("Gamma delta"))
            await pilot.pause()
            assert app.source_text == "Gamma delta"
            assert content.render().plain == "GAMMA DELTA"
            assert app.selected_index == 0

    asyncio.run(exercise())


def test_tui_long_pastes_scroll_after_each_clear_cycle():
    long_text = " ".join(["alpha", "beta", "gamma", "delta"] * 75)
    app = StressmarkApp("", analyzer=_analysis_for_words)

    async def exercise():
        async with app.run_test(size=(32, 16)) as pilot:
            document = app.query_one("#document")
            content = app.query_one("#document-content")

            for _ in range(2):
                app.post_message(events.Paste(long_text))
                await pilot.pause()
                await pilot.press("end")
                await pilot.pause()
                assert app.selected_index == 299
                assert document.max_scroll_y > 20
                assert document.scroll_y > 20

                await pilot.press("escape")
                await pilot.pause()
                assert content.render().plain == ""
                assert document.scroll_y == 0

    asyncio.run(exercise())


def test_tui_pastes_reuse_all_modes_selected_at_launch():
    calls = []

    def analyzer(text, nuclear_only=False):
        calls.append((text, nuclear_only))
        raw_tokens, results = _analysis_for_words(
            text,
            nuclear_only=nuclear_only,
        )
        for result in results:
            if result.cls == "content":
                result.rule = 1
                result.is_heteronym = True
        return raw_tokens, results

    app = StressmarkApp(
        "",
        analyzer=analyzer,
        nuclear_only=True,
        show_rules=True,
        flag_heteronyms=True,
    )

    async def exercise():
        async with app.run_test(size=(60, 16)) as pilot:
            for text in ("Alpha beta", "Gamma delta"):
                app.post_message(events.Paste(text))
                await pilot.pause()
                assert app.display_results[0].primary == -1
                assert app.lexical_results[0].primary == 0
                assert "[R1]" in app.base_text.plain
                assert "⚠HET" in app.base_text.plain
                await pilot.press("escape")

    asyncio.run(exercise())

    assert calls == [
        ("Alpha beta", True),
        ("Alpha beta", False),
        ("Gamma delta", True),
        ("Gamma delta", False),
    ]
    assert app.show_rules is True
    assert app.flag_heteronyms is True


def test_tui_mode_popup_toggles_modes_and_preserves_document_context():
    calls = []

    def analyzer(text, nuclear_only=False):
        calls.append((text, nuclear_only))
        raw_tokens, results = _analysis_for_words(
            text,
            nuclear_only=nuclear_only,
        )
        for result in results:
            if result.cls == "content":
                result.rule = 1
                result.is_heteronym = True
        return raw_tokens, results

    app = StressmarkApp("Alpha beta", analyzer=analyzer)

    async def exercise():
        async with app.run_test(size=(72, 24)) as pilot:
            await pilot.press("w")
            assert app.selected_index == 1

            await pilot.press("m")
            assert isinstance(app.screen, ModeScreen)
            modes = app.screen.query_one("#mode-list", SelectionList)
            assert modes.selected == []

            await pilot.press("space", "down", "space", "j", "space", "enter")
            await pilot.pause()
            assert not isinstance(app.screen, ModeScreen)
            assert app.nuclear_only is True
            assert app.show_rules is True
            assert app.flag_heteronyms is True
            assert app.selected_index == 1
            assert app.display_results[0].primary == -1
            assert app.lexical_results[0].primary == 0
            assert "[R1]" in app.base_text.plain
            assert "⚠HET" in app.base_text.plain

            # Esc belongs to the modal while it is open: cancelling a pending
            # mode change must leave both settings and document untouched.
            await pilot.press("m", "space", "escape")
            await pilot.pause()
            assert app.nuclear_only is True
            assert app.source_text == "Alpha beta"
            assert app.selected_index == 1

            # Applied popup settings remain session modes for later pastes.
            app.post_message(events.Paste("Gamma delta"))
            await pilot.pause()
            assert app.nuclear_only is True
            assert "[R1]" in app.base_text.plain
            assert "⚠HET" in app.base_text.plain

    asyncio.run(exercise())

    assert calls == [
        ("Alpha beta", False),
        ("Alpha beta", True),
        ("Alpha beta", False),
        ("Gamma delta", True),
        ("Gamma delta", False),
    ]


def test_cli_tui_passes_file_text_to_reusable_viewer(monkeypatch, tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("Alpha", encoding="utf-8")

    captured = {}

    def fake_run_tui(initial_text, **options):
        captured.update(initial_text=initial_text, options=options)

    monkeypatch.setattr("stressmark.tui.run_tui", fake_run_tui)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stressmark",
            str(input_path),
            "--tui",
            "--nuclear-only",
            "--explain",
            "--flag-heteronyms",
        ],
    )

    cli.main()

    assert captured["initial_text"] == "Alpha"
    assert captured["options"]["nuclear_only"] is True
    assert captured["options"]["show_rules"] is True
    assert captured["options"]["flag_heteronyms"] is True


def test_cli_tui_without_a_file_starts_blank_without_reading_stdin(monkeypatch):
    class UnreadableStdin:
        def read(self):
            raise AssertionError("interactive TUI must not consume stdin")

    captured = {}

    def fake_run_tui(initial_text, **options):
        captured.update(initial_text=initial_text, options=options)

    monkeypatch.setattr(sys, "stdin", UnreadableStdin())
    monkeypatch.setattr(sys, "argv", ["stressmark", "--tui"])
    monkeypatch.setattr("stressmark.tui.run_tui", fake_run_tui)

    cli.main()

    assert captured["initial_text"] == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["input.txt", "--tui", "-o", "output.txt"],
        ["input.txt", "--tui", "--format", "pdf"],
    ],
)
def test_cli_rejects_incompatible_tui_inputs(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["stressmark", *arguments])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
