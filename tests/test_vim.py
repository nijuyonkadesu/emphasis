import io
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stressmark import cli
from stressmark.vim import (
    EditorCommand,
    EditorDocument,
    EditorSpan,
    EditorStyle,
    EditorUnavailableError,
    EditorWord,
    _write_adapter_files,
    build_editor_document,
    editor_payload,
    resolve_editor,
    run_vim_viewer,
)


def _result(raw, *, nuclear_only=False):
    return SimpleNamespace(
        raw=raw,
        cls="content",
        tier=None,
        syllables=[raw],
        primary=-1 if nuclear_only else 0,
        secondary=set(),
        confidence="predicted",
        rule=1,
        tag="NN",
        is_heteronym=True,
    )


def _analyzer(text, nuclear_only=False):
    return [(True, text)], [_result(text, nuclear_only=nuclear_only)]


def _which(mapping):
    return lambda executable: mapping.get(executable)


def test_editor_resolution_honors_vim_family_environment_commands():
    paths = {
        "nvim": "/usr/bin/nvim",
        "vim": "/usr/bin/vim",
        "nano": "/usr/bin/nano",
    }

    assert resolve_editor(
        {"VISUAL": "nvim --clean", "EDITOR": "vim"},
        _which(paths),
    ) == EditorCommand("nvim", ("/usr/bin/nvim", "--clean"))

    # A non-Vim $VISUAL is ignored rather than being handed generated scripts.
    assert resolve_editor(
        {"VISUAL": "nano", "EDITOR": "vim -f"},
        _which(paths),
    ) == EditorCommand("vim", ("/usr/bin/vim", "-f"))


def test_editor_resolution_prefers_neovim_fallback_and_reports_absence():
    paths = {"nvim": "/opt/bin/nvim", "vim": "/usr/bin/vim"}
    assert resolve_editor({}, _which(paths)).kind == "nvim"

    with pytest.raises(EditorUnavailableError):
        resolve_editor({}, _which({}))


def test_editor_payload_uses_utf8_byte_columns_and_shared_rendering():
    document = build_editor_document(
        "Alpha",
        show_rules=True,
        flag_heteronyms=True,
        analyzer=_analyzer,
    )
    payload = editor_payload(document)

    assert document.text == "ALPHA≈[R1]⚠HET"
    assert payload["words"][0]["start_col"] == 0
    assert payload["words"][0]["end_col"] == len(document.text.encode("utf-8"))
    assert "Raw stressmark: ALPHA≈" in payload["words"][0]["detail"]
    assert {span["style"] for span in payload["spans"]} <= {
        style["name"] for style in payload["styles"]
    }


def test_native_viewer_keeps_lexical_detail_in_nuclear_only_mode():
    calls = []

    def analyzer(text, nuclear_only=False):
        calls.append((text, nuclear_only))
        return _analyzer(text, nuclear_only=nuclear_only)

    document = build_editor_document(
        "Alpha",
        nuclear_only=True,
        analyzer=analyzer,
    )

    assert document.text == "alpha≈"
    assert "Raw stressmark: ALPHA≈" in document.words[0].detail
    assert calls == [("Alpha", True), ("Alpha", False)]


@pytest.mark.parametrize("kind", ["nvim", "vim"])
def test_native_viewer_builds_safe_commands_and_cleans_temporary_files(
    monkeypatch,
    kind,
):
    document = EditorDocument(
        "ALPHA",
        (EditorStyle("StressmarkStyle0", "#ffffff", None, 7, None, ("bold",)),),
        (EditorSpan(0, 5, "StressmarkStyle0"),),
        (EditorWord(0, 5, "Word 1/1  Alpha"),),
    )
    monkeypatch.setattr("stressmark.vim.build_editor_document", lambda *_args, **_kwargs: document)
    observed = {}

    def runner(command, check):
        observed["command"] = command
        observed["directory"] = next(
            path.parent
            for argument in command
            if (path := __import__("pathlib").Path(argument)).name == "stressmark.txt"
        )
        assert check is False
        assert (observed["directory"] / "stressmark.txt").read_text() == "ALPHA"
        assert json.loads((observed["directory"] / "stressmark.json").read_text())[
            "words"
        ]
        return SimpleNamespace(returncode=7)

    result = run_vim_viewer(
        "ignored",
        editor=EditorCommand(kind, (f"/usr/bin/{kind}", "--custom-option")),
        runner=runner,
    )

    assert result == 7
    assert "-u" not in observed["command"]
    assert "-Nu" not in observed["command"]
    assert not observed["directory"].exists()


def test_cli_dispatches_file_and_modes_to_native_viewer(monkeypatch, tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("Alpha", encoding="utf-8")
    captured = {}

    def fake_viewer(text, **options):
        captured.update(text=text, options=options)
        return 0

    monkeypatch.setattr("stressmark.vim.run_vim_viewer", fake_viewer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stressmark",
            str(source),
            "--vim",
            "--nuclear-only",
            "--explain",
            "--flag-heteronyms",
        ],
    )

    cli.main()

    assert captured["text"] == "Alpha"
    assert captured["options"] == {
        "nuclear_only": True,
        "show_rules": True,
        "flag_heteronyms": True,
        "source_name": str(source),
    }


def test_cli_native_viewer_accepts_piped_input_and_propagates_exit(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Alpha"))
    monkeypatch.setattr(sys, "argv", ["stressmark", "--vim"])
    monkeypatch.setattr("stressmark.vim.run_vim_viewer", lambda *_args, **_kwargs: 23)

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 23


@pytest.mark.parametrize(
    "arguments",
    [
        ["--vim", "--tui"],
        ["input.txt", "--vim", "-o", "output.txt"],
        ["input.txt", "--vim", "--format", "pdf"],
    ],
)
def test_cli_rejects_incompatible_native_viewer_options(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["stressmark", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_cli_native_viewer_rejects_an_empty_terminal(monkeypatch):
    class TerminalInput(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", TerminalInput())
    monkeypatch.setattr(sys, "argv", ["stressmark", "--vim"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.skipif(shutil.which("nvim") is None, reason="Neovim is unavailable")
def test_neovim_adapter_headlessly_applies_extmarks_and_cursor_detail(tmp_path):
    document = build_editor_document("Alpha", analyzer=_analyzer)
    buffer_path, script_path = _write_adapter_files(
        tmp_path,
        document,
        title="stressmark test",
        editor_kind="nvim",
    )
    validation = (
        "local ns=vim.api.nvim_get_namespaces().stressmark; "
        "local ok=not vim.bo.modifiable and vim.bo.readonly and "
        "#vim.api.nvim_buf_get_extmarks(0,ns,0,-1,{})>0 and "
        "StressmarkStatusline():match('Word 1/1'); "
        "if ok then vim.cmd('qa!') else vim.cmd('cquit') end"
    )
    completed = subprocess.run(
        [
            shutil.which("nvim"),
            "--headless",
            "-u",
            "NONE",
            "-n",
            "-R",
            str(buffer_path),
            "-c",
            f"lua dofile({str(script_path)!r})",
            "-c",
            f"lua {validation}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(shutil.which("vim") is None, reason="Vim is unavailable")
def test_vim_adapter_headlessly_applies_properties_and_cursor_detail(tmp_path):
    document = build_editor_document("Alpha", analyzer=_analyzer)
    buffer_path, script_path = _write_adapter_files(
        tmp_path,
        document,
        title="stressmark test",
        editor_kind="vim",
    )
    validation = (
        "if &modifiable || !&readonly || empty(prop_list(1)) || "
        "StressmarkStatusline() !~# 'Word 1/1' | cquit | else | qa! | endif"
    )
    completed = subprocess.run(
        [
            shutil.which("vim"),
            "-Nu",
            "NONE",
            "-n",
            "-R",
            "-es",
            "-S",
            str(script_path),
            str(buffer_path),
            "-c",
            validation,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
