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


def _punctuated_analyzer(_text, nuclear_only=False):
    return (
        [(True, "Alpha"), (False, "!")],
        [_result("Alpha", nuclear_only=nuclear_only), None],
    )


def _which(mapping):
    return lambda executable: mapping.get(executable)


def _mock_worker_command(tmp_path):
    worker = tmp_path / "mock_vim_worker.py"
    worker.write_text(
        """\
import json
import sys
import time

print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    text = request["text"]
    if text == "slow":
        time.sleep(0.1)
    elif text == "latest":
        time.sleep(0.4)
    else:
        time.sleep(0.15)
    if text == "FAIL":
        response = {"type": "error", "id": request["id"], "error": "mock failure"}
    else:
        rendered = text.upper()
        spans = []
        for row, rendered_line in enumerate(rendered.split("\\n")):
            if rendered_line:
                spans.append({
                    "row": row,
                    "start_col": 0,
                    "end_col": len(rendered_line.encode("utf-8")),
                    "style": "StressmarkStyle0",
                })
        response = {
            "type": "result",
            "id": request["id"],
            "payload": {
                "title": "stressmark: mock paste",
                "text": rendered,
                "styles": [{
                    "name": "StressmarkStyle0",
                    "guifg": "#ffffff",
                    "guibg": None,
                    "ctermfg": 7,
                    "ctermbg": None,
                    "attrs": ["bold"],
                }],
                "spans": spans,
                "words": [],
            },
        }
    print(json.dumps(response, ensure_ascii=False), flush=True)
""",
        encoding="utf-8",
    )
    return [sys.executable, "-u", str(worker)]


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
    assert "\nRaw stressmark:" in payload["words"][0]["detail"]
    assert payload["words"][0]["detail_spans"]
    used_styles = {
        span["style"]
        for span in (
            payload["spans"] + payload["words"][0]["detail_spans"]
        )
    }
    assert used_styles <= {
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
        if kind == "nvim":
            adapter = (observed["directory"] / "stressmark.lua").read_text()
            assert "--nuclear-only" in adapter
            assert "--explain" in adapter
            assert "--flag-heteronyms" in adapter
        return SimpleNamespace(returncode=7)

    result = run_vim_viewer(
        "ignored",
        nuclear_only=True,
        show_rules=True,
        flag_heteronyms=True,
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


def test_cli_native_viewer_starts_blank_without_reading_terminal_stdin(monkeypatch):
    class TerminalInput(io.StringIO):
        def isatty(self):
            return True

        def read(self, *_args, **_kwargs):
            raise AssertionError("blank native viewer must retain terminal stdin")

    captured = {}
    monkeypatch.setattr(sys, "stdin", TerminalInput())
    monkeypatch.setattr(sys, "argv", ["stressmark", "--vim"])
    monkeypatch.setattr(
        "stressmark.vim.run_vim_viewer",
        lambda text, **options: captured.update(text=text, options=options) or 0,
    )

    cli.main()

    assert captured["text"] == ""


def test_blank_native_viewer_requires_neovim():
    with pytest.raises(EditorUnavailableError, match="empty reusable.*requires Neovim"):
        run_vim_viewer("", editor=EditorCommand("vim", ("/usr/bin/vim",)))


@pytest.mark.skipif(shutil.which("nvim") is None, reason="Neovim is unavailable")
def test_neovim_adapter_headlessly_applies_wrapping_and_colored_detail_popup(tmp_path):
    document = build_editor_document("Alpha!", analyzer=_punctuated_analyzer)
    buffer_path, script_path = _write_adapter_files(
        tmp_path,
        document,
        title="stressmark test",
        editor_kind="nvim",
    )
    validation = (
        "local namespaces=vim.api.nvim_get_namespaces(); "
        "local detail_buf=vim.b.stressmark_detail_buf; "
        "local detail_win=vim.b.stressmark_detail_win; "
        "local j=vim.fn.maparg('j','n',false,true); "
        "local k=vim.fn.maparg('k','n',false,true); "
        "local initial=table.concat(vim.api.nvim_buf_get_lines(detail_buf,0,-1,false),'\\n'); "
        "vim.api.nvim_win_set_cursor(0,{1,8}); vim.cmd('doautocmd CursorMoved'); "
        "local neutral=table.concat(vim.api.nvim_buf_get_lines(detail_buf,0,-1,false),'\\n'); "
        "vim.api.nvim_win_set_cursor(0,{1,0}); vim.cmd('doautocmd CursorMoved'); "
        "local updated=table.concat(vim.api.nvim_buf_get_lines(detail_buf,0,-1,false),'\\n'); "
        "local ok=not vim.bo.modifiable and vim.bo.readonly and "
        "vim.wo.wrap and vim.wo.linebreak and "
        "j.buffer==1 and j.rhs=='gj' and k.buffer==1 and k.rhs=='gk' and "
        "vim.o.laststatus==3 and vim.wo.statusline=='sentinel' and "
        "vim.api.nvim_win_is_valid(detail_win) and vim.api.nvim_get_current_win()~=detail_win and "
        "initial:match('Word 1/1') and initial:match('Raw stressmark: ALPHA') and "
        "neutral=='stressmark viewer' and updated:match('Word 1/1') and "
        "#vim.api.nvim_buf_get_extmarks(0,namespaces.stressmark,0,-1,{})>0 and "
        "#vim.api.nvim_buf_get_extmarks(detail_buf,namespaces.stressmark_detail,0,-1,{})>0; "
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
            "set laststatus=3",
            "-c",
            "setlocal statusline=sentinel",
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


@pytest.mark.skipif(shutil.which("nvim") is None, reason="Neovim is unavailable")
def test_neovim_adapter_reuses_viewer_for_streamed_and_register_pastes(tmp_path):
    document = build_editor_document("")
    buffer_path, script_path = _write_adapter_files(
        tmp_path,
        document,
        title="stressmark paste test",
        editor_kind="nvim",
        worker_command=_mock_worker_command(tmp_path),
    )
    validation = (
        "local ns=vim.api.nvim_get_namespaces().stressmark; "
        "local pmap=vim.fn.maparg('p','n',false,true); "
        "local Pmap=vim.fn.maparg('P','n',false,true); "
        "vim.paste({'naï'},1); vim.paste({'ve','sec'},2); vim.paste({'ond'},3); "
        "local submitted=vim.wait(1000,function() return vim.b.stressmark_generation==1 end,10); "
        "local preview=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "local rendered=vim.wait(5000,function() return vim.b.stressmark_state=='rendered' end,10); "
        "local output=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "local colored=#vim.api.nvim_buf_get_extmarks(0,ns,0,-1,{})>0; "
        "vim.fn.setreg('a','register text'); vim.api.nvim_feedkeys('\"ap','x',false); "
        "local register_rendered=vim.wait(5000,function() return vim.b.stressmark_generation==2 and vim.b.stressmark_state=='rendered' end,10); "
        "local register_output=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "vim.fn.setreg('b','named text'); vim.cmd('StressmarkPaste b'); "
        "local named_rendered=vim.wait(5000,function() return vim.b.stressmark_generation==3 and vim.b.stressmark_state=='rendered' end,10); "
        "local named_output=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "vim.cmd('StressmarkClear'); "
        "local cleared=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n')==''; "
        "local ok=submitted and rendered and register_rendered and named_rendered and "
        "preview=='naïve\\nsecond' and output=='NAÏVE\\nSECOND' and colored and "
        "type(pmap.callback)=='function' and type(Pmap.callback)=='function' and "
        "register_output=='REGISTER TEXT' and named_output=='NAMED TEXT' and cleared; "
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
            f"lua _G.stressmark_original_paste=vim.paste; dofile({str(script_path)!r})",
            "-c",
            f"lua {validation}",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(shutil.which("nvim") is None, reason="Neovim is unavailable")
def test_neovim_adapter_ignores_stale_results_recovers_and_cleans_up(tmp_path):
    document = build_editor_document("")
    buffer_path, script_path = _write_adapter_files(
        tmp_path,
        document,
        title="stressmark lifecycle test",
        editor_kind="nvim",
        worker_command=_mock_worker_command(tmp_path),
    )
    validation = (
        "local viewer=vim.api.nvim_get_current_buf(); "
        "vim.paste({'slow'},-1); "
        "local first=vim.wait(1000,function() return vim.b.stressmark_generation==1 end,10); "
        "vim.paste({'latest'},-1); "
        "local second=vim.wait(1000,function() return vim.b.stressmark_generation==2 end,10); "
        "vim.wait(250,function() return false end,10); "
        "local stale_ignored=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n')=='latest'; "
        "local latest=vim.wait(3000,function() return vim.b.stressmark_generation==2 and vim.b.stressmark_state=='rendered' end,10); "
        "local latest_output=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "vim.paste({'FAIL'},-1); "
        "local failed=vim.wait(3000,function() return tostring(vim.b.stressmark_state):match('failed')~=nil end,10); "
        "local failed_output=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "vim.paste({'recover'},-1); "
        "local recovered=vim.wait(3000,function() return vim.b.stressmark_generation==4 and vim.b.stressmark_state=='rendered' end,10); "
        "local recovered_output=table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\\n'); "
        "local worker=vim.b.stressmark_worker_job; "
        "vim.cmd('vnew'); vim.paste({'delegated'},-1); "
        "local delegated=vim.api.nvim_get_current_line():match('delegated')~=nil; vim.cmd('close'); "
        "vim.cmd('bwipeout!'); "
        "local cleaned=vim.wait(1000,function() return vim.paste==_G.stressmark_original_paste and vim.fn.jobwait({worker},0)[1]~=-1 end,10); "
        "local ok=first and second and stale_ignored and latest and latest_output=='LATEST' and "
        "failed and failed_output=='FAIL' and recovered and recovered_output=='RECOVER' and "
        "delegated and cleaned and not vim.api.nvim_buf_is_valid(viewer); "
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
            f"lua _G.stressmark_original_paste=vim.paste; dofile({str(script_path)!r})",
            "-c",
            f"lua {validation}",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(shutil.which("vim") is None, reason="Vim is unavailable")
def test_vim_adapter_headlessly_applies_wrapping_and_colored_detail_popup(tmp_path):
    document = build_editor_document("Alpha!", analyzer=_punctuated_analyzer)
    buffer_path, script_path = _write_adapter_files(
        tmp_path,
        document,
        title="stressmark test",
        editor_kind="vim",
    )
    validation = (
        "let g:stressmark_initial = join(getbufline(b:stressmark_detail_buf, 1, '$'), '\\n') | "
        "call cursor(1, 9) | doautocmd CursorMoved | "
        "let g:stressmark_neutral = join(getbufline(b:stressmark_detail_buf, 1, '$'), '\\n') | "
        "call cursor(1, 1) | doautocmd CursorMoved | "
        "let g:stressmark_updated = join(getbufline(b:stressmark_detail_buf, 1, '$'), '\\n') | "
        "if &modifiable || !&readonly || !&wrap || !&linebreak || "
        "maparg('j', 'n', 0, 1).rhs !=# 'gj' || "
        "maparg('k', 'n', 0, 1).rhs !=# 'gk' || "
        "&laststatus != 1 || &l:statusline !=# 'sentinel' || "
        "empty(prop_list(1)) || empty(popup_getpos(b:stressmark_detail_popup)) || "
        "g:stressmark_initial !~# 'Word 1/1' || "
        "g:stressmark_neutral !=# 'stressmark viewer' || "
        "g:stressmark_updated !~# 'Word 1/1' || "
        "empty(prop_list(1, {'bufnr': b:stressmark_detail_buf})) | "
        "cquit | else | qa! | endif"
    )
    completed = subprocess.run(
        [
            shutil.which("vim"),
            "-Nu",
            "NONE",
            "-n",
            "-R",
            "-es",
            "--cmd",
            "set laststatus=1",
            "--cmd",
            "setlocal statusline=sentinel",
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
