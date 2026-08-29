"""Launch Vim or Neovim as a native read-only stressmark viewer."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.style import Style
from rich.terminal_theme import MONOKAI

from stressmark import render
from stressmark.viewer import analyze_for_viewer


class EditorUnavailableError(RuntimeError):
    """Raised when no supported Vim-family executable can be found."""


@dataclass(frozen=True)
class EditorCommand:
    kind: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class EditorStyle:
    name: str
    guifg: str | None
    guibg: str | None
    ctermfg: int | None
    ctermbg: int | None
    attrs: tuple[str, ...]


@dataclass(frozen=True)
class EditorSpan:
    start: int
    end: int
    style: str


@dataclass(frozen=True)
class EditorWord:
    start: int
    end: int
    detail: str


@dataclass(frozen=True)
class EditorDocument:
    text: str
    styles: tuple[EditorStyle, ...]
    spans: tuple[EditorSpan, ...]
    words: tuple[EditorWord, ...]


_VIM_EXECUTABLES = {"vim", "vi", "view"}
_NEOVIM_EXECUTABLES = {"nvim"}


def _editor_kind(executable):
    name = Path(executable).name.lower().removesuffix(".exe")
    if name in _NEOVIM_EXECUTABLES:
        return "nvim"
    if name in _VIM_EXECUTABLES:
        return "vim"
    return None


def _resolve_candidate(command, which):
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None

    executable = which(parts[0])
    if executable is None:
        return None
    kind = _editor_kind(executable)
    if kind is None:
        return None
    return EditorCommand(kind, (executable, *parts[1:]))


def resolve_editor(environ=None, which=shutil.which):
    """Resolve a Vim-family command without invoking an arbitrary shell."""
    environ = os.environ if environ is None else environ
    for variable in ("VISUAL", "EDITOR"):
        command = environ.get(variable, "").strip()
        if command and (resolved := _resolve_candidate(command, which)) is not None:
            return resolved

    for executable in ("nvim", "vim", "vi"):
        if path := which(executable):
            return EditorCommand(_editor_kind(path), (path,))

    raise EditorUnavailableError(
        "--vim requires Neovim or Vim; set $VISUAL/$EDITOR or install nvim/vim"
    )


def _hex_color(triplet):
    return f"#{triplet.red:02x}{triplet.green:02x}{triplet.blue:02x}"


def _editor_style(name, rich_style):
    style = rich_style if isinstance(rich_style, Style) else Style.parse(rich_style)
    foreground = (
        style.color.get_truecolor(MONOKAI)
        if style.color is not None
        else MONOKAI.foreground_color
    )
    background = (
        style.bgcolor.get_truecolor(MONOKAI, foreground=False)
        if style.bgcolor is not None
        else None
    )

    if style.dim:
        page_background = MONOKAI.background_color
        foreground = type(foreground)(
            *(int((value + backdrop) / 2) for value, backdrop in zip(foreground, page_background))
        )

    attributes = (
        ("bold", "bold"),
        ("italic", "italic"),
        ("underline", "underline"),
        ("reverse", "reverse"),
        ("strike", "strikethrough"),
    )
    attrs = tuple(
        editor_attribute
        for rich_attribute, editor_attribute in attributes
        if getattr(style, rich_attribute)
    )
    return EditorStyle(
        name=name,
        guifg=_hex_color(foreground) if style.color is not None or style.dim else None,
        guibg=_hex_color(background) if background is not None else None,
        ctermfg=(
            style.color.number
            if style.color is not None and style.color.number is not None
            else (8 if style.dim else None)
        ),
        ctermbg=(
            style.bgcolor.number
            if style.bgcolor is not None and style.bgcolor.number is not None
            else None
        ),
        attrs=attrs,
    )


def build_editor_document(
    text,
    *,
    nuclear_only=False,
    show_rules=False,
    flag_heteronyms=False,
    analyzer=None,
):
    """Build editor-neutral text, highlights, and cursor detail metadata."""
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    analysis_options = {"nuclear_only": nuclear_only}
    if analyzer is not None:
        analysis_options["analyzer"] = analyzer
    analysis = analyze_for_viewer(normalized_text, **analysis_options)

    word_ranges = []
    rich_text = render._render_rich_text(
        analysis.raw_tokens,
        analysis.display_results,
        show_rules,
        flag_heteronyms,
        word_ranges=word_ranges,
    )

    style_names = {}
    styles = []
    spans = []
    for rich_span in rich_text.spans:
        if rich_span.end <= rich_span.start:
            continue
        rich_style = rich_span.style
        style_key = str(rich_style)
        if style_key not in style_names:
            style_name = f"StressmarkStyle{len(style_names)}"
            style_names[style_key] = style_name
            styles.append(_editor_style(style_name, rich_style))
        spans.append(
            EditorSpan(
                rich_span.start,
                rich_span.end,
                style_names[style_key],
            )
        )

    words = []
    total_words = len(word_ranges)
    for word_number, (token_index, start, end) in enumerate(word_ranges, 1):
        result = analysis.lexical_results[token_index]
        original = analysis.raw_tokens[token_index][1]
        raw_stress = render.render_word(result).plain
        detail = (
            f"Word {word_number}/{total_words}  {original}  "
            f"Raw stressmark: {raw_stress}  POS: {result.tag or 'unknown'}  "
            f"source: {render.confidence_label(result.confidence)}"
        )
        words.append(EditorWord(start, end, detail))

    return EditorDocument(
        rich_text.plain,
        tuple(styles),
        tuple(spans),
        tuple(words),
    )


def _split_range(text, start, end):
    """Yield row and UTF-8 byte columns for a character-offset range."""
    line_start = 0
    for row, line in enumerate(text.split("\n")):
        line_end = line_start + len(line)
        part_start = max(start, line_start)
        part_end = min(end, line_end)
        if part_end > part_start:
            start_col = len(text[line_start:part_start].encode("utf-8"))
            end_col = len(text[line_start:part_end].encode("utf-8"))
            yield row, start_col, end_col
        if line_end >= end:
            break
        line_start = line_end + 1


def editor_payload(document, *, title="stressmark output"):
    """Convert character-offset metadata to editor byte coordinates."""
    spans = [
        {
            "row": row,
            "start_col": start_col,
            "end_col": end_col,
            "style": span.style,
        }
        for span in document.spans
        for row, start_col, end_col in _split_range(
            document.text,
            span.start,
            span.end,
        )
    ]
    words = [
        {
            "row": row,
            "start_col": start_col,
            "end_col": end_col,
            "detail": word.detail,
        }
        for word in document.words
        for row, start_col, end_col in _split_range(
            document.text,
            word.start,
            word.end,
        )
    ]
    return {
        "title": title,
        "styles": [asdict(style) for style in document.styles],
        "spans": spans,
        "words": words,
    }


_NEOVIM_SCRIPT = r'''
local data_path = __DATA_PATH__
local data = vim.json.decode(table.concat(vim.fn.readfile(data_path), "\n"))
local buffer = vim.api.nvim_get_current_buf()
local namespace = vim.api.nvim_create_namespace("stressmark")

for _, style in ipairs(data.styles) do
  local highlight = {}
  if style.guifg ~= vim.NIL then highlight.fg = style.guifg end
  if style.guibg ~= vim.NIL then highlight.bg = style.guibg end
  if style.ctermfg ~= vim.NIL then highlight.ctermfg = style.ctermfg end
  if style.ctermbg ~= vim.NIL then highlight.ctermbg = style.ctermbg end
  local cterm = {}
  for _, attribute in ipairs(style.attrs) do
    highlight[attribute] = true
    cterm[attribute] = true
  end
  if next(cterm) then highlight.cterm = cterm end
  vim.api.nvim_set_hl(0, style.name, highlight)
end

for _, span in ipairs(data.spans) do
  vim.api.nvim_buf_set_extmark(buffer, namespace, span.row, span.start_col, {
    end_row = span.row,
    end_col = span.end_col,
    hl_group = span.style,
    priority = 100,
  })
end

vim.b.stressmark_words = data.words
_G.StressmarkStatusline = function()
  local cursor = vim.api.nvim_win_get_cursor(0)
  local row, column = cursor[1] - 1, cursor[2]
  for _, word in ipairs(vim.b.stressmark_words or {}) do
    if word.row == row and column >= word.start_col and column < word.end_col then
      return word.detail:gsub("%%", "%%%%")
    end
  end
  return "stressmark viewer"
end

vim.o.laststatus = 2
vim.wo.statusline = "%!v:lua.StressmarkStatusline()"
vim.api.nvim_create_autocmd({"CursorMoved", "CursorMovedI"}, {
  buffer = buffer,
  callback = function() vim.cmd("redrawstatus") end,
})
vim.bo[buffer].readonly = true
vim.bo[buffer].modifiable = false
vim.bo[buffer].swapfile = false
vim.bo[buffer].buflisted = false
vim.bo[buffer].bufhidden = "wipe"
vim.bo[buffer].buftype = "nofile"
vim.bo[buffer].filetype = "stressmark"
vim.b.stressmark_source = data.title
'''


_VIM_SCRIPT = r'''
if !has('textprop') || !exists('*json_decode')
  echoerr 'stressmark --vim requires Vim with text properties and JSON support'
  cquit
endif

let s:data = json_decode(join(readfile('__DATA_PATH__'), "\n"))
let s:buffer = bufnr('%')
for s:style in s:data.styles
  let s:command = 'highlight ' . s:style.name
  if s:style.guifg isnot v:null | let s:command .= ' guifg=' . s:style.guifg | endif
  if s:style.guibg isnot v:null | let s:command .= ' guibg=' . s:style.guibg | endif
  if s:style.ctermfg isnot v:null | let s:command .= ' ctermfg=' . s:style.ctermfg | endif
  if s:style.ctermbg isnot v:null | let s:command .= ' ctermbg=' . s:style.ctermbg | endif
  let s:attributes = empty(s:style.attrs) ? 'NONE' : join(s:style.attrs, ',')
  let s:command .= ' gui=' . s:attributes . ' cterm=' . s:attributes
  execute s:command
  call prop_type_add(s:style.name, {'highlight': s:style.name, 'bufnr': s:buffer})
endfor

for s:span in s:data.spans
  call prop_add(s:span.row + 1, s:span.start_col + 1, {
        \ 'bufnr': s:buffer,
        \ 'length': s:span.end_col - s:span.start_col,
        \ 'type': s:span.style,
        \ })
endfor

let b:stressmark_words = s:data.words
function! StressmarkStatusline() abort
  if !exists('b:stressmark_words')
    return 'stressmark viewer'
  endif
  let l:row = line('.') - 1
  let l:column = col('.') - 1
  for l:word in b:stressmark_words
    if l:word.row == l:row && l:column >= l:word.start_col && l:column < l:word.end_col
      return substitute(l:word.detail, '%', '%%', 'g')
    endif
  endfor
  return 'stressmark viewer'
endfunction

set laststatus=2
let &l:statusline = '%!StressmarkStatusline()'
augroup stressmark_viewer
  autocmd! * <buffer>
  autocmd CursorMoved,CursorMovedI <buffer> redrawstatus
augroup END
setlocal readonly nomodifiable noswapfile nobuflisted bufhidden=wipe buftype=nofile
setlocal filetype=stressmark
let b:stressmark_source = s:data.title
'''


def _vim_quote(value):
    return value.replace("'", "''")


def _write_adapter_files(directory, document, *, title, editor_kind):
    buffer_path = directory / "stressmark.txt"
    data_path = directory / "stressmark.json"
    script_path = directory / ("stressmark.lua" if editor_kind == "nvim" else "stressmark.vim")

    buffer_path.write_text(document.text, encoding="utf-8")
    data_path.write_text(
        json.dumps(editor_payload(document, title=title), ensure_ascii=False),
        encoding="utf-8",
    )
    if editor_kind == "nvim":
        script = _NEOVIM_SCRIPT.replace("__DATA_PATH__", json.dumps(str(data_path)))
    else:
        script = _VIM_SCRIPT.replace("__DATA_PATH__", _vim_quote(str(data_path)))
    script_path.write_text(script, encoding="utf-8")
    return buffer_path, script_path


def run_vim_viewer(
    text,
    *,
    nuclear_only=False,
    show_rules=False,
    flag_heteronyms=False,
    source_name=None,
    editor=None,
    runner=subprocess.run,
):
    """Analyze text, launch the native editor viewer, and return its exit code."""
    editor = resolve_editor() if editor is None else editor
    document = build_editor_document(
        text,
        nuclear_only=nuclear_only,
        show_rules=show_rules,
        flag_heteronyms=flag_heteronyms,
    )
    title = f"stressmark: {source_name}" if source_name else "stressmark: stdin"

    with tempfile.TemporaryDirectory(prefix="stressmark-vim-") as temporary:
        directory = Path(temporary)
        buffer_path, script_path = _write_adapter_files(
            directory,
            document,
            title=title,
            editor_kind=editor.kind,
        )
        if editor.kind == "nvim":
            command = [
                *editor.argv,
                "-n",
                "-R",
                str(buffer_path),
                "-c",
                f"lua dofile({json.dumps(str(script_path))})",
            ]
        else:
            command = [
                *editor.argv,
                "-n",
                "-R",
                "-S",
                str(script_path),
                str(buffer_path),
            ]
        completed = runner(command, check=False)
        return completed.returncode
