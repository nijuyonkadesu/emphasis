"""Launch Vim or Neovim as a native stressmark viewer."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
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
    detail_spans: tuple[EditorSpan, ...] = ()


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


def _collect_rich_spans(rich_text, style_names, styles):
    """Translate Rich spans while sharing one editor highlight registry."""
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
    return tuple(spans)


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
    spans = _collect_rich_spans(rich_text, style_names, styles)

    words = []
    total_words = len(word_ranges)
    for word_number, (token_index, start, end) in enumerate(word_ranges, 1):
        result = analysis.lexical_results[token_index]
        original = analysis.raw_tokens[token_index][1]
        detail = render.render_word_detail(
            result,
            original,
            word_number,
            total_words,
        )
        words.append(
            EditorWord(
                start,
                end,
                detail.plain,
                _collect_rich_spans(detail, style_names, styles),
            )
        )

    return EditorDocument(
        rich_text.plain,
        tuple(styles),
        spans,
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
            "detail_spans": [
                {
                    "row": detail_row,
                    "start_col": detail_start,
                    "end_col": detail_end,
                    "style": detail_span.style,
                }
                for detail_span in word.detail_spans
                for detail_row, detail_start, detail_end in _split_range(
                    word.detail,
                    detail_span.start,
                    detail_span.end,
                )
            ],
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
        "text": document.text,
        "styles": [asdict(style) for style in document.styles],
        "spans": spans,
        "words": words,
    }


_NEOVIM_SCRIPT = r'''
local data_path = __DATA_PATH__
local data = vim.json.decode(table.concat(vim.fn.readfile(data_path), "\n"))
local worker_command = vim.json.decode(__WORKER_COMMAND__)
local buffer = vim.api.nvim_get_current_buf()
local window = vim.api.nvim_get_current_win()
local document_namespace = vim.api.nvim_create_namespace("stressmark")
local detail_namespace = vim.api.nvim_create_namespace("stressmark_detail")
local detail_buffer = vim.api.nvim_create_buf(false, true)
local detail_window = nil
local detail_state = "document"
local disposed = false
local generation = 0
local analyzing = false
local worker_job = nil
local worker_stdout = { "" }
local worker_stderr = { "" }
local worker_stderr_log = {}
local spinner_timer = nil
local spinner_position = 1
local paste_chunks = nil
local original_paste = vim.paste
local paste_handler = nil

local function apply_styles(styles)
  for _, style in ipairs(styles) do
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
end

local function detail_dimensions(lines)
  local longest = 1
  for _, line in ipairs(lines) do
    longest = math.max(longest, vim.fn.strdisplaywidth(line))
  end
  local available_width = math.max(vim.o.columns - 4, 1)
  local width = math.min(math.max(longest, 17), math.min(72, available_width))
  local height = 0
  for _, line in ipairs(lines) do
    height = height + math.max(1, math.ceil(vim.fn.strdisplaywidth(line) / width))
  end
  local available_height = math.max(vim.o.lines - vim.o.cmdheight - 4, 1)
  return width, math.min(height, available_height)
end

local function detail_config(lines)
  local width, height = detail_dimensions(lines)
  return {
    relative = "editor",
    anchor = "SE",
    row = math.max(vim.o.lines - vim.o.cmdheight - 1, 1),
    col = vim.o.columns,
    width = width,
    height = height,
    style = "minimal",
    border = "rounded",
    focusable = false,
    zindex = 60,
  }
end

local function ensure_detail_window(lines)
  local config = detail_config(lines)
  if detail_window and vim.api.nvim_win_is_valid(detail_window) then
    vim.api.nvim_win_set_config(detail_window, config)
    return
  end
  detail_window = vim.api.nvim_open_win(detail_buffer, false, config)
  vim.wo[detail_window].wrap = true
  vim.wo[detail_window].linebreak = true
  vim.wo[detail_window].winblend = 0
  vim.wo[detail_window].winhighlight = "Normal:NormalFloat,FloatBorder:FloatBorder"
  vim.b[buffer].stressmark_detail_win = detail_window
  vim.b[buffer].stressmark_detail_buf = detail_buffer
end

local function word_at_cursor()
  if not vim.api.nvim_win_is_valid(window) then return nil end
  local cursor = vim.api.nvim_win_get_cursor(window)
  local row, column = cursor[1] - 1, cursor[2]
  for _, word in ipairs(data.words) do
    if word.row == row and column >= word.start_col and column < word.end_col then
      return word
    end
  end
  return nil
end

local function set_detail_content(text, spans, status_highlight)
  if not vim.api.nvim_buf_is_valid(buffer) or not vim.api.nvim_buf_is_valid(detail_buffer) then
    return
  end
  local lines = vim.split(text, "\n", { plain = true })

  vim.bo[detail_buffer].modifiable = true
  vim.api.nvim_buf_set_lines(detail_buffer, 0, -1, false, lines)
  vim.api.nvim_buf_clear_namespace(detail_buffer, detail_namespace, 0, -1)
  for _, span in ipairs(spans) do
    vim.api.nvim_buf_set_extmark(detail_buffer, detail_namespace, span.row, span.start_col, {
      end_row = span.row,
      end_col = span.end_col,
      hl_group = span.style,
      priority = 100,
    })
  end
  if status_highlight and #lines[1] > 0 then
    vim.api.nvim_buf_set_extmark(detail_buffer, detail_namespace, 0, 0, {
      end_row = 0,
      end_col = #lines[1],
      hl_group = status_highlight,
      priority = 110,
    })
  end
  vim.bo[detail_buffer].modifiable = false
  if vim.api.nvim_get_current_buf() == buffer then ensure_detail_window(lines) end
end

local function show_status(text, highlight)
  detail_state = "status"
  vim.b[buffer].stressmark_state = text
  set_detail_content(text, {}, highlight)
end

local function update_detail()
  if detail_state ~= "document" then return end
  local word = word_at_cursor()
  if word then
    set_detail_content(word.detail, word.detail_spans, nil)
  elseif data.text == "" then
    set_detail_content(
      "Paste text to analyze — p/P, terminal paste, or :StressmarkPaste",
      {},
      "Comment"
    )
  else
    set_detail_content("stressmark viewer", {}, nil)
  end
end

local function text_lines(text)
  local lines = vim.split(text, "\n", { plain = true })
  if #lines > 1 and lines[#lines] == "" then
    table.remove(lines)
  end
  if #lines == 0 then return { "" } end
  return lines
end

local function replace_buffer_text(text)
  local mode = vim.api.nvim_get_mode().mode
  if mode:sub(1, 1) == "i" or mode:sub(1, 1) == "R" then
    vim.cmd("stopinsert")
  end
  vim.bo[buffer].readonly = false
  vim.bo[buffer].modifiable = true
  vim.api.nvim_buf_set_lines(buffer, 0, -1, false, text_lines(text))
  vim.bo[buffer].modifiable = false
  vim.bo[buffer].readonly = true
  vim.bo[buffer].modified = false
  if vim.api.nvim_win_is_valid(window) then
    vim.api.nvim_win_set_cursor(window, { 1, 0 })
  end
end

local function apply_document(payload)
  data = payload
  apply_styles(data.styles)
  replace_buffer_text(data.text)
  vim.api.nvim_buf_clear_namespace(buffer, document_namespace, 0, -1)
  for _, span in ipairs(data.spans) do
    vim.api.nvim_buf_set_extmark(buffer, document_namespace, span.row, span.start_col, {
      end_row = span.row,
      end_col = span.end_col,
      hl_group = span.style,
      priority = 100,
    })
  end
  vim.b[buffer].stressmark_source = data.title
  vim.b[buffer].stressmark_state = "rendered"
  detail_state = "document"
  update_detail()
  vim.cmd("redraw")
end

local function show_plain_text(text)
  data = { text = text, styles = {}, spans = {}, words = {}, title = "stressmark: pasted text" }
  vim.api.nvim_buf_clear_namespace(buffer, document_namespace, 0, -1)
  replace_buffer_text(text)
  detail_state = "status"
  vim.b[buffer].stressmark_source = data.title
  vim.b[buffer].stressmark_state = "plain"
end

local spinner_frames = { "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" }

local function stop_spinner()
  analyzing = false
  if spinner_timer then
    pcall(spinner_timer.stop, spinner_timer)
    pcall(spinner_timer.close, spinner_timer)
    spinner_timer = nil
  end
end

local function start_spinner()
  stop_spinner()
  analyzing = true
  spinner_position = 1
  spinner_timer = vim.uv.new_timer()
  spinner_timer:start(0, 100, vim.schedule_wrap(function()
    if disposed or not analyzing then return end
    local frame = spinner_frames[spinner_position]
    spinner_position = spinner_position % #spinner_frames + 1
    show_status(frame .. " Analyzing pasted text…", "DiagnosticInfo")
  end))
end

local function consume_channel_lines(accumulator, chunks, handler)
  if #chunks == 0 then return end
  accumulator[#accumulator] = accumulator[#accumulator] .. chunks[1]
  for index = 2, #chunks do
    table.insert(accumulator, chunks[index])
  end
  while #accumulator > 1 do
    local line = table.remove(accumulator, 1)
    if line ~= "" then handler(line) end
  end
end

local function worker_error(message)
  stop_spinner()
  show_status("Stressmark analysis failed: " .. message, "ErrorMsg")
end

local function handle_worker_message(line)
  if disposed then return end
  local decoded, message = pcall(vim.json.decode, line)
  if not decoded then
    if analyzing then worker_error("invalid worker response") end
    return
  end
  if message.type == "result" and message.id == generation then
    stop_spinner()
    local applied, apply_error = pcall(apply_document, message.payload)
    if not applied then worker_error(tostring(apply_error)) end
  elseif message.type == "error" and (message.id == generation or message.id == vim.NIL) then
    worker_error(message.error or "unknown worker error")
  elseif message.type == "fatal" then
    worker_error(message.error or "worker failed to start")
  end
end

local function worker_stderr_text()
  local pieces = vim.deepcopy(worker_stderr_log)
  if worker_stderr[1] ~= "" then table.insert(pieces, worker_stderr[1]) end
  local message = table.concat(pieces, "\n"):gsub("^%s+", ""):gsub("%s+$", "")
  return message ~= "" and message or "analysis worker stopped"
end

local function ensure_worker()
  if worker_job and vim.fn.jobwait({ worker_job }, 0)[1] == -1 then
    return true
  end
  if #worker_command == 0 then return false end

  worker_stdout = { "" }
  worker_stderr = { "" }
  worker_stderr_log = {}
  local job_id
  job_id = vim.fn.jobstart(worker_command, {
    on_stdout = function(_, chunks)
      consume_channel_lines(worker_stdout, chunks, handle_worker_message)
    end,
    on_stderr = function(_, chunks)
      consume_channel_lines(worker_stderr, chunks, function(line)
        table.insert(worker_stderr_log, line)
      end)
    end,
    on_exit = function(exited_job)
      if worker_job ~= exited_job then return end
      worker_job = nil
      if vim.api.nvim_buf_is_valid(buffer) then
        vim.b[buffer].stressmark_worker_job = 0
      end
      if not disposed and analyzing then worker_error(worker_stderr_text()) end
    end,
  })
  if job_id <= 0 then return false end
  worker_job = job_id
  vim.b[buffer].stressmark_worker_job = worker_job
  return true
end

local function normalize_text(text)
  return text:gsub("\r\n", "\n"):gsub("\r", "\n")
end

local function clear_document()
  generation = generation + 1
  vim.b[buffer].stressmark_generation = generation
  stop_spinner()
  show_plain_text("")
  show_status(
    "Paste text to analyze — p/P, terminal paste, or :StressmarkPaste",
    "Comment"
  )
end

local function submit_text(text)
  text = normalize_text(text)
  generation = generation + 1
  vim.b[buffer].stressmark_generation = generation
  stop_spinner()
  show_plain_text(text)
  if text == "" then
    show_status(
      "Paste text to analyze — p/P, terminal paste, or :StressmarkPaste",
      "Comment"
    )
    return
  end

  start_spinner()
  if not ensure_worker() then
    worker_error("unable to start the analysis worker")
    return
  end
  local request = vim.json.encode({ id = generation, text = text }) .. "\n"
  if vim.fn.chansend(worker_job, request) == 0 then
    worker_error("unable to send text to the analysis worker")
  end
end

local function register_text(register_name)
  local info = vim.fn.getreginfo(register_name)
  local contents = info.regcontents or {}
  local text = table.concat(contents, "\n")
  if type(info.regtype) == "string" and info.regtype:sub(1, 1) == "V" and #contents > 0 then
    text = text .. "\n"
  end
  return text
end

local function submit_register(register_name)
  submit_text(register_text(register_name ~= "" and register_name or '"'))
end

local function append_paste_chunk(lines, reset)
  if reset or not paste_chunks then
    paste_chunks = vim.deepcopy(lines)
    if #paste_chunks == 0 then paste_chunks = { "" } end
    return
  end
  if #lines == 0 then return end
  paste_chunks[#paste_chunks] = paste_chunks[#paste_chunks] .. lines[1]
  for index = 2, #lines do
    table.insert(paste_chunks, lines[index])
  end
end

local function finish_paste(lines)
  local text = table.concat(lines, "\n")
  vim.schedule(function()
    if not disposed and vim.api.nvim_buf_is_valid(buffer) then submit_text(text) end
  end)
end

paste_handler = function(lines, phase)
  local mode = vim.api.nvim_get_mode().mode
  if disposed or vim.api.nvim_get_current_buf() ~= buffer or mode:sub(1, 1) == "c" then
    return original_paste(lines, phase)
  end
  if phase == -1 then
    finish_paste(lines)
  elseif phase == 1 then
    append_paste_chunk(lines, true)
  elseif phase == 2 then
    append_paste_chunk(lines, false)
  elseif phase == 3 then
    append_paste_chunk(lines, false)
    local completed = paste_chunks or { "" }
    paste_chunks = nil
    finish_paste(completed)
  else
    return false
  end
  return true
end

local function hide_detail()
  if detail_window and vim.api.nvim_win_is_valid(detail_window) then
    vim.api.nvim_win_hide(detail_window)
  end
end

local function show_detail()
  if detail_state == "document" then
    update_detail()
  elseif vim.api.nvim_buf_is_valid(detail_buffer) then
    ensure_detail_window(vim.api.nvim_buf_get_lines(detail_buffer, 0, -1, false))
  end
end

local function cleanup()
  disposed = true
  stop_spinner()
  if vim.paste == paste_handler then vim.paste = original_paste end
  if worker_job then
    local running_job = worker_job
    worker_job = nil
    pcall(vim.fn.jobstop, running_job)
  end
  hide_detail()
  if vim.api.nvim_buf_is_valid(detail_buffer) then
    vim.api.nvim_buf_delete(detail_buffer, { force = true })
  end
end

vim.bo[detail_buffer].buftype = "nofile"
vim.bo[detail_buffer].bufhidden = "hide"
vim.bo[detail_buffer].swapfile = false
vim.bo[detail_buffer].buflisted = false

local group = vim.api.nvim_create_augroup("stressmark_native_viewer", { clear = true })
vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI" }, {
  group = group,
  buffer = buffer,
  callback = update_detail,
})
vim.api.nvim_create_autocmd("BufLeave", {
  group = group,
  buffer = buffer,
  callback = hide_detail,
})
vim.api.nvim_create_autocmd("BufEnter", {
  group = group,
  buffer = buffer,
  callback = show_detail,
})
vim.api.nvim_create_autocmd("VimResized", {
  group = group,
  callback = function()
    if vim.api.nvim_get_current_buf() == buffer then update_detail() end
  end,
})
vim.api.nvim_create_autocmd("BufWipeout", {
  group = group,
  buffer = buffer,
  once = true,
  callback = function()
    cleanup()
    vim.schedule(function()
      pcall(vim.api.nvim_del_augroup_by_id, group)
    end)
  end,
})

vim.wo[window].wrap = true
vim.wo[window].linebreak = true
vim.keymap.set("n", "j", "gj", { buffer = buffer, silent = true, noremap = true })
vim.keymap.set("n", "k", "gk", { buffer = buffer, silent = true, noremap = true })
vim.keymap.set("n", "p", function() submit_register(vim.v.register) end, {
  buffer = buffer,
  silent = true,
  noremap = true,
  desc = "Replace with register text and analyze",
})
vim.keymap.set("n", "P", function() submit_register(vim.v.register) end, {
  buffer = buffer,
  silent = true,
  noremap = true,
  desc = "Replace with register text and analyze",
})
vim.api.nvim_buf_create_user_command(buffer, "StressmarkPaste", function(options)
  submit_register(options.args)
end, { nargs = "?", desc = "Replace with register text and analyze" })
vim.api.nvim_buf_create_user_command(buffer, "StressmarkClear", clear_document, {
  desc = "Clear the stressmark viewer",
})
vim.paste = paste_handler
vim.bo[buffer].readonly = true
vim.bo[buffer].modifiable = false
vim.bo[buffer].swapfile = false
vim.bo[buffer].buflisted = false
vim.bo[buffer].bufhidden = "wipe"
vim.bo[buffer].buftype = "nofile"
vim.bo[buffer].filetype = "stressmark"
vim.b[buffer].stressmark_generation = generation
apply_document(data)
ensure_worker()
'''


_VIM_SCRIPT = r'''
if !has('textprop') || !exists('*json_decode') || !exists('*popup_create')
  echoerr 'stressmark --vim requires Vim with text properties, JSON, and popup-window support'
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

let s:detail_popup = popup_create(['stressmark viewer'], {
      \ 'pos': 'botright',
      \ 'line': max([1, &lines - &cmdheight]),
      \ 'col': &columns,
      \ 'minwidth': min([17, max([1, &columns - 4])]),
      \ 'maxwidth': min([72, max([1, &columns - 4])]),
      \ 'maxheight': max([1, &lines - &cmdheight - 4]),
      \ 'padding': [0, 1, 0, 1],
      \ 'border': [],
      \ 'borderchars': ['─', '│', '─', '│', '╭', '╮', '╯', '╰'],
      \ 'mapping': 0,
      \ 'scrollbar': 0,
      \ 'wrap': 1,
      \ 'zindex': 60,
      \ })
let s:detail_buffer = winbufnr(s:detail_popup)
for s:style in s:data.styles
  call prop_type_add(s:style.name, {
        \ 'highlight': s:style.name,
        \ 'bufnr': s:detail_buffer,
        \ })
endfor

function! s:StressmarkWordAtCursor() abort
  if bufnr('%') != s:buffer
    return v:null
  endif
  let l:row = line('.') - 1
  let l:column = col('.') - 1
  for l:word in s:data.words
    if l:word.row == l:row && l:column >= l:word.start_col && l:column < l:word.end_col
      return l:word
    endif
  endfor
  return v:null
endfunction

function! s:StressmarkUpdateDetail() abort
  if empty(popup_getpos(s:detail_popup)) || !bufexists(s:buffer)
    return
  endif
  let l:word = s:StressmarkWordAtCursor()
  let l:text = l:word is v:null ? 'stressmark viewer' : l:word.detail
  let l:spans = l:word is v:null ? [] : l:word.detail_spans
  let l:lines = map(split(l:text, "\n", 1), '{"text": v:val, "props": []}')
  for l:span in l:spans
    call add(l:lines[l:span.row].props, {
          \ 'col': l:span.start_col + 1,
          \ 'length': l:span.end_col - l:span.start_col,
          \ 'type': l:span.style,
          \ })
  endfor
  call popup_settext(s:detail_popup, l:lines)
  call popup_setoptions(s:detail_popup, {
        \ 'line': max([1, &lines - &cmdheight]),
        \ 'col': &columns,
        \ 'minwidth': min([17, max([1, &columns - 4])]),
        \ 'maxwidth': min([72, max([1, &columns - 4])]),
        \ 'maxheight': max([1, &lines - &cmdheight - 4]),
        \ })
endfunction

function! s:StressmarkHideDetail() abort
  if !empty(popup_getpos(s:detail_popup))
    call popup_hide(s:detail_popup)
  endif
endfunction

function! s:StressmarkShowDetail() abort
  if !empty(popup_getpos(s:detail_popup))
    call popup_show(s:detail_popup)
    call s:StressmarkUpdateDetail()
  endif
endfunction

function! s:StressmarkCloseDetail() abort
  if !empty(popup_getpos(s:detail_popup))
    call popup_close(s:detail_popup)
  endif
  execute 'autocmd! stressmark_native_viewer'
endfunction

let b:stressmark_detail_popup = s:detail_popup
let b:stressmark_detail_buf = s:detail_buffer
setlocal wrap linebreak
nnoremap <silent><buffer> j gj
nnoremap <silent><buffer> k gk
augroup stressmark_native_viewer
  autocmd! * <buffer>
  autocmd CursorMoved,CursorMovedI <buffer> call <SID>StressmarkUpdateDetail()
  autocmd BufLeave <buffer> call <SID>StressmarkHideDetail()
  autocmd BufEnter <buffer> call <SID>StressmarkShowDetail()
  autocmd BufWipeout <buffer> call <SID>StressmarkCloseDetail()
  autocmd VimResized * call <SID>StressmarkUpdateDetail()
augroup END
setlocal readonly nomodifiable noswapfile nobuflisted bufhidden=wipe buftype=nofile
setlocal filetype=stressmark
let b:stressmark_source = s:data.title
call s:StressmarkUpdateDetail()
'''


def _vim_quote(value):
    return value.replace("'", "''")


def _write_adapter_files(
    directory,
    document,
    *,
    title,
    editor_kind,
    worker_command=(),
):
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
        script = script.replace(
            "__WORKER_COMMAND__",
            json.dumps(json.dumps(worker_command)),
        )
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
    if not text and editor.kind != "nvim":
        raise EditorUnavailableError(
            "an empty reusable --vim viewer requires Neovim; "
            "classic Vim remains available when initial text is provided"
        )
    document = build_editor_document(
        text,
        nuclear_only=nuclear_only,
        show_rules=show_rules,
        flag_heteronyms=flag_heteronyms,
    )
    title = (
        f"stressmark: {source_name}"
        if source_name
        else "stressmark: stdin" if text else "stressmark: paste-ready"
    )
    worker_command = [sys.executable, "-m", "stressmark.vim_worker"]
    if nuclear_only:
        worker_command.append("--nuclear-only")
    if show_rules:
        worker_command.append("--explain")
    if flag_heteronyms:
        worker_command.append("--flag-heteronyms")

    with tempfile.TemporaryDirectory(prefix="stressmark-vim-") as temporary:
        directory = Path(temporary)
        buffer_path, script_path = _write_adapter_files(
            directory,
            document,
            title=title,
            editor_kind=editor.kind,
            worker_command=worker_command if editor.kind == "nvim" else (),
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
