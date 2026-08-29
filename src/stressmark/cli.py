#!/usr/bin/env python3
"""
stressmark -- sentence-aware English stress marker.

Usage:
    stressmark transcript.txt
    cat article.txt | stressmark
    stressmark transcript.txt --format html -o out.html
    stressmark transcript.txt --format pdf -o out.pdf
    stressmark transcript.txt --format json -o out.json
    stressmark transcript.txt --tui
    stressmark transcript.txt --vim
    stressmark transcript.txt --explain
    stressmark transcript.txt --flag-heteronyms
"""
import argparse
import sys

from stressmark import render
from stressmark.engine import RULES, analyze
from stressmark.model import APP_NAME, OutputFormat


def main():
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Mark primary/secondary/unstressed syllables in English text, "
                    "with sentence-aware nuclear stress and repeat backgrounding.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input text file. Omit to read stdin, or start blank in --tui/--vim.",
    )
    parser.add_argument("-o", "--output", help="Write output to this file instead of stdout.")
    parser.add_argument("--format", type=OutputFormat, choices=tuple(OutputFormat), default=OutputFormat.TERMINAL,
                         help="Output format (default: terminal).")
    parser.add_argument("--explain", action="store_true",
                         help=f"Annotate words with which of the {len(RULES)} stress rules applied.")
    parser.add_argument("--flag-heteronyms", action="store_true",
                         help="Mark words whose stress depends on part of speech (record/object/...).")
    parser.add_argument("--nuclear-only", action="store_true",
                         help="Only highlight the nuclear (focus) word per clause; demote all other content words to secondary/unstressed.")
    parser.add_argument("--tui", "--interactive", dest="tui", action="store_true",
                         help="Open the reusable paste-ready viewer with Vim or arrow navigation.")
    parser.add_argument("--vim", action="store_true",
                         help="Open the reusable native Neovim/Vim stress viewer.")
    args = parser.parse_args()

    if args.tui and args.vim:
        parser.error("--tui and --vim are mutually exclusive")
    if args.tui and args.output:
        parser.error("--tui cannot be combined with --output")
    if args.tui and args.format != OutputFormat.TERMINAL:
        parser.error("--tui cannot be combined with a non-terminal --format")
    if args.vim and args.output:
        parser.error("--vim cannot be combined with --output")
    if args.vim and args.format != OutputFormat.TERMINAL:
        parser.error("--vim cannot be combined with a non-terminal --format")

    if args.vim:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            is_terminal = getattr(sys.stdin, "isatty", lambda: False)()
            # Keep terminal stdin attached to the editor so bracketed paste
            # and native key handling remain available in a blank viewer.
            text = "" if is_terminal else sys.stdin.read()

        from stressmark.vim import EditorUnavailableError, run_vim_viewer

        try:
            return_code = run_vim_viewer(
                text,
                nuclear_only=args.nuclear_only,
                show_rules=args.explain,
                flag_heteronyms=args.flag_heteronyms,
                source_name=args.input,
            )
        except EditorUnavailableError as error:
            parser.error(str(error))
        if return_code:
            raise SystemExit(return_code)
        return

    if args.tui:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            # Do not consume stdin: the running terminal must remain available
            # for navigation keys and bracketed-paste events.
            text = ""
        from stressmark.tui import run_tui

        run_tui(
            text,
            nuclear_only=args.nuclear_only,
            show_rules=args.explain,
            flag_heteronyms=args.flag_heteronyms,
            source_name=args.input,
        )
        return

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    raw_tokens, results = analyze(text, nuclear_only=args.nuclear_only)

    if args.format == OutputFormat.TERMINAL:
        if args.output:
            from rich.console import Console
            with open(args.output, "w", encoding="utf-8") as f:
                con = Console(file=f, force_terminal=True, width=render.OUTPUT_COLUMNS)
                render.render_terminal(raw_tokens, results, args.explain, args.flag_heteronyms, console=con)
        else:
            render.render_terminal(raw_tokens, results, args.explain, args.flag_heteronyms)
        return

    if args.format == OutputFormat.PDF:
        out = render.render_pdf(raw_tokens, results, args.explain, args.flag_heteronyms)
        if args.output:
            with open(args.output, "wb") as f:
                f.write(out)
        else:
            sys.stdout.buffer.write(out)
        return

    if args.format == OutputFormat.HTML:
        out = render.render_html(
            raw_tokens,
            results,
            args.explain,
            args.flag_heteronyms,
        )
    else:
        out = render.render_json(raw_tokens, results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
