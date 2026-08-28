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
    parser.add_argument("input", nargs="?", help="Input text file. Omit to read from stdin.")
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
                         help="Browse the terminal output interactively with Vim or arrow keys.")
    args = parser.parse_args()

    if args.tui and not args.input:
        parser.error("--tui requires an input file because stdin is used for navigation")
    if args.tui and args.output:
        parser.error("--tui cannot be combined with --output")
    if args.tui and args.format != OutputFormat.TERMINAL:
        parser.error("--tui cannot be combined with a non-terminal --format")

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    raw_tokens, results = analyze(text, nuclear_only=args.nuclear_only)

    if args.tui:
        # The main pane obeys --nuclear-only. The detail pane deliberately
        # receives an ordinary analysis so it can always reveal the selected
        # word's underlying lexical stress, including a demoted primary.
        lexical_results = results
        if args.nuclear_only:
            lexical_tokens, lexical_results = analyze(text, nuclear_only=False)
            if lexical_tokens != raw_tokens:
                raise RuntimeError("TUI analyses produced different token streams")
        from stressmark.tui import run_tui

        run_tui(
            raw_tokens,
            results,
            lexical_results,
            show_rules=args.explain,
            flag_heteronyms=args.flag_heteronyms,
            source_name=args.input,
        )
        return

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
