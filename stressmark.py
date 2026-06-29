#!/usr/bin/env python3
"""
stressmark -- sentence-aware English stress marker.

Usage:
    stressmark transcript.txt
    cat article.txt | stressmark
    stressmark transcript.txt --format html -o out.html
    stressmark transcript.txt --format json -o out.json
    stressmark transcript.txt --explain
    stressmark transcript.txt --flag-heteronyms
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stressmark_engine import analyze, HETERONYMS
import stressmark_render as render


def main():
    parser = argparse.ArgumentParser(
        prog="stressmark",
        description="Mark primary/secondary/unstressed syllables in English text, "
                    "with sentence-aware nuclear stress and repeat backgrounding.",
    )
    parser.add_argument("input", nargs="?", help="Input text file. Omit to read from stdin.")
    parser.add_argument("-o", "--output", help="Write output to this file instead of stdout.")
    parser.add_argument("--format", choices=["terminal", "html", "json"], default="terminal",
                         help="Output format (default: terminal).")
    parser.add_argument("--explain", action="store_true",
                         help="Annotate words with which of the 9 stress rules applied.")
    parser.add_argument("--flag-heteronyms", action="store_true",
                         help="Mark words whose stress depends on part of speech (record/object/...).")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    render.set_heteronym_words(HETERONYMS.keys())
    raw_tokens, results = analyze(text)

    if args.format == "terminal":
        if args.output:
            from rich.console import Console
            with open(args.output, "w", encoding="utf-8") as f:
                con = Console(file=f, force_terminal=True, width=100)
                render.render_terminal(raw_tokens, results, args.explain, args.flag_heteronyms, console=con)
        else:
            render.render_terminal(raw_tokens, results, args.explain, args.flag_heteronyms)
        return

    if args.format == "html":
        out = render.render_html(raw_tokens, results, args.explain)
    else:
        out = render.render_json(raw_tokens, results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
