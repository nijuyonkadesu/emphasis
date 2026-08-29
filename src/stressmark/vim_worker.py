"""Persistent newline-delimited JSON worker for the native Neovim viewer."""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import sys
import threading


_STOP = object()


def _write_message(stream, lock, message):
    with lock:
        stream.write(json.dumps(message, ensure_ascii=False) + "\n")
        stream.flush()


def _replace_pending(requests, request):
    """Keep only the newest request that has not started analysis yet."""
    try:
        requests.get_nowait()
    except queue.Empty:
        pass
    requests.put_nowait(request)


def serve(input_stream, output_stream, *, renderer):
    """Process requests serially while coalescing queued work to the newest."""
    requests = queue.Queue(maxsize=1)
    output_lock = threading.Lock()

    def analyze_requests():
        while True:
            request = requests.get()
            if request is _STOP:
                return
            request_id = request.get("id")
            try:
                text = request["text"]
                if not isinstance(request_id, int) or not isinstance(text, str):
                    raise ValueError("request requires an integer id and string text")
                with contextlib.redirect_stdout(sys.stderr):
                    payload = renderer(text)
                response = {
                    "type": "result",
                    "id": request_id,
                    "payload": payload,
                }
            except Exception as error:  # Keep the worker alive for later pastes.
                response = {
                    "type": "error",
                    "id": request_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            _write_message(output_stream, output_lock, response)

    worker = threading.Thread(target=analyze_requests, name="stressmark-vim-worker")
    worker.start()
    _write_message(output_stream, output_lock, {"type": "ready"})

    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except Exception as error:
            _write_message(
                output_stream,
                output_lock,
                {"type": "error", "id": None, "error": f"Invalid request: {error}"},
            )
            continue
        _replace_pending(requests, request)

    requests.put(_STOP)
    worker.join()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nuclear-only", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--flag-heteronyms", action="store_true")
    args = parser.parse_args(argv)

    protocol_output = sys.stdout
    try:
        # Third-party imports occasionally report setup information. Keep the
        # stdout channel exclusively reserved for the JSON protocol.
        with contextlib.redirect_stdout(sys.stderr):
            from stressmark.vim import build_editor_document, editor_payload

        def renderer(text):
            document = build_editor_document(
                text,
                nuclear_only=args.nuclear_only,
                show_rules=args.explain,
                flag_heteronyms=args.flag_heteronyms,
            )
            return editor_payload(document, title="stressmark: pasted text")

        serve(sys.stdin, protocol_output, renderer=renderer)
    except Exception as error:
        _write_message(
            protocol_output,
            threading.Lock(),
            {"type": "fatal", "error": f"{type(error).__name__}: {error}"},
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
