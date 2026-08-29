import io
import json
import queue

from stressmark.vim_worker import _replace_pending, serve


def _messages(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_worker_emits_ready_and_unicode_result():
    requests = io.StringIO(json.dumps({"id": 7, "text": "naïve"}) + "\n")
    responses = io.StringIO()

    serve(requests, responses, renderer=lambda text: {"text": text.upper()})

    assert _messages(responses) == [
        {"type": "ready"},
        {"type": "result", "id": 7, "payload": {"text": "NAÏVE"}},
    ]


def test_worker_reports_request_failure_without_breaking_protocol():
    requests = io.StringIO(json.dumps({"id": 4, "text": "broken"}) + "\n")
    responses = io.StringIO()

    def fail(_text):
        raise RuntimeError("no pronunciation")

    serve(requests, responses, renderer=fail)

    messages = _messages(responses)
    assert messages[0] == {"type": "ready"}
    assert messages[1]["type"] == "error"
    assert messages[1]["id"] == 4
    assert "no pronunciation" in messages[1]["error"]


def test_worker_pending_queue_keeps_only_the_newest_request():
    requests = queue.Queue(maxsize=1)
    _replace_pending(requests, {"id": 1})
    _replace_pending(requests, {"id": 2})

    assert requests.get_nowait() == {"id": 2}
