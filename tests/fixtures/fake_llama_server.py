"""Scriptable stand-in for llama-server.

Serves the endpoints the workbench depends on — ``/health``, ``/v1/models``,
``/v1/chat/completions`` (SSE), ``/metrics`` — with canned behavior, so the
streaming client and the server manager are testable without a GPU.

Run in-process (``with FakeLlamaServer() as server:``) for client tests, or as
a subprocess via :func:`fake_server_command` for manager lifecycle tests.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    path: str
    body: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class FakeBehavior:
    """Canned responses; ``responses`` is consumed in order, the last entry repeats."""

    models: tuple[str, ...] = ("fake-model",)
    responses: tuple[str, ...] = ("text",)
    ready_after: int = 0
    chunk_delay_s: float = 0.0
    text_chunks: tuple[str, ...] = ("Hello", " from", " fake")
    reasoning_chunks: tuple[str, ...] = ()
    tool_call_id: str = "call_1"
    tool_name: str = "bash"
    tool_arguments: str = '{"command": "ls"}'


class _State:
    def __init__(self, behavior: FakeBehavior) -> None:
        self.behavior = behavior
        self.requests: list[RecordedRequest] = []
        self.health_hits = 0
        self.response_index = 0
        self.lock = threading.Lock()


def _chunk(payload: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {
        "id": "cmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        **payload,
    }


def _text_stream(behavior: FakeBehavior, model: str) -> list[dict[str, Any]]:
    def delta_chunk(delta: dict[str, Any]) -> dict[str, Any]:
        return _chunk(
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            model=model,
        )

    chunks = [
        delta_chunk({"role": "assistant", "reasoning_content": piece})
        for piece in behavior.reasoning_chunks
    ]
    chunks.extend(
        delta_chunk({"role": "assistant", "content": piece})
        for piece in behavior.text_chunks
    )
    stop_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    chunks.append(_chunk(stop_chunk, model=model))
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    chunks.append(_chunk({"choices": [], "usage": usage}, model=model))
    return chunks


def _tool_call_stream(behavior: FakeBehavior, model: str) -> list[dict[str, Any]]:
    call = {
        "index": 0,
        "id": behavior.tool_call_id,
        "type": "function",
        "function": {"name": behavior.tool_name, "arguments": ""},
    }
    start_delta = {"role": "assistant", "tool_calls": [call]}
    chunks = [
        _chunk(
            {"choices": [{"index": 0, "delta": start_delta, "finish_reason": None}]},
            model=model,
        ),
        _chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": behavior.tool_arguments}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            model=model,
        ),
        _chunk(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            model=model,
        ),
        _chunk(
            {
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            },
            model=model,
        ),
    ]
    return chunks


def _make_handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args: object) -> None:
            pass  # keep test output clean

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, body: str, content_type: str) -> None:
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            behavior = state.behavior
            if self.path == "/health":
                with state.lock:
                    state.health_hits += 1
                    ready = state.health_hits > behavior.ready_after
                if ready:
                    self._send_text("ok", "text/plain")
                else:
                    self._send_json({"error": "loading"}, status=503)
            elif self.path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {"id": model, "object": "model", "owned_by": "local"}
                            for model in behavior.models
                        ],
                    }
                )
            elif self.path == "/metrics":
                self._send_text(
                    "# HELP llama_server_requests_total Chat completion requests\n"
                    "# TYPE llama_server_requests_total counter\n"
                    "llama_server_requests_total 1\n",
                    "text/plain; version=0.0.4",
                )
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw) if raw else None
            with state.lock:
                state.requests.append(RecordedRequest("POST", self.path, body))
                behavior = state.behavior
                name = behavior.responses[
                    min(state.response_index, len(behavior.responses) - 1)
                ]
                state.response_index += 1

            if name == "error500":
                self._send_json({"error": {"message": "fake failure"}}, status=500)
                return

            model = (body or {}).get("model", behavior.models[0])
            if name == "tool_call":
                chunks = _tool_call_stream(behavior, model)
            else:
                chunks = _text_stream(behavior, model)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                if behavior.chunk_delay_s:
                    time.sleep(behavior.chunk_delay_s)
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class FakeLlamaServer:
    """Threaded in-process fake server; use as a context manager."""

    def __init__(
        self,
        behavior: FakeBehavior | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._behavior = behavior or FakeBehavior()
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        assert self._server is not None, "server not started"
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self.port}"

    @property
    def requests(self) -> list[RecordedRequest]:
        return list(_state.requests) if (_state := self._state) else []

    def start(self) -> None:
        self._state = _State(self._behavior)
        self._server = ThreadingHTTPServer(
            (self._host, self._port), _make_handler(self._state)
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FakeLlamaServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def free_port() -> int:
    """Reserve an ephemeral loopback port the OS hands out (manager tests)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fake_server_command(
    port: int,
    *,
    ready_after: int = 0,
    responses: str = "text",
    chunk_delay_s: float = 0.0,
) -> list[str]:
    """Command line that runs this fixture as a subprocess (server manager tests)."""
    script = Path(__file__).resolve().parent / "fake_llama_server.py"
    return [
        sys.executable,
        str(script),
        "--port",
        str(port),
        "--ready-after",
        str(ready_after),
        "--responses",
        responses,
        "--chunk-delay",
        str(chunk_delay_s),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scriptable fake llama-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-after", type=int, default=0)
    parser.add_argument("--chunk-delay", type=float, default=0.0)
    parser.add_argument("--responses", default="text", help="comma: text|tool_call|error500")
    parser.add_argument("--models", default="fake-model")
    args = parser.parse_args(argv)

    behavior = FakeBehavior(
        models=tuple(args.models.split(",")),
        responses=tuple(args.responses.split(",")),
        ready_after=args.ready_after,
        chunk_delay_s=args.chunk_delay,
    )
    server = FakeLlamaServer(behavior, host=args.host, port=args.port)
    server.start()
    print(f"fake llama-server listening on {args.host}:{server.port}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
