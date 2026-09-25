"""A localhost stand-in for the Hugging Face Hub (PLAN §9: mocked endpoints).

Serves the two API shapes huggingface_hub needs for a snapshot — the repo tree
listing and per-file resolve URLs with ETag/commit headers and Range support —
so tests exercise the real client against a controlled server instead of
monkeypatching its internals. Unknown paths are recorded as UNHANDLED lines.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from fixtures.fake_llama_server import free_port

__all__ = ["FakeHub", "free_port", "serve"]


class FakeHub:
    """In-memory repos: {repo_id: {path: bytes}} plus a request log."""

    def __init__(self) -> None:
        self.repos: dict[str, dict[str, bytes]] = {}
        self.requests: list[str] = []

    def add_repo(self, repo_id: str, files: dict[str, bytes]) -> None:
        self.repos[repo_id] = files


def _oid(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _commit(repo_id: str) -> str:
    return hashlib.sha256(repo_id.encode()).hexdigest()[:40]


def _make_handler(hub: FakeHub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            """Tests read hub.requests; the default stderr log would be noise."""

        def do_GET(self) -> None:
            self._handle(with_body=True)

        def do_HEAD(self) -> None:
            self._handle(with_body=False)

        # --- response helpers -------------------------------------------------

        def _json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _not_found(self, message: str) -> None:
            self._json({"error": message}, status=404)

        # --- routing ----------------------------------------------------------

        def _handle(self, *, with_body: bool) -> None:
            path = unquote(urlparse(self.path).path)
            hub.requests.append(f"{self.command} {path}")

            if path.startswith("/api/models/"):
                self._api_model(path.removeprefix("/api/models/"))
                return
            if "/resolve/" in path:
                self._resolve(path, with_body=with_body)
                return
            hub.requests.append(f"UNHANDLED {path}")
            self._not_found("not found")

        def _api_model(self, rest: str) -> None:
            # /<repo>/revision/<rev> — revision resolution (id/sha/siblings).
            parts = rest.split("/")
            if len(parts) == 4 and parts[2] == "revision":
                repo_id = "/".join(parts[:2])
                files = hub.repos.get(repo_id)
                if files is None:
                    self._not_found("Repository not found")
                    return
                self._json(
                    {
                        "id": repo_id,
                        "sha": _commit(repo_id),
                        "siblings": [{"rfilename": name} for name in sorted(files)],
                    }
                )
                return
            # /<repo>/tree/<revision>[/sub/path] — snapshot's file listing.
            if len(parts) >= 4 and parts[2] == "tree":
                repo_id = "/".join(parts[:2])
                files = hub.repos.get(repo_id)
                if files is None:
                    self._not_found("Repository not found")
                    return
                self._json(
                    [
                        {"type": "file", "oid": _oid(data), "size": len(data), "path": name}
                        for name, data in sorted(files.items())
                    ]
                )
                return
            self._not_found("unsupported api path")

        def _resolve(self, path: str, *, with_body: bool) -> None:
            # /<repo>/resolve/<revision>/<file path in repo>
            repo_id, _, tail = path.lstrip("/").partition("/resolve/")
            _revision, _, file_path = tail.partition("/")
            files = hub.repos.get(repo_id)
            if files is None or file_path not in files:
                self._not_found("file not found")
                return
            data = files[file_path]
            etag = f'"{_oid(data)}"'
            status = 200
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                first, _, last = range_header[len("bytes=") :].partition(",")
                start_text, _, end_text = first.partition("-")
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else len(data) - 1
                data = data[start : end + 1]
                status = 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("X-Repo-Commit", _commit(repo_id))
            self.end_headers()
            if with_body:
                self.wfile.write(data)

    return Handler


@contextmanager
def serve(hub: FakeHub) -> Iterator[str]:
    """Run the fake Hub on a free port; yield the HF_ENDPOINT base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(hub))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
