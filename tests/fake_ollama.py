"""A tiny fake Ollama HTTP server for tests (runs in a background thread).

Why a fake server instead of mocking the `ollama` library? The tests then exercise the real
code path end to end (library -> HTTP -> our code), so they catch mistakes that a mock would
hide, and they need neither a real Ollama nor a model. The price is that the fake only knows
the few endpoints the app uses, with simple deterministic answers.
"""
import http.server
import json
import threading
import time
from datetime import datetime, timedelta, timezone


def stats(body, text):
    # Deterministic numbers let the tests assert exact results: with these values a 2-word reply
    # is 2 tokens in 0.5 s = 4.0 tokens/s.
    """Fixed timing fields: 10 prompt tokens per message, one reply token per word, 0.5 s of eval."""
    return {
        "prompt_eval_count": 10 * len(body["messages"]),
        "eval_count": len(text.split(" ")),
        "eval_duration": 500_000_000,
        "load_duration": 100_000_000,
        "total_duration": 700_000_000,
    }


class FakeOllama:
    """Serves /api/tags, /api/chat and /api/embed and records the chat/embed request bodies.

    Chat replies echo "msgs=<count> last=<text>". A prompt of "boom" gets an HTTP 500 and
    "slow" is answered after `slow_seconds`. The next `fail_next` chat requests get an
    HTTP 503. Models named "missing" do not exist for pull/show/delete (HTTP 404).
    """

    def __init__(self, models=("fake:1", "nomic-embed-text:latest")):
        self.models = list(models)
        self.requests = []  # (path, JSON body) of every POST/DELETE, so tests can inspect what was sent
        self.fail_next = 0  # answer this many chat requests with HTTP 503 (for the retry tests)
        self.context_length = 4096
        self.running = [self.models[0]] if self.models else []  # what /api/ps reports as loaded
        self.slow_seconds = 1.0
        # Inside the nested Handler class, `self` is the request handler, not the FakeOllama.
        # `outer` is how the handler reaches the fake's state (models, requests, ...).
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence the default one-line-per-request logging

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up (timeout or cancelled reply) before we answered

            def _json(self, obj, status=200):
                """Send a JSON reply. Content-Length lets the client know when the body ends."""
                body = (json.dumps(obj) + "\n").encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            # BaseHTTPRequestHandler calls do_GET / do_POST / do_DELETE by the HTTP method name.
            def do_GET(self):
                if self.path == "/api/ps":
                    return self._json({"models": [
                        {"model": m, "name": m, "size": 4_000_000_000, "size_vram": 3_000_000_000,
                         "context_length": outer.context_length,
                         "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}
                        for m in outer.running
                    ]})
                self._json({"models": [
                    {"model": m, "modified_at": "2026-01-01T00:00:00Z"} for m in outer.models
                ]})

            def _read_body(self):
                """Parse the JSON request body and record it for the test to look at."""
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                return body

            def do_DELETE(self):
                body = self._read_body()
                if body["model"] == "missing" or body["model"] not in outer.models:
                    return self._json({"error": f"model '{body['model']}' not found"}, status=404)
                outer.models.remove(body["model"])
                self._json({})

            def do_POST(self):
                body = self._read_body()
                if self.path == "/api/show":
                    if body["model"] == "missing":
                        return self._json({"error": "model not found"}, status=404)
                    return self._json({
                        "modified_at": "2026-01-01T00:00:00Z",
                        "capabilities": ["completion"], "model_info": {},
                        "details": {"format": "gguf", "family": "llama",
                                    "parameter_size": "7B", "quantization_level": "Q4_0"},
                    })
                if self.path == "/api/pull":
                    if body["model"] == "missing":
                        return self._json({"error": "pull model manifest: file does not exist"}, status=404)
                    # Streaming answers are "ndjson": one JSON object per line, sent as they happen.
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for event in (
                        {"status": "pulling manifest"},
                        {"status": "pulling abc", "digest": "abc", "total": 200, "completed": 50},
                        {"status": "pulling abc", "digest": "abc", "total": 200, "completed": 200},
                        {"status": "success"},
                    ):
                        self.wfile.write((json.dumps(event) + "\n").encode())
                    outer.models.append(body["model"])
                    return
                if self.path == "/api/embed":
                    return self._json({"model": body["model"], "embeddings": [[0.1, 0.2, 0.3]]})

                if self.path == "/api/chat" and outer.fail_next > 0:
                    outer.fail_next -= 1
                    return self._json({"error": "server busy"}, status=503)

                # Chat: the reply echoes how many messages it got and the last one, so tests can
                # check that history is sent ("msgs=3 ...") without needing a real model.
                last = body["messages"][-1]["content"]
                if last == "slow":
                    time.sleep(outer.slow_seconds)
                if last == "boom":
                    return self._json({"error": "model exploded"}, status=500)
                text = f"msgs={len(body['messages'])} last={last}"
                if not body.get("stream", True):
                    return self._json({
                        "model": body["model"],
                        "message": {"role": "assistant", "content": text},
                        "done": True, **stats(body, text),
                    })
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                words = text.split(" ")
                for i, word in enumerate(words):
                    piece = word + (" " if i < len(words) - 1 else "")
                    chunk = {"model": body["model"], "done": False,
                             "message": {"role": "assistant", "content": piece}}
                    self.wfile.write((json.dumps(chunk) + "\n").encode())
                done = {"model": body["model"], "done": True,
                        "message": {"role": "assistant", "content": ""}, **stats(body, text)}
                self.wfile.write((json.dumps(done) + "\n").encode())

        # Port 0 asks the OS for any free port, so tests never clash with a real Ollama (11434)
        # or with each other. Threading = one thread per request, so a "slow" request does not
        # block the others.
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        # daemon=True: if a test forgets stop(), the thread cannot keep the process alive.
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
