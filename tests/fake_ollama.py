"""A tiny fake Ollama HTTP server for tests (runs in a background thread)."""
import http.server
import json
import threading
import time


class FakeOllama:
    """Serves /api/tags, /api/chat and /api/embed and records the chat/embed request bodies.

    Chat replies echo "msgs=<count> last=<text>". A prompt of "boom" gets an HTTP 500 and
    "slow" is answered after `slow_seconds`. The next `fail_next` chat requests get an
    HTTP 503. Models named "missing" do not exist for pull/show/delete (HTTP 404).
    """

    def __init__(self, models=("fake:1", "nomic-embed-text:latest")):
        self.models = list(models)
        self.requests = []
        self.fail_next = 0
        self.slow_seconds = 1.0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up (timeout or cancelled reply) before we answered

            def _json(self, obj, status=200):
                body = (json.dumps(obj) + "\n").encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._json({"models": [
                    {"model": m, "modified_at": "2026-01-01T00:00:00Z"} for m in outer.models
                ]})

            def _read_body(self):
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
                        "done": True,
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
                        "message": {"role": "assistant", "content": ""}}
                self.wfile.write((json.dumps(done) + "\n").encode())

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
