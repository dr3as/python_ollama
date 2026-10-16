"""A tiny fake Ollama HTTP server for tests (runs in a background thread)."""
import http.server
import json
import threading


class FakeOllama:
    """Serves /api/tags, /api/chat and /api/embed and records the chat/embed request bodies.

    Chat replies echo "msgs=<count> last=<text>". A prompt of "boom" gets an HTTP 500.
    """

    def __init__(self, models=("fake:1", "nomic-embed-text:latest")):
        self.models = list(models)
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

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

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if self.path == "/api/embed":
                    return self._json({"model": body["model"], "embeddings": [[0.1, 0.2, 0.3]]})

                last = body["messages"][-1]["content"]
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
