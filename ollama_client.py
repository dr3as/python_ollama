import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

import httpx
import ollama

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 120.0  # seconds without data from the server before a request is abandoned
DEFAULT_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0  # first backoff in seconds; doubles on every further retry

T = TypeVar("T")


def chat_stats(response) -> Dict[str, Any]:
    """Token counts and speed from a finished chat response (or the final streamed chunk).

    Ollama reports durations in nanoseconds. Missing values count as 0.
    """
    prompt_tokens = response.prompt_eval_count or 0
    reply_tokens = response.eval_count or 0
    eval_seconds = (response.eval_duration or 0) / 1e9
    return {
        "prompt_tokens": prompt_tokens,
        "reply_tokens": reply_tokens,
        "tokens_per_second": reply_tokens / eval_seconds if eval_seconds > 0 else 0.0,
        "load_seconds": (response.load_duration or 0) / 1e9,
        "total_seconds": (response.total_duration or 0) / 1e9,
    }


def _is_transient(error: Exception) -> bool:
    """True for errors that may go away on a retry: network trouble, timeouts, overloaded server."""
    if isinstance(error, ollama.ResponseError):
        return error.status_code in (429, 502, 503, 504)
    return isinstance(error, (ConnectionError, httpx.TransportError))


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_HOST,
        default_model: str = DEFAULT_MODEL,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        """timeout is httpx's per-operation limit (e.g. waiting for the next bytes), not a cap on
        the whole request, so a long streamed answer is fine as long as data keeps arriving.
        retries is the number of extra attempts for transient errors on non-streaming calls."""
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = ollama.Client(host=self.base_url, timeout=timeout)

    def _with_retries(self, call: Callable[[], T]) -> T:
        """Run call(), retrying transient errors with exponential backoff."""
        for attempt in range(self.retries + 1):
            try:
                return call()
            except Exception as e:
                if attempt == self.retries or not _is_transient(e):
                    raise
                time.sleep(self.retry_delay * 2 ** attempt)
        raise AssertionError("unreachable")

    def check_connection(self) -> bool:
        """Check whether Ollama is reachable."""
        try:
            self._client.list()
            return True
        except Exception as e:
            print(f"Ollama connection error: {e}", file=sys.stderr)
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """Return the available models."""
        return [
            {"name": m.model, "modified_at": m.modified_at}
            for m in self._with_retries(self._client.list).models
        ]

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Union[str, Iterator[str]]:
        """Generate a response. Returns the full text, or an iterator of text chunks if stream=True."""
        options: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        def call():
            return self._client.chat(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                options=options,
            )

        if stream:  # never retried: chunks may already have been shown to the caller
            return self._iter_content(call())
        return self._with_retries(call).message.content or ""

    @staticmethod
    def _iter_content(chunks) -> Iterator[str]:
        for chunk in chunks:
            content = chunk.message.content
            if content:
                yield content

    def batch_process(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_workers: int = 4,
    ) -> Dict[int, Dict[str, Any]]:
        """Process several prompts concurrently. Results are keyed by prompt index."""
        model = model or self.default_model
        results: Dict[int, Dict[str, Any]] = {}

        def process_prompt(index: int, prompt: str):
            try:
                response = self._with_retries(lambda: self._client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                ))
                return index, {"status": "success", "response": response.message.content}
            except Exception as e:
                return index, {"status": "error", "error": str(e)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_prompt, i, p) for i, p in enumerate(prompts)]
            for future in concurrent.futures.as_completed(futures):
                index, result = future.result()
                results[index] = result

        return dict(sorted(results.items()))

    def system_message(
        self,
        system_prompt: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Send a user prompt together with a system prompt and return the reply."""
        response = self._with_retries(lambda: self._client.chat(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        ))
        return response.message.content or ""

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Return the embedding vector for the text."""
        response = self._with_retries(
            lambda: self._client.embed(model=model or DEFAULT_EMBED_MODEL, input=text)
        )
        return list(response.embeddings[0])

    def pull_model(self, model: str) -> Iterator[Dict[str, Any]]:
        """Download a model. Yields progress dicts with status, digest, completed and total."""
        for progress in self._client.pull(model, stream=True):
            yield {
                "status": progress.status,
                "digest": progress.digest,
                "completed": progress.completed,
                "total": progress.total,
            }

    def show_model(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Return details about an installed model."""
        info = self._with_retries(lambda: self._client.show(model or self.default_model))
        details = info.details
        return {
            "family": details.family if details else None,
            "format": details.format if details else None,
            "parameter_size": details.parameter_size if details else None,
            "quantization_level": details.quantization_level if details else None,
            "capabilities": list(info.capabilities or []),
            "modified_at": info.modified_at,
        }

    def running_models(self) -> List[Dict[str, Any]]:
        """Models currently loaded in memory, with how much of each sits in VRAM."""
        return [
            {
                "name": m.model or m.name,
                "size": m.size or 0,
                "size_vram": m.size_vram or 0,
                "context_length": m.context_length,
                "expires_at": m.expires_at,
            }
            for m in self._with_retries(self._client.ps).models
        ]

    def delete_model(self, model: str) -> None:
        """Delete an installed model. Raises ollama.ResponseError if it does not exist."""
        self._client.delete(model)

    def list_embeddings_models(self) -> List[str]:
        """List installed models that look like embedding models (name contains 'embed' or 'nomic')."""
        try:
            names = [m["name"] for m in self.list_models()]
        except Exception as e:
            print(f"Failed to fetch models: {e}", file=sys.stderr)
            return []
        return [n for n in names if "nomic" in n.lower() or "embed" in n.lower()]

    def run_agent(
        self,
        instructions: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """Run a single prompt with system instructions, streaming internally. Returns None on failure."""
        try:
            response = self._client.chat(
                model=model or self.default_model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            return "".join(self._iter_content(response))
        except Exception as e:
            print(f"Agent run failed: {e}", file=sys.stderr)
            return None

    def stream_generation(self, prompt: str, model: Optional[str] = None) -> None:
        """Print the streamed response as it arrives."""
        try:
            for content in self.generate(prompt, model=model, stream=True):
                print(content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Streaming failed: {e}", file=sys.stderr)


class ChatSession:
    """A multi-turn conversation that keeps its message history."""

    def __init__(
        self,
        client: OllamaClient,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.client = client
        self.model = model or client.default_model
        self.system = system
        self.temperature = temperature
        self.messages: List[Dict[str, str]] = []
        self.last_stats: Optional[Dict[str, Any]] = None  # chat_stats() of the latest finished reply
        self.reset()

    def reset(self) -> None:
        """Clear the history (the system prompt, if any, is kept)."""
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def send(self, message: str, stream: bool = False) -> Union[str, Iterator[str]]:
        """Send a message and return the reply, or an iterator of chunks if stream=True.

        The reply is added to the history once it is complete. If a stream is
        not fully consumed, or the request fails, the user message is removed
        again so the history stays consistent.
        """
        self.messages.append({"role": "user", "content": message})
        try:
            response = self.client._client.chat(
                model=self.model,
                messages=self.messages,
                stream=stream,
                options={"temperature": self.temperature},
            )
        except Exception:
            self.messages.pop()
            raise

        if not stream:
            reply = response.message.content or ""
            self.messages.append({"role": "assistant", "content": reply})
            self.last_stats = chat_stats(response)
            return reply
        return self._stream_reply(response)

    def _stream_reply(self, response) -> Iterator[str]:
        parts: List[str] = []
        final_chunk = None
        try:
            for chunk in response:
                if chunk.done:
                    final_chunk = chunk
                content = chunk.message.content
                if content:
                    parts.append(content)
                    yield content
        except GeneratorExit:
            self.messages.pop()  # abandoned stream: drop the unanswered user message
            raise
        except Exception:
            self.messages.pop()
            raise
        self.messages.append({"role": "assistant", "content": "".join(parts)})
        if final_chunk is not None:
            self.last_stats = chat_stats(final_chunk)


def _resolve_model(client: OllamaClient, requested: Optional[str]) -> str:
    """Use the requested model, or fall back to the first installed one."""
    if requested:
        return requested
    models = client.list_models()
    if not models:
        raise SystemExit("No models installed. Pull one first, e.g. `ollama pull llama3`.")
    return models[0]["name"]


def _cmd_list(client: OllamaClient, args: argparse.Namespace) -> int:
    for m in client.list_models():
        print(m["name"])
    return 0


def _cmd_ask(client: OllamaClient, args: argparse.Namespace) -> int:
    parts = [args.prompt] if args.prompt else []
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
    if not parts:
        print("Nothing to ask: give a prompt argument and/or pipe text on stdin.", file=sys.stderr)
        return 2

    model = _resolve_model(client, args.model)
    session = ChatSession(client, model=model, system=args.system, temperature=args.temperature)
    for chunk in session.send("\n\n".join(parts), stream=True):
        print(chunk, end="", flush=True)
    print()
    return 0


def _cmd_chat(client: OllamaClient, args: argparse.Namespace) -> int:
    model = _resolve_model(client, args.model)
    session = ChatSession(client, model=model, system=args.system, temperature=args.temperature)
    print(f"Chatting with {model}. Commands: /reset clears history, /exit quits.")
    while True:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message in ("/exit", "/quit"):
            return 0
        if message == "/reset":
            session.reset()
            print("History cleared.")
            continue
        try:
            for chunk in session.send(message, stream=True):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


def _cmd_embed(client: OllamaClient, args: argparse.Namespace) -> int:
    vector = client.embed(args.text, model=args.model)
    print(json.dumps(vector))
    return 0


def _cmd_pull(client: OllamaClient, args: argparse.Namespace) -> int:
    live = sys.stdout.isatty()  # rewrite one line in a terminal, plain lines otherwise
    last_status = None
    for p in client.pull_model(args.name):
        if p["total"]:
            percent = (p["completed"] or 0) * 100 // p["total"]
            line = f"{p['status']}: {percent}%"
            if live:
                print(f"\r\033[K{line}", end="", flush=True)
            elif p["completed"] == p["total"] or last_status != p["status"]:
                print(line)
            last_status = p["status"]
        elif p["status"] != last_status:
            print(("\n" if live and last_status else "") + p["status"])
            last_status = p["status"]
    return 0


def _cmd_show(client: OllamaClient, args: argparse.Namespace) -> int:
    info = client.show_model(args.name or _resolve_model(client, args.model))
    for key, value in info.items():
        if isinstance(value, list):
            value = ", ".join(value)
        print(f"{key}: {value}")
    return 0


def _cmd_delete(client: OllamaClient, args: argparse.Namespace) -> int:
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to delete without confirmation: use --yes.", file=sys.stderr)
            return 2
        if input(f"Delete model {args.name}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1
    client.delete_model(args.name)
    print(f"Deleted {args.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Talk to a local Ollama server.")
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
        help=f"Ollama server URL (env OLLAMA_HOST, default {DEFAULT_HOST})",
    )
    parser.add_argument(
        "-m", "--model",
        default=os.environ.get("OLLAMA_MODEL"),
        help="model name (env OLLAMA_MODEL, default: first installed model)",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"seconds to wait for data from the server (default {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_RETRIES,
        help=f"extra attempts after a transient error (default {DEFAULT_RETRIES})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list installed models").set_defaults(func=_cmd_list)

    def add_chat_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("-s", "--system", help="system prompt")
        p.add_argument("-t", "--temperature", type=float, default=0.7)

    p_ask = sub.add_parser("ask", help="ask a single question (stdin is appended to the prompt)")
    p_ask.add_argument("prompt", nargs="?", help="the question; may be omitted if text is piped in")
    add_chat_options(p_ask)
    p_ask.set_defaults(func=_cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive chat with history")
    add_chat_options(p_chat)
    p_chat.set_defaults(func=_cmd_chat)

    p_embed = sub.add_parser("embed", help="print the embedding vector for a text as JSON")
    p_embed.add_argument("text")
    p_embed.set_defaults(func=_cmd_embed)

    p_pull = sub.add_parser("pull", help="download a model, showing progress")
    p_pull.add_argument("name")
    p_pull.set_defaults(func=_cmd_pull)

    p_show = sub.add_parser("show", help="show details about an installed model")
    p_show.add_argument("name", nargs="?", help="model name (default: --model, else the first installed)")
    p_show.set_defaults(func=_cmd_show)

    p_delete = sub.add_parser("delete", help="delete an installed model")
    p_delete.add_argument("name")
    p_delete.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p_delete.set_defaults(func=_cmd_delete)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaClient(args.host, timeout=args.timeout, retries=args.retries)
    try:
        return args.func(client, args)
    except KeyboardInterrupt:
        return 130
    except ollama.ResponseError as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return 1
    except (ConnectionError, httpx.TransportError) as e:
        print(f"Cannot reach Ollama at {client.base_url}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
