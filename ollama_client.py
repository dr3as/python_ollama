"""Talk to a local Ollama server from Python or the command line.

How the file is organised, top to bottom:

1. Constants and two small helpers (chat_stats, _is_transient).
2. OllamaClient: a thin wrapper around the official `ollama` package that adds timeouts,
   retries and a few convenience methods (batch_process, model management, ...).
3. ChatSession: a conversation that remembers its history.
4. The config file (load_config) that supplies defaults to both command lines.
5. File helpers: attaching text files to a prompt and saving an answer to a file.
6. The command line interface: one `_cmd_*` function per sub-command, wired up in
   build_parser(), and started by main().

The full-screen app in ollama_tui.py builds on OllamaClient and ChatSession.
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import httpx
import ollama

try:
    import tomllib  # standard library from Python 3.11
except ImportError:  # Python 3.10: the config file is simply unavailable
    tomllib = None

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 120.0  # seconds without data from the server before a request is abandoned
DEFAULT_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0  # first backoff in seconds; doubles on every further retry

# A type variable lets _with_retries say "returns whatever the wrapped call returns", so type
# checkers and editors still know the real result type (str, a list, ...) after a retry wrapper.
T = TypeVar("T")


def chat_stats(response) -> Dict[str, Any]:
    """Token counts and speed from a finished chat response (or the final streamed chunk).

    Ollama reports durations in nanoseconds. Missing values count as 0.
    """
    # Every field can be None (older servers omit them, and a cached prompt reports fewer
    # tokens), so `or 0` turns "missing" into a number we can safely do arithmetic on.
    prompt_tokens = response.prompt_eval_count or 0
    reply_tokens = response.eval_count or 0
    eval_seconds = (response.eval_duration or 0) / 1e9  # 1e9 nanoseconds in a second
    return {
        "prompt_tokens": prompt_tokens,
        "reply_tokens": reply_tokens,
        # Guard the division: a zero duration would raise ZeroDivisionError.
        "tokens_per_second": reply_tokens / eval_seconds if eval_seconds > 0 else 0.0,
        "load_seconds": (response.load_duration or 0) / 1e9,
        "total_seconds": (response.total_duration or 0) / 1e9,
    }


def _is_transient(error: Exception) -> bool:
    """True for errors that may go away on a retry: network trouble, timeouts, overloaded server."""
    if isinstance(error, ollama.ResponseError):
        # 429 = too many requests, 502/503/504 = gateway or server temporarily unavailable.
        # A plain 500 or 404 (e.g. unknown model) will fail again, so it is not retried.
        return error.status_code in (429, 502, 503, 504)
    # httpx.TransportError is the parent of timeouts, refused connections, dropped sockets, ...
    return isinstance(error, (ConnectionError, httpx.TransportError))


class OllamaClient:
    """A small, friendlier layer over `ollama.Client`.

    The underscore-prefixed `_client` is the raw library client. Everything public here adds
    retries, plain-dict results (so callers need not know the library's types), or both.
    """

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
        self.base_url = base_url.rstrip("/")  # "http://x:11434/" and "http://x:11434" are the same
        self.default_model = default_model
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = ollama.Client(host=self.base_url, timeout=timeout)

    def _with_retries(self, call: Callable[[], T]) -> T:
        """Run call(), retrying transient errors with exponential backoff."""
        # `call` is a function with no arguments. Callers wrap their request in a lambda so it
        # can be run again from scratch: self._with_retries(lambda: self._client.chat(...)).
        for attempt in range(self.retries + 1):  # first try + `retries` extra attempts
            try:
                return call()
            except Exception as e:
                # Re-raise on the last attempt, or at once for errors a retry cannot fix.
                if attempt == self.retries or not _is_transient(e):
                    raise
                # Exponential backoff: wait 1x, 2x, 4x ... the base delay, so a struggling
                # server gets breathing room instead of being hammered.
                time.sleep(self.retry_delay * 2 ** attempt)
        # Unreachable: the loop always returns or raises. It keeps type checkers happy.
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
            # Passing the method itself (no parentheses) lets _with_retries call it later.
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
            options["num_predict"] = max_tokens  # Ollama's name for "maximum tokens to generate"

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
        """Turn a stream of response chunks into a stream of plain text pieces."""
        # This is a generator (it uses `yield`): it produces values one at a time, on demand,
        # instead of building a list. That is what lets text appear while the model is typing.
        for chunk in chunks:
            content = chunk.message.content
            if content:  # the last chunk carries statistics but no text
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
            # Never raise out of a worker: one failing prompt must not lose the other results.
            # The error is reported in the result instead.
            try:
                response = self._with_retries(lambda: self._client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                ))
                return index, {"status": "success", "response": response.message.content}
            except Exception as e:
                return index, {"status": "error", "error": str(e)}

        # Threads (not asyncio) fit here: each request spends its time waiting on the network,
        # during which Python releases the GIL, so several can run side by side.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_prompt, i, p) for i, p in enumerate(prompts)]
            # as_completed yields futures in the order they FINISH, not the order submitted.
            for future in concurrent.futures.as_completed(futures):
                index, result = future.result()
                results[index] = result

        # Finishing order is random, so sort by prompt index to give callers a stable result.
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
        # `embeddings` holds one vector per input text; we sent one text, so take the first.
        return list(response.embeddings[0])

    def pull_model(self, model: str) -> Iterator[Dict[str, Any]]:
        """Download a model. Yields progress dicts with status, digest, completed and total."""
        # Not retried: a download streams for minutes and a retry would restart the display.
        for progress in self._client.pull(model, stream=True):
            yield {
                "status": progress.status,
                "digest": progress.digest,  # identifies which layer of the model is downloading
                "completed": progress.completed,
                "total": progress.total,
            }

    def show_model(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Return details about an installed model."""
        info = self._with_retries(lambda: self._client.show(model or self.default_model))
        details = info.details
        # `details` can be missing, so every field is guarded with `if details else None`.
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
                # size_vram < size means part of the model runs on the CPU, which is much slower.
                "size_vram": m.size_vram or 0,
                "context_length": m.context_length,
                "expires_at": m.expires_at,  # when Ollama will unload an idle model
            }
            for m in self._with_retries(self._client.ps).models  # "ps" = process status
        ]

    def unload_model(self, model: str) -> None:
        """Free the model's memory now instead of waiting for Ollama's idle timeout (5 minutes).

        Ollama has no "unload" endpoint. The documented way is a request for that model with
        keep_alive=0 ("keep it loaded for zero seconds"), which unloads it as soon as it is done.
        Used by the model comparison so each model starts from an empty GPU.
        """
        self._with_retries(lambda: self._client.generate(model=model, keep_alive=0))

    def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                    temperature: float = 0.7) -> Iterator[Any]:
        """Start a streamed chat and return the raw chunks (the last one carries the timings).

        Unlike generate(), nothing is stripped away, so callers can time the first chunk and read
        chat_stats() from the final one. Not retried, for the same reason as generate(stream=True).
        """
        return self._client.chat(
            model=model or self.default_model,
            messages=messages,
            stream=True,
            options={"temperature": temperature},
        )

    def chat_once(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                  temperature: float = 0.0, json_format: bool = False) -> str:
        """One non-streamed chat call returning the reply text. json_format asks for valid JSON."""
        response = self._with_retries(lambda: self._client.chat(
            model=model or self.default_model,
            messages=messages,
            options={"temperature": temperature},
            format="json" if json_format else None,
        ))
        return response.message.content or ""

    def delete_model(self, model: str) -> None:
        """Delete an installed model. Raises ollama.ResponseError if it does not exist."""
        # Not retried, so a repeated delete can never hit a model that was already removed.
        self._client.delete(model)

    def list_embeddings_models(self) -> List[str]:
        """List installed models that look like embedding models (name contains 'embed' or 'nomic')."""
        try:
            names = [m["name"] for m in self.list_models()]
        except Exception as e:
            print(f"Failed to fetch models: {e}", file=sys.stderr)
            return []
        # Ollama has no "is this an embedding model?" flag, so we guess from the name.
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
                # end="" keeps the text on one line; flush=True shows it now instead of when
                # Python's output buffer happens to fill up.
                print(content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Streaming failed: {e}", file=sys.stderr)


class ChatSession:
    """A multi-turn conversation that keeps its message history."""

    # The model has no memory between requests. "Remembering" the conversation just means
    # sending the whole history (self.messages) again with every new message.

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
        # The user message goes into the history first because the request sends the whole
        # history. Every failure path below has to undo this, or the history would end with
        # a question that never got an answer.
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
            raise  # bare `raise` re-raises the original error with its traceback

        if not stream:
            reply = response.message.content or ""
            self.messages.append({"role": "assistant", "content": reply})
            self.last_stats = chat_stats(response)
            return reply
        return self._stream_reply(response)

    def _stream_reply(self, response) -> Iterator[str]:
        """Yield the reply piece by piece, and update the history when (and only if) it completes."""
        parts: List[str] = []
        final_chunk = None  # the last chunk carries the timing statistics
        try:
            for chunk in response:
                if chunk.done:
                    final_chunk = chunk
                content = chunk.message.content
                if content:
                    parts.append(content)
                    yield content
        except GeneratorExit:
            # Python raises GeneratorExit inside a generator when its consumer calls .close()
            # (or drops it) before it is exhausted, e.g. the user pressed Esc to stop a reply.
            self.messages.pop()  # abandoned stream: drop the unanswered user message
            raise
        except Exception:
            self.messages.pop()
            raise
        # Only reached when the loop ran to the end without being closed or failing.
        self.messages.append({"role": "assistant", "content": "".join(parts)})
        if final_chunk is not None:
            self.last_stats = chat_stats(final_chunk)


# ---- config file ---------------------------------------------------------------------------------
#
# Defaults can live in a TOML file so you do not retype --host and friends. Precedence, highest
# first: command-line option, environment variable (OLLAMA_HOST / OLLAMA_MODEL), config file,
# built-in default. Both ollama_client.py and ollama_tui.py read the same file.

# Allowed keys and their types. Anything else is rejected, so a typo like `timout = 30` is
# reported instead of being silently ignored.
CONFIG_KEYS = {
    "host": str,
    "model": str,
    "timeout": (int, float),
    "retries": int,
    "temperature": (int, float),
    "system": str,
    "db": str,  # chat database path (used by the TUI only); ~ is expanded
}


class ConfigError(ValueError):
    """The config file cannot be read or contains something invalid."""


def default_config_path() -> Path:
    """~/.config/python_ollama/config.toml, following the XDG convention."""
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config_home) / "python_ollama" / "config.toml"


def config_path_from_argv(argv: Optional[List[str]]) -> Optional[Path]:
    """Find --config in the arguments before the real parsing.

    The config file supplies the *defaults* of the real parser, so it has to be loaded before
    that parser is built. A tiny parser that only knows --config solves the chicken-and-egg
    problem; parse_known_args ignores everything else.
    """
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", type=Path)
    return pre.parse_known_args(argv)[0].config


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read and validate the config file; return {} when the default file does not exist.

    An explicitly given path must exist (you asked for it); the default location is optional.
    """
    explicit = path is not None
    path = path or default_config_path()
    if not path.is_file():
        if explicit:
            raise ConfigError(f"config file not found: {path}")
        return {}
    if tomllib is None:
        raise ConfigError("reading a config file needs Python 3.11 or newer")
    try:
        with open(path, "rb") as f:  # tomllib wants a binary file
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    for key, value in data.items():
        if key not in CONFIG_KEYS:
            raise ConfigError(f"{path}: unknown setting '{key}' (allowed: {', '.join(CONFIG_KEYS)})")
        # bool is a subclass of int in Python, so `retries = true` would pass an int check.
        if isinstance(value, bool) or not isinstance(value, CONFIG_KEYS[key]):
            raise ConfigError(f"{path}: '{key}' has the wrong type ({type(value).__name__})")
    return data


def connection_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults for --host, --model, --timeout and --retries (environment beats config file)."""
    return {
        "host": os.environ.get("OLLAMA_HOST") or config.get("host") or DEFAULT_HOST,
        "model": os.environ.get("OLLAMA_MODEL") or config.get("model"),
        "timeout": float(config.get("timeout", DEFAULT_TIMEOUT)),
        "retries": config.get("retries", DEFAULT_RETRIES),
    }


# ---- attaching files and saving answers ----------------------------------------------------------
#
# Shared by the command line (`ask --file`, `--output`) and the TUI (`/file`, `/save`).

MAX_FILE_BYTES = 200_000  # roughly 50k tokens; more would not fit a typical context window anyway


class FileError(ValueError):
    """A file cannot be attached or saved; the message says why and is safe to show the user."""


def read_text_file(path: Union[str, Path]) -> Tuple[str, str]:
    """Read a UTF-8 text file; return (file name, content). Refuses big and binary files."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileError(f"not a file: {path}")
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise FileError(f"{path.name} is too large ({size:,} bytes, limit {MAX_FILE_BYTES:,}); "
                            "cut out the relevant part first")
        data = path.read_bytes()
    except OSError as e:
        raise FileError(f"cannot read {path}: {e.strerror or e}") from e
    # A NUL byte is the classic sign of a binary file; text files never contain one.
    if b"\0" in data:
        raise FileError(f"{path.name} looks like a binary file")
    try:
        return path.name, data.decode("utf-8").rstrip("\n")
    except UnicodeDecodeError:
        raise FileError(f"{path.name} is not UTF-8 text") from None


# The markers must be unlikely to occur in a real file. hide_attachments() uses the same pattern
# to find the blocks again, so the chat view can show "📎 name" instead of the whole file.
_FILE_BLOCK = re.compile(r"\n*----- file: (?P<name>[^\n]*) -----\n(?P<body>.*?)\n----- end of file -----", re.DOTALL)


def build_prompt_with_files(prompt: str, files: List[Tuple[str, str]]) -> str:
    """Append the files to the prompt, each between marker lines that include its name."""
    blocks = [f"----- file: {name} -----\n{text}\n----- end of file -----" for name, text in files]
    return "\n\n".join([prompt, *blocks]) if prompt else "\n\n".join(blocks)


def hide_attachments(content: str) -> str:
    """Replace every attached file in a message by a one-line summary (for display only)."""
    def summary(m: "re.Match[str]") -> str:
        lines = m.group("body").count("\n") + 1
        return f"\n📎 {m.group('name')} ({lines} line{'s' if lines != 1 else ''})"
    return _FILE_BLOCK.sub(summary, content).strip()


def extract_code_blocks(text: str) -> str:
    """The contents of all ``` fenced blocks in a reply, separated by blank lines."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        raise FileError("the reply contains no code block")
    return "\n".join(b.rstrip("\n") + "\n" for b in blocks)


def write_text_file(path: Union[str, Path], text: str, overwrite: bool = False) -> Path:
    """Write text to a file. Without overwrite=True an existing file is never touched."""
    path = Path(path).expanduser()
    try:
        # Mode "x" fails if the file exists, which is race-free, unlike checking exists() first.
        with open(path, "w" if overwrite else "x", encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else text + "\n")
    except FileExistsError:
        raise FileError(f"{path} already exists") from None
    except OSError as e:
        raise FileError(f"cannot write {path}: {e.strerror or e}") from e
    return path


# ---- command line interface ---------------------------------------------------------------------
#
# Each sub-command (`list`, `ask`, ...) is a function taking the client and the parsed arguments
# and returning the process exit code: 0 = success, 1 = error, 2 = bad usage, which is the
# convention shell scripts rely on. build_parser() maps each command name to its function.


def _resolve_model(client: OllamaClient, requested: Optional[str]) -> str:
    """Use the requested model, or fall back to the first installed one."""
    if requested:
        return requested
    models = client.list_models()
    if not models:
        # SystemExit with a string prints it to stderr and exits with status 1.
        raise SystemExit("No models installed. Pull one first, e.g. `ollama pull llama3`.")
    return models[0]["name"]


def _cmd_list(client: OllamaClient, args: argparse.Namespace) -> int:
    for m in client.list_models():
        print(m["name"])
    return 0


def _cmd_ask(client: OllamaClient, args: argparse.Namespace) -> int:
    parts = [args.prompt] if args.prompt else []
    # isatty() is False when input is piped (`cat log | script ask ...`). Reading stdin only in
    # that case matters: in a terminal, read() would wait forever for the user to press Ctrl+D.
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
    if args.code and not args.output:
        print("--code only makes sense together with --output.", file=sys.stderr)
        return 2
    if not parts and not args.file:
        print("Nothing to ask: give a prompt argument, --file and/or pipe text on stdin.",
              file=sys.stderr)
        return 2
    # Fail before the (slow) model call, not after it: a bad file or an existing output file
    # would otherwise be discovered only once the answer has been generated.
    try:
        files = [read_text_file(path) for path in args.file]
        if args.output and Path(args.output).expanduser().exists() and not args.overwrite:
            raise FileError(f"{args.output} already exists (use --overwrite to replace it)")
    except FileError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    model = _resolve_model(client, args.model)
    session = ChatSession(client, model=model, system=args.system, temperature=args.temperature)
    prompt = build_prompt_with_files("\n\n".join(parts), files)
    if not args.output:
        for chunk in session.send(prompt, stream=True):
            print(chunk, end="", flush=True)
        print()
        return 0

    # With --output nothing is printed; the answer is collected and written to the file instead.
    reply = "".join(session.send(prompt, stream=True))
    try:
        text = extract_code_blocks(reply) if args.code else reply
        path = write_text_file(args.output, text, overwrite=args.overwrite)
    except FileError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Saved {len(text):,} characters to {path}", file=sys.stderr)
    return 0


def _cmd_chat(client: OllamaClient, args: argparse.Namespace) -> int:
    model = _resolve_model(client, args.model)
    session = ChatSession(client, model=model, system=args.system, temperature=args.temperature)
    print(f"Chatting with {model}. Commands: /file PATH attaches a file to your next message, "
          "/save PATH saves the last answer, /reset clears history, /exit quits.")
    attachments: List[Tuple[str, str]] = []
    last_reply = ""
    while True:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D or Ctrl+C: leave quietly
            print()
            return 0
        if not message:
            continue
        if message in ("/exit", "/quit"):
            return 0
        if message == "/reset":
            session.reset()
            attachments.clear()
            print("History cleared.")
            continue
        command, _, argument = message.partition(" ")
        if command in ("/file", "/save"):
            argument = argument.strip()
            if not argument:
                print(f"Usage: {command} PATH", file=sys.stderr)
                continue
            try:
                if command == "/file":
                    attachments.append(read_text_file(argument))
                    print(f"Attached {attachments[-1][0]}; it goes with your next message.")
                elif not last_reply:
                    print("Nothing to save yet.", file=sys.stderr)
                else:
                    path = write_text_file(argument, last_reply)
                    print(f"Saved the last answer to {path}")
            except FileError as e:
                print(f"Error: {e}", file=sys.stderr)
            continue
        try:
            chunks = []
            for chunk in session.send(build_prompt_with_files(message, attachments), stream=True):
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            print()
            last_reply = "".join(chunks)
            attachments.clear()  # a file belongs to one message, like in the TUI
        except Exception as e:
            # A failed message must not end the whole chat; ChatSession already undid it.
            print(f"\nError: {e}", file=sys.stderr)


def _cmd_embed(client: OllamaClient, args: argparse.Namespace) -> int:
    vector = client.embed(args.text, model=args.model)
    print(json.dumps(vector))  # JSON so other tools can parse the output
    return 0


def _cmd_pull(client: OllamaClient, args: argparse.Namespace) -> int:
    live = sys.stdout.isatty()  # rewrite one line in a terminal, plain lines otherwise
    last_status = None
    for p in client.pull_model(args.name):
        if p["total"]:  # a layer is downloading: show a percentage
            percent = (p["completed"] or 0) * 100 // p["total"]
            line = f"{p['status']}: {percent}%"
            if live:
                # "\r" returns to the start of the line and "\033[K" is an ANSI escape that
                # clears the rest of it, so each update overwrites the previous one.
                print(f"\r\033[K{line}", end="", flush=True)
            elif p["completed"] == p["total"] or last_status != p["status"]:
                # When output goes to a file or pipe, print only the first and last line of a
                # layer instead of hundreds of progress lines.
                print(line)
            last_status = p["status"]
        elif p["status"] != last_status:
            # A status without numbers ("pulling manifest", "success"). In a terminal, start a
            # new line first so it does not overwrite the progress line above.
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
        # Without a terminal there is nobody to answer the question, so refuse instead of
        # hanging or guessing: deleting a multi-gigabyte model is not something to do by default.
        if not sys.stdin.isatty():
            print("Refusing to delete without confirmation: use --yes.", file=sys.stderr)
            return 2
        if input(f"Delete model {args.name}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1
    client.delete_model(args.name)
    print(f"Deleted {args.name}")
    return 0


def build_parser(config: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    """Define the command line: global options first, then one sub-parser per command.

    `config` is the validated config file (see load_config); its values become the defaults.
    """
    config = config or {}
    d = connection_defaults(config)
    parser = argparse.ArgumentParser(description="Talk to a local Ollama server.")
    # Global options come BEFORE the command: `script --host X list`, not `script list --host X`.
    parser.add_argument(
        "--host", default=d["host"],
        help=f"Ollama server URL (env OLLAMA_HOST, config 'host'; default {d['host']})",
    )
    parser.add_argument(
        "-m", "--model", default=d["model"],
        help="model name (env OLLAMA_MODEL, config 'model'; default: first installed model)",
    )
    parser.add_argument(
        "--timeout", type=float, default=d["timeout"],
        help=f"seconds to wait for data from the server (default {d['timeout']:g})",
    )
    parser.add_argument(
        "--retries", type=int, default=d["retries"],
        help=f"extra attempts after a transient error (default {d['retries']})",
    )
    parser.add_argument(
        "--config", type=Path,
        help=f"config file (default {default_config_path()})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # set_defaults(func=...) attaches the handler to the command, so main() can simply call
    # args.func(...) without a big if/elif chain.
    sub.add_parser("list", help="list installed models").set_defaults(func=_cmd_list)

    def add_chat_options(p: argparse.ArgumentParser) -> None:
        """Options shared by `ask` and `chat`."""
        p.add_argument("-s", "--system", default=config.get("system"), help="system prompt")
        p.add_argument("-t", "--temperature", type=float,
                       default=float(config.get("temperature", 0.7)))

    p_ask = sub.add_parser("ask", help="ask a single question (stdin is appended to the prompt)")
    p_ask.add_argument("prompt", nargs="?", help="the question; may be omitted if text is piped in")
    p_ask.add_argument("-f", "--file", action="append", default=[], metavar="PATH",
                       help="attach a text file to the prompt (repeatable)")
    p_ask.add_argument("-o", "--output", metavar="PATH",
                       help="write the answer to this file instead of printing it")
    p_ask.add_argument("--code", action="store_true",
                       help="with --output: save only the code blocks of the answer")
    p_ask.add_argument("--overwrite", action="store_true",
                       help="with --output: replace the file if it exists")
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
    """Entry point. Takes argv as a parameter (None means sys.argv) so tests can call it directly."""
    try:
        config = load_config(config_path_from_argv(argv))
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    args = build_parser(config).parse_args(argv)
    client = OllamaClient(args.host, timeout=args.timeout, retries=args.retries)
    # Translate expected failures into a short message and an exit code instead of a traceback.
    try:
        return args.func(client, args)
    except KeyboardInterrupt:
        return 130  # the shell convention: 128 + signal number 2 (SIGINT)
    except ollama.ResponseError as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return 1
    except (ConnectionError, httpx.TransportError) as e:
        print(f"Cannot reach Ollama at {client.base_url}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
