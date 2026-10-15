import argparse
import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Union

import ollama

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_HOST,
        default_model: str = DEFAULT_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client = ollama.Client(host=self.base_url)

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
            for m in self._client.list().models
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

        response = self._client.chat(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            options=options,
        )

        if stream:
            return self._iter_content(response)
        return response.message.content or ""

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
                response = self._client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
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
        response = self._client.chat(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.message.content or ""

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Return the embedding vector for the text."""
        response = self._client.embed(model=model or DEFAULT_EMBED_MODEL, input=text)
        return list(response.embeddings[0])

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
            return reply
        return self._stream_reply(response)

    def _stream_reply(self, response) -> Iterator[str]:
        parts: List[str] = []
        try:
            for content in OllamaClient._iter_content(response):
                parts.append(content)
                yield content
        except GeneratorExit:
            self.messages.pop()  # abandoned stream: drop the unanswered user message
            raise
        except Exception:
            self.messages.pop()
            raise
        self.messages.append({"role": "assistant", "content": "".join(parts)})


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

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaClient(args.host)
    try:
        return args.func(client, args)
    except KeyboardInterrupt:
        return 130
    except ollama.ResponseError as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return 1
    except ConnectionError as e:
        print(f"Cannot reach Ollama at {client.base_url}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
