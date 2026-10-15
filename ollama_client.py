import concurrent.futures
import sys
from typing import Any, Dict, Iterator, List, Optional, Union

import ollama

DEFAULT_MODEL = "llama2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
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


def main():
    client = OllamaClient()

    if not client.check_connection():
        print("Ollama is not available. Make sure it is running.")
        return

    models = client.list_models()
    print("\nAvailable models:")
    for model in models:
        print(f" - {model['name']}")

    if not models:
        print("No models installed. Pull one first, e.g. `ollama pull llama3`.")
        return

    model = models[0]["name"]
    prompt = "Tell me something about artificial intelligence in Norwegian."

    print(f"\nUsing model: {model}\n")
    client.stream_generation(prompt, model=model)

    reply = client.system_message("You are a Norwegian AI assistant.", "Hei! Hvem er du?", model=model)
    print(f"\nSystem prompt example: {reply}")


if __name__ == "__main__":
    main()
