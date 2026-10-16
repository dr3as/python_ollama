# python_ollama
A Python script that helps you talk to a local Ollama install.

- Main module: `ollama_client.py` (`OllamaClient`, `ChatSession`)
- Requires the `ollama` Python package (`pip install ollama`, preferably in a virtual environment)

## CLI

```bash
python ollama_client.py list                         # installed models
python ollama_client.py ask "What is DNS?"           # single question
cat auth.log | python ollama_client.py ask "Summarise suspicious activity" -s "You are a security analyst"
python ollama_client.py chat                         # interactive, keeps history (/reset, /exit)
python ollama_client.py embed "some text"            # embedding vector as JSON
python ollama_client.py pull llama3                  # download a model with progress
python ollama_client.py show llama3                  # family, size, quantization, capabilities
python ollama_client.py delete llama3                # asks for confirmation (-y to skip)
```

Global options (before the command): `--host URL` (env `OLLAMA_HOST`), `-m/--model NAME`
(env `OLLAMA_MODEL`; default is the first installed model), `--timeout SECONDS` (default 120)
and `--retries N` (default 2).

## Timeouts and retries

The timeout applies to each wait for data from the server, not to the whole request, so a long
streamed answer is fine while data keeps arriving. Non-streaming calls (`generate`, `system_message`,
`embed`, `list_models` and each prompt in `batch_process`) retry network errors, timeouts and
HTTP 429/502/503/504 with exponential backoff (1 s, 2 s, ...). Streaming calls, `pull` and `delete`
are never retried. In Python: `OllamaClient(timeout=30, retries=3, retry_delay=0.5)`.

## Python

```python
from ollama_client import OllamaClient, ChatSession

client = OllamaClient()
session = ChatSession(client, model="llama3", system="Be brief.")
print(session.send("Hi, I'm Andreas."))
print(session.send("What's my name?"))   # remembers the previous turn
```

## Tests

```bash
python -m unittest
```

Uses only the standard library. The tests run against a small fake Ollama server
(`tests/fake_ollama.py`), so no real Ollama or model is needed.
