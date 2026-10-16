# python_ollama
A Python script that helps you talk to a local Ollama install.

- `ollama_client.py`: `OllamaClient`, `ChatSession` and the command-line interface
- `ollama_tui.py`: full-screen terminal chat app (see below)

## Install

**With a virtual environment** (any system with Python 3.10+):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

**Without a venv, from apt** (Ubuntu Server; checked against Ubuntu 26.04, where these packages
are `python3-ollama` 0.6.1 and `python3-textual` 2.1.2):

```bash
sudo apt install python3-ollama python3-textual
```

apt pulls in the rest (`python3-httpx`, `python3-pydantic`, `python3-rich`, ...). The CLI
(`ollama_client.py`) only needs `python3-ollama`; `python3-textual` is for the TUI. If your
Ubuntu release has no `python3-textual`, use the venv method for it. You also need an Ollama
server to talk to (local or remote via `--host`).

## TUI

```bash
python3 ollama_tui.py                      # same --host, -m, --timeout, --retries as the CLI
python3 ollama_tui.py --host http://server:11434 --db ~/chats.db
```

A chat window with a side panel (Ctrl+B toggles it; it starts hidden on terminals narrower
than 100 columns) with three tabs:

- **Models**: installed models (Enter selects one), Show, Delete (asks first), Refresh, and a Pull field with a progress bar.
- **Chats**: saved conversations (Enter loads one), New, Delete.
- **Settings**: system prompt and temperature.

Everything is also available as slash commands, with suggestions while you type (press → to
accept): `/help`, `/new`, `/model [name]`, `/pull name`, `/show [name]`, `/delete name`,
`/system [text]`, `/temp [number]`, `/rename title`, `/quit`.
Esc stops a running reply, Ctrl+N starts a new chat, Ctrl+Q quits. Replies are rendered as
Markdown with highlighted code blocks.

Chats are saved automatically (after each completed reply) in a SQLite database at
`~/.local/share/ollama_tui/chats.db` (`$XDG_DATA_HOME` is respected; override with `--db`).
A reply you stop or that fails is shown but not saved. Esc takes effect when the next piece
of the reply arrives, so it can lag while a model is still loading.

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

Uses only `unittest` (the TUI tests need Textual installed). The tests run against a small fake Ollama server
(`tests/fake_ollama.py`), so no real Ollama or model is needed.
