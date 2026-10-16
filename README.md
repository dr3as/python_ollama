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
sudo apt install python3-psutil    # optional: CPU/RAM numbers in the TUI's Resources tab
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
- **Resources**: refreshed every 2 s while the tab is open. *Loaded models* come from Ollama's
  `/api/ps` and work against any host: size, how much sits in GPU memory versus system RAM
  (`60% GPU / 40% CPU ⚠` means part of the model runs on the CPU, which is much slower), the
  context size and when the model unloads. *This machine* (CPU, RAM, NVIDIA GPU load, VRAM and
  temperature) is shown only when Ollama runs on this computer (`localhost`, a loopback address or
  this machine's hostname), because a remote Ollama cannot report its host's load. CPU/RAM need
  `psutil`, the GPU numbers need `nvidia-smi`; if either is missing the tab says so.
- **Settings**: system prompt and temperature.

The message box takes several lines: **Enter** sends, **Ctrl+J** starts a new line (Shift+Enter
does too in terminals that report it), and pasted text keeps its line breaks. The box grows up to
8 lines. A line above it shows the key help, or the completion for the command you are typing.
Text is only treated as a `/command` when it is a single line, so pasting code that happens to
start with `/` is sent as a message.

Everything is also available as slash commands; press **Tab** to complete a command or model
name: `/help`, `/new`, `/model [name]`, `/pull name`, `/show [name]`, `/delete name`,
`/system [text]`, `/temp [number]`, `/rename title`, `/quit`.
Esc stops a running reply, Ctrl+N starts a new chat, Ctrl+Q quits. Replies are rendered as
Markdown with highlighted code blocks.

The two-line status bar under the chat shows the model, temperature and system prompt, and
below that the numbers: context used/total (`ctx ~1,500/4,096 (36%)`), tokens per second, tokens
in/out for the last reply, model load time (only if noticeable) and total reply tokens this
session. While a reply streams it shows a live token count and speed. The context figure is
approximate: Ollama reports only the tokens it evaluated for the request, which can be fewer
than the whole conversation when part of it was cached. The context total is the window the
model is actually running with, read from Ollama's `/api/ps`.

### Comparing models (F2 or `/compare`)

Press **F2** for a full-screen view that runs one prompt on several models and shows the
answers side by side with a table of numbers. Tick two or more installed models, type a prompt
(Enter = new line), press **Ctrl+R**. It uses the chat's current system prompt and temperature.

**One model at a time, from an empty GPU.** Before each model, every loaded model is unloaded
and the app waits until Ollama's `/api/ps` confirms the memory is free. After the run the model
is unloaded too. This matters on a small GPU (8 GB): a second model loaded next to the first
would spill onto the CPU and ruin the speed numbers. If something still shows as loaded (for
example another user of the same server), the run is kept but gets a warning row. Note that this
unloads *all* models on the server, including the one in your chat (it reloads on your next
message), and that every model starts cold, so *Load time* is a true cold start.

| Row in the table | Meaning |
|---|---|
| First token / Total time | wall-clock seconds from sending the prompt (first token includes loading) |
| Tokens/s | speed of the generation itself, from Ollama's own timing |
| Load time, Prompt/Reply tokens, Context | as reported by Ollama |
| Model memory | size, and how much of it sits in GPU memory (`60% GPU ⚠` = part runs on the CPU) |
| GPU memory used / GPU peak load | peak minus baseline, sampled from `nvidia-smi` every 0.5 s; only when Ollama runs on this machine |
| Judge score | see below |

★ marks the best finished run for the speed rows and the judge score.

* **Judge (optional):** pick a model in the Setup tab. After the runs it reads all answers (in
  random order, without model names) and scores each 1-10. Treat this as a rough hint: small
  local models are unreliable judges, and a judge that is also one of the compared models
  tends to prefer its own style. It runs last, alone, so memory stays clean.
* **I prefer this one:** the button under each answer records your own pick with the
  comparison (press again to clear it).
* **History tab:** every comparison is saved in the chat database. Select one to reopen it.
  **Export** writes `compare-<id>.md` (prompt, metrics table, answers) to the folder you started
  the app from; **Delete** removes it.
* Esc stops the run (the current model stops, the rest are skipped, nothing is judged); F2 or
  Esc goes back to the chat when nothing is running.

The logic is in `ollama_compare.py`, which has no user interface and can be reused from scripts.

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

## Config file

Defaults live in an optional TOML file, `~/.config/python_ollama/config.toml`
(`$XDG_CONFIG_HOME` is respected; use `--config PATH` for another file). The CLI and the TUI read
the same file. Every key is optional:

```toml
host = "http://server:11434"   # --host
model = "llama3"               # --model
timeout = 120                  # --timeout, seconds
retries = 2                    # --retries
temperature = 0.7              # default for ask/chat and the TUI's first chat
system = "You are a security analyst."   # default system prompt
db = "~/chats.db"              # TUI chat database (--db)
```

Order of precedence, highest first: command-line option, environment variable (`OLLAMA_HOST`,
`OLLAMA_MODEL`), config file, built-in default. An unknown key or a value of the wrong type stops
the program with a `Config error` message (exit code 2) rather than being ignored. Reading the
file needs Python 3.11+ (`tomllib`); a missing default file is fine. The TUI's Settings tab and
`/system`, `/temp` still change values for the current chat only; they do not write the file.

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
