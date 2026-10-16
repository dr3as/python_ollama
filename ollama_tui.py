"""Full-screen terminal chat for Ollama, built on Textual.

Run with `python3 ollama_tui.py`. Chats are saved automatically in a SQLite database.

How the file is organised, top to bottom:

1. Small helpers that need no UI (path, time, error text).
2. ChatStore: saving chats in SQLite.
3. Pure formatting/reading functions for the status bar and Resources tab (format_stats,
   format_resources, read_gpus, ...). They take plain data and return plain data, which makes
   them easy to test without starting the app.
4. Small widgets and dialogs (PromptArea, AssistantMessage, ChoiceScreen, ...).
5. The Compare view (ResultColumn, CompareScreen). It collects the choices and shows the
   results; the comparison itself runs in ollama_compare.py.
6. OllamaTui: the application itself.
7. build_parser() and main(): the command line entry point.

The one idea to understand before reading OllamaTui is the THREADING MODEL:

* Textual runs the user interface on one thread, in an asyncio event loop. Anything slow that
  runs there freezes the whole screen.
* The Ollama calls are slow and blocking (a reply takes seconds). So they run in *thread
  workers*: methods decorated with `@work(thread=True)`.
* A worker thread must never touch widgets directly. It hands the result back with
  `self.call_from_thread(function, *args)`, which runs `function` on the UI thread.
* ChatStore (SQLite) is only used from the UI thread, so it needs no locking.
"""
import argparse
import ipaddress
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import ollama
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.suggester import Suggester
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, ListItem, ListView,
    Markdown, ProgressBar, RadioButton, RadioSet, SelectionList, Static, TabbedContent,
    TabPane, TextArea,
)
from textual.widgets.selection_list import Selection

# psutil is optional. Importing it in a try block lets the app start without it; code that needs
# it checks `psutil is None` (see read_host_stats).
try:
    import psutil  # optional: CPU and RAM of this machine
except ImportError:
    psutil = None

from ollama_client import (
    ChatSession, ConfigError, FileError, OllamaClient, build_prompt_with_files,
    config_path_from_argv, connection_defaults, default_config_path, extract_code_blocks,
    hide_attachments, load_config, read_text_file, write_text_file,
)
# format_bytes lives in ollama_compare (which this module already depends on) so there is one copy.
from ollama_compare import ModelRun, compare_models, comparison_markdown, format_bytes, metric_rows

APP_NAME = "ollama_tui"
TITLE_LENGTH = 50  # chat titles (taken from the first message) are cut to this many characters


def default_db_path() -> Path:
    """Where chats are stored, following the XDG convention (~/.local/share/<app>/ by default)."""
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / APP_NAME / "chats.db"


def _now() -> str:
    # UTC and a fixed format: such timestamps sort correctly as plain text, which the
    # ORDER BY in ChatStore.list_chats relies on.
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _describe(error: Exception) -> str:
    """Turn an exception into one short line a user can act on."""
    if isinstance(error, ollama.ResponseError):
        return f"Ollama error: {error.error or error}"
    if isinstance(error, (ConnectionError, httpx.TransportError)):
        # Some httpx errors (timeouts) have an empty message, so fall back to the class name.
        return f"Cannot reach Ollama: {error or type(error).__name__}"
    return str(error) or type(error).__name__


class ChatStore:
    """Saves chats and their messages in SQLite. Use it from one thread only (the UI thread)."""

    # Two tables: `chats` (one row per conversation, with its settings) and `messages` (many
    # rows per chat). A SQLite connection may only be used by the thread that created it,
    # which is why only the UI thread talks to the store.

    def __init__(self, path):
        # ":memory:" is SQLite's name for a throw-away database, used by the tests.
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row  # rows can be read by column name: row["title"]
        # SQLite ignores foreign keys unless asked, per connection. Without this, the
        # ON DELETE CASCADE below would silently do nothing and leave orphaned messages.
        self._db.execute("PRAGMA foreign_keys = ON")
        # IF NOT EXISTS makes this safe to run on every start: it only creates what is missing.
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                model TEXT,
                system TEXT,
                temperature REAL NOT NULL DEFAULT 0.7,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
            -- Model comparisons. Each run is stored as one JSON text (everything ModelRun holds),
            -- so adding a metric later does not need a database migration.
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY,
                prompt TEXT NOT NULL,
                system TEXT,
                temperature REAL NOT NULL,
                judge_model TEXT,
                winner TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS comparison_runs (
                id INTEGER PRIMARY KEY,
                comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                model TEXT NOT NULL,
                data TEXT NOT NULL
            );
        """)

    def close(self) -> None:
        self._db.close()

    def create_chat(self, title: str, model: Optional[str], system: Optional[str],
                    temperature: float) -> int:
        """Insert a new chat and return its id."""
        now = _now()
        # `with self._db:` is a transaction: it commits when the block ends normally and rolls
        # back if an exception escapes, so a failure never leaves half-written data.
        with self._db:
            # Values are passed separately as `?` placeholders, never pasted into the SQL
            # text. That is what prevents SQL injection.
            cur = self._db.execute(
                "INSERT INTO chats (title, model, system, temperature, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (title, model, system, temperature, now, now),
            )
        return cur.lastrowid

    def list_chats(self) -> List[Dict[str, Any]]:
        """All chats, most recently used first (the id breaks ties for chats in the same second)."""
        rows = self._db.execute(
            "SELECT id, title, model, updated_at FROM chats ORDER BY updated_at DESC, id DESC"
        )
        return [dict(r) for r in rows]

    def get_chat(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """One chat with its messages in order, or None if it does not exist."""
        row = self._db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if row is None:
            return None
        chat = dict(row)
        chat["messages"] = [
            {"role": r["role"], "content": r["content"]}
            for r in self._db.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
            )
        ]
        return chat

    def add_messages(self, chat_id: int, messages: List[Dict[str, str]]) -> None:
        """Append messages to a chat and mark the chat as just used."""
        now = _now()
        # Both statements share one transaction: either both happen or neither does.
        with self._db:
            self._db.executemany(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [(chat_id, m["role"], m["content"], now) for m in messages],
            )
            self._db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

    def update_chat(self, chat_id: int, **fields) -> None:
        """Change some columns of a chat, e.g. update_chat(3, title="New", temperature=0.2)."""
        allowed = {"title", "model", "system", "temperature"}
        if not fields or not set(fields) <= allowed:  # `<=` on sets means "is a subset of"
            raise ValueError(f"can only update {sorted(allowed)}")
        # Column names cannot be `?` placeholders, so they ARE pasted into the SQL. That is safe
        # only because the check above restricts them to the fixed allow-list.
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._db:
            self._db.execute(f"UPDATE chats SET {assignments} WHERE id = ?",
                             (*fields.values(), chat_id))

    def delete_chat(self, chat_id: int) -> None:
        """Delete a chat; its messages go too (ON DELETE CASCADE)."""
        with self._db:
            self._db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    # ---- model comparisons (see ollama_compare.py) ----

    def save_comparison(self, prompt: str, system: Optional[str], temperature: float,
                        judge_model: Optional[str], runs: List[ModelRun]) -> int:
        """Store a finished comparison with all its runs and return its id."""
        with self._db:
            cur = self._db.execute(
                "INSERT INTO comparisons (prompt, system, temperature, judge_model, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (prompt, system, temperature, judge_model, _now()),
            )
            self._db.executemany(
                "INSERT INTO comparison_runs (comparison_id, position, model, data) VALUES (?, ?, ?, ?)",
                [(cur.lastrowid, position, run.model, json.dumps(run.to_dict()))
                 for position, run in enumerate(runs)],
            )
        return cur.lastrowid

    def list_comparisons(self) -> List[Dict[str, Any]]:
        """All saved comparisons, newest first."""
        rows = self._db.execute(
            "SELECT id, prompt, winner, created_at FROM comparisons ORDER BY created_at DESC, id DESC"
        )
        return [dict(r) for r in rows]

    def get_comparison(self, comparison_id: int) -> Optional[Dict[str, Any]]:
        """One comparison with its runs (as dicts, in the order they ran), or None."""
        row = self._db.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
        if row is None:
            return None
        comparison = dict(row)
        comparison["runs"] = [
            json.loads(r["data"]) for r in self._db.execute(
                "SELECT data FROM comparison_runs WHERE comparison_id = ? ORDER BY position",
                (comparison_id,),
            )
        ]
        return comparison

    def set_winner(self, comparison_id: int, model: Optional[str]) -> None:
        """Remember which model the user preferred (None clears the choice)."""
        with self._db:
            self._db.execute("UPDATE comparisons SET winner = ? WHERE id = ?", (model, comparison_id))

    def delete_comparison(self, comparison_id: int) -> None:
        """Delete a comparison; its runs go too (ON DELETE CASCADE)."""
        with self._db:
            self._db.execute("DELETE FROM comparisons WHERE id = ?", (comparison_id,))


# ---- status bar and Resources tab: plain functions ------------------------------------------------

CONTEXT_WARN_PERCENT = 80
CONTEXT_FULL_PERCENT = 100


def context_percent(last: Optional[Dict[str, Any]], context_total: Optional[int]) -> Optional[int]:
    """How full the context window is (approximate, see format_stats), or None if unknown."""
    if last is None or not context_total:
        return None
    return (last["prompt_tokens"] + last["reply_tokens"]) * 100 // context_total


def format_stats(last: Optional[Dict[str, Any]], context_total: Optional[int],
                 session_tokens: int, live: Optional[tuple] = None) -> str:
    """One line of numbers for the status bar. live is (chunks so far, seconds) while streaming."""
    if live is not None:
        chunks, seconds = live
        speed = f" · ~{chunks / seconds:.1f} tok/s" if seconds > 0 else ""
        return f"receiving: ~{chunks:,} tokens{speed}"  # `:,` adds thousands separators
    if last is None:
        return "no replies yet"
    # prompt_tokens is what Ollama evaluated for this request (it can be lower than the whole
    # conversation when part of it was cached), so the context figure is approximate.
    used = last["prompt_tokens"] + last["reply_tokens"]
    if context_total:
        percent = used * 100 // context_total  # `//` is integer division
        warning = " ⚠" if percent >= CONTEXT_WARN_PERCENT else ""
        context = f"ctx ~{used:,}/{context_total:,} ({percent}%){warning}"
    else:
        context = f"ctx ~{used:,}"
    parts = [
        context,
        f"{last['tokens_per_second']:.1f} tok/s",
        f"last {last['prompt_tokens']:,} in / {last['reply_tokens']:,} out",
    ]
    if last["load_seconds"] >= 0.5:  # only mention loading when the model really had to load
        parts.append(f"load {last['load_seconds']:.1f}s")
    parts.append(f"session {session_tokens:,} out")
    return " · ".join(parts)


NVIDIA_SMI_QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"


def is_local_host(url: str) -> bool:
    """True when the Ollama URL points at this machine, so local CPU/RAM/GPU numbers apply."""
    # urlparse only finds the host when the text starts with "//", so add that to bare names
    # such as "localhost:11434".
    host = urlparse(url if "//" in url else f"//{url}").hostname or ""
    if host in ("localhost", socket.gethostname()):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:  # not an IP address (a hostname we do not recognise): treat as remote
        return False
    return address.is_loopback or address.is_unspecified  # 127.x.x.x, ::1, 0.0.0.0


def read_host_stats() -> Optional[Dict[str, float]]:
    """CPU percent and RAM bytes of this machine, or None when psutil is not installed."""
    if psutil is None:
        return None
    memory = psutil.virtual_memory()
    # cpu_percent() reports usage since its previous call, which is why on_mount calls it once
    # to prime it.
    return {"cpu_percent": psutil.cpu_percent(), "ram_used": memory.used, "ram_total": memory.total}


def parse_nvidia_smi(output: str) -> List[Dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=NVIDIA_SMI_QUERY --format=csv,noheader,nounits` (MiB values)."""
    def number(text: str) -> Optional[float]:
        try:
            return float(text)
        except ValueError:
            return None  # e.g. "[N/A]"

    gpus = []
    for line in output.strip().splitlines():  # one line per GPU
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 5:  # skip lines that do not match the query we asked for
            continue
        used, total = number(fields[2]), number(fields[3])
        gpus.append({
            "name": fields[0],
            "util_percent": number(fields[1]),
            # nvidia-smi reports MiB; convert to bytes (2**20 = 1,048,576) like the rest of the app.
            "vram_used": used * 2**20 if used is not None else None,
            "vram_total": total * 2**20 if total is not None else None,
            "temperature": number(fields[4]),
        })
    return gpus


def read_gpus() -> Optional[List[Dict[str, Any]]]:
    """NVIDIA GPUs of this machine, or None when nvidia-smi is missing or fails."""
    if shutil.which("nvidia-smi") is None:  # `which` returns None when the program is not installed
        return None
    try:
        # A list of arguments (not one string) avoids going through a shell. The timeout stops
        # a hung driver from freezing the refresh; check=True turns a non-zero exit code into
        # an exception.
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(result.stdout) or None  # an empty list also means "no GPU data"


def _bar(percent: Optional[float], width: int = 10) -> str:
    """A text meter such as "███░░░░░░░" for 30%."""
    if percent is None:
        return "?" * width
    filled = round(max(0, min(100, percent)) / 100 * width)  # clamp to 0-100 first
    return "█" * filled + "░" * (width - filled)


def _expires_in(expires_at: Optional[datetime]) -> str:
    """"unloads in 4m" for a model Ollama will unload when idle."""
    if expires_at is None:
        return ""
    seconds = int((expires_at - datetime.now(timezone.utc)).total_seconds())
    if seconds <= 0:
        return "unloading"
    return f"unloads in {seconds // 60}m" if seconds >= 60 else f"unloads in {seconds}s"


def format_resources(loaded: Optional[List[Dict[str, Any]]], error: Optional[str], local: bool,
                     host: Optional[Dict[str, float]], gpus: Optional[List[Dict[str, Any]]]) -> str:
    """Text for the Resources tab. loaded comes from Ollama (any host); host and gpus only if local."""
    lines = ["LOADED MODELS (from Ollama)"]
    if error:
        lines.append(f"  {error}")
    elif not loaded:
        lines.append("  none (a model loads on its first request)")
    for m in loaded or []:
        lines.append(f"  {m['name']}")
        size, in_vram = m["size"], m["size_vram"]
        if size and in_vram >= size:
            where = "100% GPU"
        elif size:
            # Part of the model did not fit in GPU memory and runs on the CPU: much slower.
            gpu = in_vram * 100 // size
            where = f"{gpu}% GPU / {100 - gpu}% CPU ⚠"
        else:
            where = "size unknown"
        lines.append(f"    {format_bytes(size)} · {where}")
        extra = [f"ctx {m['context_length']:,}"] if m.get("context_length") else []
        extra.append(_expires_in(m.get("expires_at")))
        lines.append("    " + " · ".join(e for e in extra if e))  # `if e` drops empty parts
    lines.append("")

    # A remote Ollama cannot tell us about its host's CPU, RAM or GPU, so say so instead of
    # showing numbers about the wrong machine.
    if not local:
        lines += ["THIS MACHINE", "  CPU, RAM and GPU load are only",
                  "  shown when Ollama runs on this",
                  "  machine (localhost). The numbers",
                  "  above come from the Ollama server."]
        return "\n".join(lines)

    lines.append("THIS MACHINE")
    if host is None:
        lines.append("  CPU/RAM: install psutil")
        lines.append("  (apt: python3-psutil)")
    else:
        ram_percent = host["ram_used"] * 100 / host["ram_total"] if host["ram_total"] else 0
        lines.append(f"  CPU  {_bar(host['cpu_percent'])} {host['cpu_percent']:.0f}%")
        lines.append(f"  RAM  {_bar(ram_percent)} {ram_percent:.0f}%")
        lines.append(f"       {format_bytes(host['ram_used'])} / {format_bytes(host['ram_total'])}")
    if gpus is None:
        lines.append("  GPU: nvidia-smi not available")
    for gpu in gpus or []:
        lines.append(f"  GPU  {gpu['name']}")
        util = gpu["util_percent"]
        lines.append(f"  load {_bar(util)} {'?' if util is None else f'{util:.0f}'}%")
        used, total = gpu["vram_used"], gpu["vram_total"]
        if used is not None and total:
            lines.append(f"  VRAM {_bar(used * 100 / total)} {used * 100 / total:.0f}%")
            lines.append(f"       {format_bytes(used)} / {format_bytes(total)}")
        if gpu["temperature"] is not None:
            lines.append(f"  temp {gpu['temperature']:.0f} °C")
    return "\n".join(lines)


# ---- widgets and dialogs -------------------------------------------------------------------------


class CommandSuggester(Suggester):
    """Completes slash commands, and model names after /model, /show, /delete and /pull."""

    def __init__(self, commands: List[str], models):
        super().__init__(use_cache=False, case_sensitive=False)
        self.commands = commands
        self.models = models  # callable returning the current model names

    async def get_suggestion(self, value: str) -> Optional[str]:
        """Return the full text the input should become, or None. Textual calls this as you type."""
        if not value.startswith("/"):
            return None
        command, space, rest = value.partition(" ")  # "/model nom" -> ("/model", " ", "nom")
        if not space:  # still typing the command name itself
            return next((c for c in self.commands if c.startswith(command.lower()) and c != command), None)
        if command.lower() in ("/model", "/show", "/delete") and rest:
            # next(generator, None) takes the first match, or None when there is none.
            return next((f"{command} {m}" for m in self.models()
                         if m.lower().startswith(rest.lower()) and m != rest), None)
        return None


HINT_DEFAULT = "Enter sends · Ctrl+J new line · Tab completes /commands"


class PromptArea(TextArea):
    """The message box: Enter sends, Ctrl+J starts a new line, Tab completes a /command.

    It grows with its text (up to MAX_ROWS) and keeps the hint line above it up to date. The hint
    shows the pending completion, which stands in for the inline ghost text of a one-line Input.
    """

    MAX_ROWS = 8
    # Key bindings map a key to an *action*: "submit" runs the method action_submit().
    BINDINGS = [
        # Priority, because TextArea itself would turn Enter into a newline and Tab into an indent.
        Binding("enter", "submit", "Send", show=False, priority=True),
        # Ctrl+J works in every terminal; Shift+Enter only in those that report it separately.
        Binding("ctrl+j,shift+enter", "newline", "New line", show=False, priority=True),
        Binding("tab", "complete", "Complete", show=False, priority=True),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter. The app handles it with @on(PromptArea.Submitted)."""

        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def __init__(self, suggester: Suggester, **kwargs):
        super().__init__(soft_wrap=True, show_line_numbers=False, **kwargs)
        self.suggester = suggester
        self.hint_text = HINT_DEFAULT  # what the hint line shows (also read by the tests)

    def action_submit(self) -> None:
        # The widget does not send anything itself. It announces "the user submitted this" as a
        # message, and the app decides what that means (a command or a chat message).
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")

    async def _suggestion(self) -> Optional[str]:
        """A completion for a one-line /command, or None."""
        text = self.text
        if not text or "\n" in text:  # multi-line text is a message, never a command
            return None
        suggestion = await self.suggester.get_suggestion(text)
        return suggestion if suggestion and suggestion != text else None

    async def action_complete(self) -> None:
        suggestion = await self._suggestion()
        if suggestion:
            self.load_text(suggestion)
            self.move_cursor(self.document.end)
        else:
            self.screen.focus_next()  # nothing to complete: Tab behaves like it normally does

    def _rows(self) -> int:
        """How many text rows the box needs, counting lines that wrap, capped at MAX_ROWS."""
        width = max(self.size.width - 4, 20)  # minus border and padding
        # -(-a // b) is "round up" integer division (ceiling) without importing math.
        rows = sum(max(1, -(-len(line) // width)) for line in self.text.split("\n"))
        return min(rows, self.MAX_ROWS)

    # Textual finds handlers by name: a message class TextArea.Changed is handled by a method
    # called on_text_area_changed (class name in snake_case, prefixed with "on_").
    async def on_text_area_changed(self) -> None:
        self.styles.height = self._rows() + 2  # plus the border
        suggestion = await self._suggestion()
        self.hint_text = f"Tab → {suggestion}" if suggestion else HINT_DEFAULT
        self.screen.query_one("#hint", Static).update(Text(self.hint_text))


class KeyedItem(ListItem):
    """A list row that remembers the model name or chat id it stands for."""

    def __init__(self, label: str, key):
        super().__init__(Label(label))
        self.key = key


class AssistantMessage(Markdown):
    """A Markdown message that remembers its source text (Markdown itself does not keep it)."""

    def __init__(self, text: str, **kwargs):
        super().__init__(text, **kwargs)
        self.text = text

    def update(self, markdown: str):
        # Override only to record the text, then let Markdown do the real work. The return value
        # is passed on because update() returns something that callers `await`.
        self.text = markdown
        return super().update(markdown)


class UserMessage(Static):
    """A plain-text message that remembers its text."""

    def __init__(self, text: str, **kwargs):
        # Text(...) from rich shows the string literally. Passing a plain str would make Textual
        # read things like "[bold]" in a user's message as formatting markup. Attached files are
        # shown as a "📎 name" line; the full text (self.text) is what the model gets.
        super().__init__(Text(hide_attachments(text)), **kwargs)
        self.text = text


# A ModalScreen is a dialog drawn on top of the app that blocks the screen below it. The type in
# brackets, ModalScreen[bool], is what the dialog "returns" via self.dismiss(value).


class ChoiceScreen(ModalScreen[Optional[str]]):
    """A question with one button per choice. Esc dismisses with None."""

    BINDINGS = [Binding("escape", "dismiss_none", "Cancel")]

    def __init__(self, question: str, choices: List[tuple]):
        super().__init__()
        self.question = question
        self.choices = choices  # (label, value) pairs

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.question)
            with Vertical(classes="choices"):
                for index, (label, _) in enumerate(self.choices):
                    yield Button(label, id=f"choice-{index}")

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        # The button id is "choice-<index>"; the index picks the matching (label, value) pair.
        self.dismiss(self.choices[int(event.button.id.split("-")[1])][1])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class InfoScreen(ModalScreen[None]):
    """A read-only dialog with a title and some text (used by /help and model info)."""

    BINDINGS = [Binding("escape,enter,q", "dismiss_screen", "Close")]

    def __init__(self, title: str, body: str):
        super().__init__()
        self.heading = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.heading, classes="dialog-title")
            yield Static(Text(self.body))
            yield Button("Close", id="close")

    @on(Button.Pressed)
    def _close_pressed(self) -> None:
        self.dismiss(None)

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no question. Dismisses with True for Yes, False for No or Esc."""

    BINDINGS = [Binding("escape,n", "answer(False)", "No"), Binding("y", "answer(True)", "Yes")]

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.question)
            with Horizontal(classes="buttons"):
                yield Button("Yes", id="yes", variant="error")
                yield Button("No", id="no", variant="primary")

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, answer: bool) -> None:
        # Bindings can pass arguments: "answer(True)" above calls action_answer(True).
        self.dismiss(answer)


# ---- the Compare view ------------------------------------------------------------------------------
#
# A comparison runs the same prompt on several models, one after another (the logic is in
# ollama_compare.py). This screen collects the choices, starts that in a worker thread and shows
# the answers side by side with a table of numbers underneath.


class ResultColumn(Vertical):
    """One model's column in the Compare view: its name, a status line, the answer, a pick button."""

    def __init__(self, model: str):
        super().__init__(classes="cmp-column")
        self.model = model
        self.status_text = ""  # also read by the tests
        self.answer_text = ""

    def compose(self) -> ComposeResult:
        yield Label(self.model, classes="cmp-col-title")
        yield Static("waiting...", classes="cmp-col-status")
        with VerticalScroll(classes="cmp-col-scroll"):
            yield AssistantMessage("", classes="cmp-col-answer")
        yield Button("★ I prefer this one", classes="cmp-pick")

    def set_status(self, text: str) -> None:
        self.status_text = text
        self.query_one(".cmp-col-status", Static).update(Text(text))

    async def set_answer(self, text: str) -> None:
        self.answer_text = text
        await self.query_one(".cmp-col-answer", AssistantMessage).update(text)

    def mark_winner(self, is_winner: bool) -> None:
        self.set_class(is_winner, "winner")
        self.query_one(".cmp-pick", Button).label = "★ Your pick" if is_winner else "★ I prefer this one"


def run_summary(run: ModelRun) -> str:
    """The status line under a model's name once it has finished, e.g. "ok · 12.3 s · 41.0 tok/s"."""
    if run.status == "error":
        return f"error: {run.error}"
    if run.status in ("waiting", "skipped"):
        return run.status
    text = f"{run.status} · {run.total_seconds or 0:.1f} s · {run.tokens_per_second:.1f} tok/s"
    if run.judge_score is not None:
        text += f"\njudge: {run.judge_score:.0f}/10 {run.judge_reason}".rstrip()
    return text


class CompareScreen(Screen):
    """Pick models, type a prompt, press Ctrl+R: the answers and numbers appear side by side."""

    # A screen's CSS applies to the whole app, so every id here starts with "cmp-" to avoid
    # clashing with the chat screen's rules.
    CSS = """
    #cmp-body { height: 1fr; }
    #cmp-side { width: 38; border-right: solid $primary; padding: 0 1; }
    #cmp-side SelectionList { height: 8; }
    #cmp-judge-box { height: 9; }
    #cmp-info { color: $text-muted; margin: 1 0; }
    #cmp-history { height: 1fr; }
    #cmp-main { width: 1fr; padding: 0 1; }
    #cmp-prompt { height: 5; }
    #cmp-banner { height: 1; color: $text-muted; }
    #cmp-columns { height: 1fr; overflow-x: auto; }
    .cmp-column { width: 1fr; min-width: 34; border: round $primary-darken-2; padding: 0 1; }
    .cmp-column.winner { border: thick $success; }
    .cmp-col-title { text-style: bold; }
    .cmp-col-status { color: $text-muted; height: auto; min-height: 1; }
    .cmp-col-scroll { height: 1fr; }
    .cmp-pick { width: 100%; }
    #cmp-table { height: 15; }
    .buttons { height: auto; }
    """

    BINDINGS = [
        Binding("ctrl+r", "run", "Run"),
        Binding("f2", "back", "Back to chat"),
        Binding("escape", "stop_or_back", "Stop / back"),
    ]

    def __init__(self, models: List[str]):
        super().__init__()
        self.models = models
        self.running = False
        self.cancel = threading.Event()  # set by Esc; checked by the worker between chunks
        self.runs: List[ModelRun] = []
        self.prompt = ""
        self.judge_model: Optional[str] = None
        self.comparison_id: Optional[int] = None  # None until the comparison has been saved
        self.winner: Optional[str] = None
        self.columns: Dict[str, ResultColumn] = {}
        self.banner_text = ""  # also read by the tests
        self.export_dir: Optional[Path] = None  # where /export writes; None means the current folder
        self._last_paint = 0.0

    @property
    def store(self) -> ChatStore:
        return self.app.store

    # ---- layout ----

    def compose(self) -> ComposeResult:
        with Horizontal(id="cmp-body"):
            with Vertical(id="cmp-side"):
                with TabbedContent():
                    with TabPane("Setup", id="cmp-tab-setup"):
                        yield Label("Models to compare (Space ticks)")
                        yield SelectionList(*[Selection(name, name) for name in self.models],
                                            id="cmp-models")
                        yield Label("Judge (optional)")
                        with VerticalScroll(id="cmp-judge-box"):
                            with RadioSet(id="cmp-judge"):
                                yield RadioButton("No judge", value=True)
                                for name in self.models:
                                    yield RadioButton(name)
                        yield Static(self._info_text(), id="cmp-info")
                        yield Button("Run (Ctrl+R)", id="cmp-run", variant="primary")
                    with TabPane("History", id="cmp-tab-history"):
                        yield ListView(id="cmp-history")
                        with Horizontal(classes="buttons"):
                            yield Button("Export", id="cmp-export")
                            yield Button("Delete", id="cmp-delete", variant="error")
            with Vertical(id="cmp-main"):
                yield Label("Prompt (Enter = new line, Ctrl+R = run)")
                yield TextArea(id="cmp-prompt")
                yield Static("", id="cmp-banner")
                yield Horizontal(id="cmp-columns")
                yield DataTable(id="cmp-table", zebra_stripes=True, cursor_type="none")
        yield Footer()

    def _info_text(self) -> str:
        session = self.app.session
        gpu = "on" if self.app.is_local else "off (Ollama is not on this machine)"
        return (f"Uses the chat's settings:\n  system prompt: {'set' if session.system else 'none'}"
                f"\n  temperature: {session.temperature:g}\n  GPU sampling: {gpu}")

    def on_mount(self) -> None:
        self._set_banner("Each model runs alone: all loaded models are unloaded before it starts.")
        self._refresh_history()
        self.query_one("#cmp-prompt", TextArea).focus()

    def _set_banner(self, text: str) -> None:
        self.banner_text = text
        self.query_one("#cmp-banner", Static).update(Text(text))

    # ---- leaving ----

    def action_back(self) -> None:
        if self.running:
            self.notify("A comparison is running. Press Esc to stop it first.", severity="warning")
            return
        self.dismiss()

    def action_stop_or_back(self) -> None:
        if self.running:
            self.cancel.set()
            self._set_banner("Stopping...")
        else:
            self.dismiss()

    # ---- running ----

    @on(Button.Pressed, "#cmp-run")
    async def action_run(self) -> None:
        if self.running:
            self.notify("A comparison is already running.", severity="warning")
            return
        models = list(self.query_one("#cmp-models", SelectionList).selected)
        prompt = self.query_one("#cmp-prompt", TextArea).text.strip()
        if len(models) < 2:
            self.notify("Tick at least two models.", severity="warning")
            return
        if not prompt:
            self.notify("Type a prompt first.", severity="warning")
            return
        judge_index = self.query_one("#cmp-judge", RadioSet).pressed_index
        self.judge_model = self.models[judge_index - 1] if judge_index > 0 else None
        self.prompt = prompt
        self.comparison_id = None
        self.winner = None
        self.runs = [ModelRun(model=m) for m in models]
        self.cancel.clear()
        self.running = True
        await self._build_columns()
        self._update_table()
        self._set_banner("Starting...")
        session = self.app.session
        self._compare_worker(prompt, models, session.system, session.temperature, self.judge_model)

    @work(thread=True, exit_on_error=False)
    def _compare_worker(self, prompt: str, models: List[str], system: Optional[str],
                        temperature: float, judge_model: Optional[str]) -> None:
        """Runs in a thread: compare_models blocks for minutes. Results go to the UI via call_from_thread."""
        call = self.app.call_from_thread

        def on_text(model: str, text: str, chunks: int, seconds: float) -> None:
            now = time.monotonic()
            if now - self._last_paint >= 0.1:  # repaint at most ten times a second
                self._last_paint = now
                call(self._show_text, model, text)

        gpu_reader = read_gpus if self.app.is_local and read_gpus() else None
        try:
            runs, problem = compare_models(
                self.app.client, prompt, models, system=system, temperature=temperature,
                cancel=self.cancel, gpu_reader=gpu_reader, judge_model=judge_model,
                on_status=lambda model, text: call(self._show_status, model, text),
                on_text=on_text,
                on_run=lambda run: call(self._run_finished, run),
                on_judging=lambda name: call(self._set_banner, f"Judging the answers with {name}..."),
            )
        except Exception as e:
            call(self._failed, e)
            return
        call(self._all_finished, runs, problem)

    async def _build_columns(self) -> None:
        container = self.query_one("#cmp-columns", Horizontal)
        await container.remove_children()
        self.columns = {}
        for run in self.runs:
            column = ResultColumn(run.model)
            self.columns[run.model] = column
            await container.mount(column)
            await self._fill_column(column, run)

    async def _fill_column(self, column: ResultColumn, run: ModelRun) -> None:
        column.set_status(run_summary(run))
        await column.set_answer(run.answer)
        column.mark_winner(run.model == self.winner)

    def _show_status(self, model: str, text: str) -> None:
        self._set_banner(f"{model}: {text}")
        self.columns[model].set_status(text)

    async def _show_text(self, model: str, text: str) -> None:
        await self.columns[model].set_answer(text)

    async def _run_finished(self, run: ModelRun) -> None:
        """One model is done: show its final answer and the numbers collected so far."""
        for index, existing in enumerate(self.runs):
            if existing.model == run.model:
                self.runs[index] = run
        await self._fill_column(self.columns[run.model], run)
        self._update_table()

    async def _all_finished(self, runs: List[ModelRun], judge_problem: Optional[str]) -> None:
        self.runs = runs
        self.running = False
        for run in runs:
            await self._fill_column(self.columns[run.model], run)  # the judge scores arrived late
        self._update_table()
        self.comparison_id = self.store.save_comparison(
            self.prompt, self.app.session.system, self.app.session.temperature,
            self.judge_model, runs,
        )
        self._refresh_history()
        if judge_problem:
            self.notify(f"Judging did not work: {judge_problem}", severity="warning", timeout=10)
        stopped = any(r.status in ("stopped", "skipped") for r in runs)
        self._set_banner(("Stopped. " if stopped else "Done. ")
                         + "Press a model's ★ button to record which answer you preferred.")

    def _failed(self, error: Exception) -> None:
        self.running = False
        self._set_banner("The comparison failed.")
        self.notify(_describe(error), severity="error", timeout=10)

    def _update_table(self) -> None:
        table = self.query_one("#cmp-table", DataTable)
        table.clear(columns=True)
        table.add_column("")
        for run in self.runs:
            table.add_column(run.model)
        for label, cells in metric_rows(self.runs):
            table.add_row(label, *cells)

    # ---- the winner ----

    @on(Button.Pressed, ".cmp-pick")
    def _pick_pressed(self, event: Button.Pressed) -> None:
        column = event.button.parent
        if self.running or self.comparison_id is None:
            self.notify("Wait until the comparison has finished.", severity="warning")
            return
        self.winner = None if self.winner == column.model else column.model  # pressing again clears it
        for other in self.columns.values():
            other.mark_winner(other.model == self.winner)
        self.store.set_winner(self.comparison_id, self.winner)
        self._refresh_history()

    # ---- history, export, delete ----

    def _refresh_history(self) -> None:
        view = self.query_one("#cmp-history", ListView)
        view.clear()
        view.extend(
            KeyedItem(f"{c['prompt'].splitlines()[0][:24]}  ({c['created_at'][:10]})", c["id"])
            for c in self.store.list_comparisons() if c["prompt"].strip()
        )

    @on(ListView.Selected, "#cmp-history")
    async def _history_selected(self, event: ListView.Selected) -> None:
        if self.running:
            self.notify("Wait for the running comparison to finish.", severity="warning")
            return
        data = self.store.get_comparison(event.item.key)
        if data is None:
            return
        self.comparison_id = data["id"]
        self.prompt = data["prompt"]
        self.winner = data["winner"]
        self.judge_model = data["judge_model"]
        self.runs = [ModelRun.from_dict(r) for r in data["runs"]]
        self.query_one("#cmp-prompt", TextArea).load_text(self.prompt)
        await self._build_columns()
        self._update_table()
        self._set_banner(f"Saved comparison from {data['created_at']} UTC.")

    @on(Button.Pressed, "#cmp-export")
    def _export_pressed(self) -> None:
        if self.comparison_id is None:
            self.notify("Run a comparison or open one from the History tab first.", severity="warning")
            return
        data = self.store.get_comparison(self.comparison_id)
        folder = self.export_dir or Path.cwd()
        path = folder / f"compare-{self.comparison_id}.md"
        try:
            path.write_text(comparison_markdown(data), encoding="utf-8")
        except OSError as e:
            self.notify(f"Could not write {path}: {e}", severity="error")
            return
        self.notify(f"Saved {path}")

    @on(Button.Pressed, "#cmp-delete")
    def _delete_pressed(self) -> None:
        item = self.query_one("#cmp-history", ListView).highlighted_child
        if not isinstance(item, KeyedItem):
            self.notify("Select a comparison in the list first.", severity="warning")
            return
        comparison_id = item.key

        async def answered(yes: Optional[bool]) -> None:
            if not yes:
                return
            self.store.delete_comparison(comparison_id)
            if comparison_id == self.comparison_id:
                self.comparison_id = None
                self.runs = []
                self.winner = None
                await self._build_columns()
                self._update_table()
            self._refresh_history()

        self.app.push_screen(ConfirmScreen("Delete this comparison?"), answered)



COMMANDS = {
    "/help": "show this list",
    "/new": "start a new chat (keeps model, system prompt, temperature)",
    "/model [name]": "show or switch the model",
    "/pull name": "download a model",
    "/show [name]": "details about a model",
    "/delete name": "delete a model",
    "/system [text]": "set the system prompt (empty clears it)",
    "/temp [number]": "show or set the temperature (0-2)",
    "/rename title": "rename the current chat",
    "/file path": "attach a text file to your next message (/file alone lists, /file clear removes)",
    "/save [--code] [path]": "save the last reply (or only its code blocks) to a file",
    "/compare": "compare several models on one prompt (also F2)",
    "/quit": "exit",
}
# "/model [name]" -> "/model": keep only the first word, deduplicate with a set, then sort.
COMMAND_NAMES = sorted({c.split()[0] for c in COMMANDS})


# ---- the application -----------------------------------------------------------------------------


class OllamaTui(App):
    """The chat application: a side panel with tabs, the conversation, a status bar and a prompt.

    Rough map of the methods below (they appear in this order):
      layout              compose(), on_mount()
      status bar          _update_status() and friends
      chat                _submit_message() -> _stream_reply() (thread) -> _reply_finished()
      commands            _run_command() and the _apply_*/_cmd_* helpers
      Resources tab       _poll_resources() -> _read_resources() (thread) -> _show_resources()
      models tab          _load_models(), _pull(), _show(), _delete_model()
      chats tab           _load_chat(), _refresh_chats()
      compare             action_compare() opens CompareScreen (F2 or /compare)
    """

    TITLE = "Ollama"

    # Textual styles widgets with CSS-like rules. A few things worth knowing:
    #   #name   selects the widget with that id        .name  selects widgets with that class
    #   1fr     "one fraction": take the remaining space      $primary  a colour from the theme
    # The sidebar is hidden by adding the "hidden" class, which sets display: none.
    CSS = """
    #body { height: 1fr; }
    #sidebar { width: 40; border-right: solid $primary; }
    #sidebar.hidden { display: none; }
    #sidebar ListView { height: 1fr; }
    #sidebar .buttons { height: auto; }
    #sidebar .buttons Button { min-width: 8; margin-right: 1; }
    #main { width: 1fr; }
    #log { height: 1fr; padding: 0 1; }
    .msg { margin: 1 0 0 0; padding: 0 1; }
    .msg.user { border-left: thick $accent; }
    .msg.assistant { border-left: thick $success; }
    .msg.error { border-left: thick $error; }
    #welcome { color: $text-muted; margin: 1 2; }
    #status { height: 2; padding: 0 1; background: $boost; color: $text-muted; }
    #hint { height: 1; padding: 0 2; color: $text-muted; }
    #prompt { margin: 0 1; height: 3; border: round $primary; padding: 0 1; }
    #progress { display: none; margin: 1 0 0 0; }
    #progress.visible { display: block; }
    #pull-status { color: $text-muted; }
    .field-label { margin-top: 1; }
    .dialog { width: 70; max-width: 90%; height: auto; max-height: 90%; padding: 1 2;
              border: thick $primary; background: $surface; }
    InfoScreen, ConfirmScreen, ChoiceScreen { align: center middle; }
    .choices { height: auto; margin-top: 1; }
    .choices Button { width: 100%; margin-bottom: 1; }
    .dialog-title { text-style: bold; margin-bottom: 1; }
    .dialog .buttons { height: auto; margin-top: 1; }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Side panel"),
        Binding("ctrl+n", "new_chat", "New chat"),
        Binding("escape", "cancel", "Stop reply"),
        Binding("f1", "help", "Help"),
        Binding("f2", "compare", "Compare models"),
    ]

    def __init__(self, client: OllamaClient, store: ChatStore, model: Optional[str] = None,
                 host_label: Optional[str] = None, system: Optional[str] = None,
                 temperature: float = 0.7):
        super().__init__()
        self.client = client
        self.store = store
        self.model = model
        self.model_names: List[str] = []
        self.session = ChatSession(client, system=system, temperature=temperature)
        self.chat_id: Optional[int] = None  # None until the first reply is saved
        self.attachments: List[tuple] = []  # (name, text) files waiting for the next message
        self.busy = False  # True while a reply is streaming in
        # A threading.Event is a flag that is safe to share between threads. The UI thread sets it
        # (Esc) and the streaming thread checks it between chunks.
        self._cancel = threading.Event()
        self.context_total: Optional[int] = None  # context window of the loaded model, from /api/ps
        self._context_warned = 0  # highest context warning level already shown for this chat
        self.models_loaded = False  # True once the installed-model list has been fetched
        self.session_tokens = 0  # reply tokens received since the app started
        self._live: Optional[tuple] = None  # (chunks, seconds) while a reply streams in
        self.status_text = ""  # what the status bar shows (also read by the tests)
        self.is_local = is_local_host(client.base_url)
        self.resources_text = ""  # what the Resources tab shows (also read by the tests)
        self._polling = False  # True while a Resources refresh is running
        self.sub_title = host_label or client.base_url  # shown by the Header widget

    # ---- layout -------------------------------------------------------------------------

    @property
    def main_screen(self):
        """The chat screen. App.query_one() only searches the screen on top, which is a dialog
        while one is open, so timers and background workers must look things up here."""
        return self.screen_stack[0]

    def compose(self) -> ComposeResult:
        """Build the widget tree. `with` nests widgets inside a container; `yield` adds one."""
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                with TabbedContent():
                    with TabPane("Models", id="tab-models"):
                        yield ListView(id="models")
                        with Horizontal(classes="buttons"):
                            yield Button("Show", id="show-model")
                            yield Button("Delete", id="delete-model", variant="error")
                            yield Button("Refresh", id="refresh-models")
                        yield Input(placeholder="model to pull, e.g. llama3", id="pull-name")
                        yield Button("Pull", id="pull-model", variant="primary")
                        yield ProgressBar(id="progress", show_eta=False)
                        yield Label("", id="pull-status")
                    with TabPane("Chats", id="tab-chats"):
                        yield ListView(id="chats")
                        with Horizontal(classes="buttons"):
                            yield Button("New", id="new-chat", variant="primary")
                            yield Button("Delete", id="delete-chat", variant="error")
                    with TabPane("Resources", id="tab-resources"):
                        with VerticalScroll():
                            yield Static("Loading...", id="resources")
                    with TabPane("Settings", id="tab-settings"):
                        yield Label("System prompt (Enter to apply)", classes="field-label")
                        yield Input(id="setting-system")
                        yield Label("Temperature (Enter to apply)", classes="field-label")
                        yield Input(id="setting-temp", value=str(self.session.temperature))
            with Vertical(id="main"):
                with VerticalScroll(id="log"):
                    yield Static(self._welcome_text(), id="welcome")
                yield Static("", id="status")
                yield Static(Text(HINT_DEFAULT), id="hint")
                # The lambda is evaluated each time a suggestion is needed, so the suggester
                # always sees the *current* model list, not the empty one from startup.
                yield PromptArea(CommandSuggester(COMMAND_NAMES, lambda: self.model_names), id="prompt")
        yield Footer()

    @staticmethod
    def _welcome_text() -> str:
        return ("Type a message and press Enter; Ctrl+J starts a new line. Commands start with / (try /help).\n"
                "Ctrl+B hides the side panel, Esc stops a running reply, Tab completes a command.")

    def on_mount(self) -> None:
        """Runs once, after the widgets exist. Textual calls on_<event> methods automatically."""
        if self.size.width < 100:  # a 40-column panel would crowd a narrow terminal
            self.main_screen.query_one("#sidebar").add_class("hidden")
        self.main_screen.query_one("#prompt").focus()
        self._refresh_chats()
        self._load_models(select_first=self.model is None)
        self._update_status()
        if psutil is not None:
            psutil.cpu_percent()  # the first call has no earlier sample to compare with
        self.set_interval(2.0, self._poll_resources)  # call it every 2 seconds from now on

    # ---- status and small helpers -------------------------------------------------------------

    def _update_status(self) -> None:
        """Redraw the two-line status bar from the current state."""
        state = "generating... (Esc to stop)" if self.busy else "ready"
        system = "set" if self.session.system else "none"
        stats = format_stats(self.session.last_stats, self.context_total,
                             self.session_tokens, self._live if self.busy else None)
        files = f" · 📎 {len(self.attachments)}" if self.attachments else ""
        self.status_text = (f"model: {self.model or '(none)'} · temp {self.session.temperature:g}"
                            f" · system: {system}{files} · {state}\n{stats}")
        self.main_screen.query_one("#status", Static).update(self.status_text)

    def _clear_stats(self) -> None:
        """Forget per-conversation numbers (new chat, loaded chat or another model)."""
        self.session.last_stats = None
        self.context_total = None
        self._context_warned = 0
        self._live = None

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._update_status()

    def _guard(self) -> bool:
        """True when it is safe to change the conversation; otherwise tell the user."""
        if self.busy:
            self.notify("Wait for the reply to finish, or press Esc to stop it.", severity="warning")
            return False
        return True

    def _fail(self, error: Exception) -> None:
        self.notify(_describe(error), severity="error", timeout=8)

    def _set_model(self, name: str) -> None:
        if name != self.model:
            self._clear_stats()  # the numbers belonged to the previous model
        self.model = name
        self.session.model = name
        if self.chat_id is not None:
            self.store.update_chat(self.chat_id, model=name)
        self._update_status()

    # ---- actions ------------------------------------------------------------------------------
    # An "action" is a method named action_<name>. Key bindings (BINDINGS above) and the
    # self.action_...() calls elsewhere trigger them.

    def action_toggle_sidebar(self) -> None:
        self.main_screen.query_one("#sidebar").toggle_class("hidden")

    async def action_new_chat(self) -> None:
        if not self._guard():
            return
        self.chat_id = None
        self.attachments = []
        # A fresh session keeps the current settings (system prompt, temperature) but no history.
        self.session = ChatSession(self.client, model=self.model, system=self.session.system,
                                   temperature=self.session.temperature)
        self._clear_stats()
        self._update_status()
        await self._render_messages([])
        self.main_screen.query_one("#prompt").focus()

    def action_cancel(self) -> None:
        if self.busy:
            self._cancel.set()  # the streaming thread notices this on its next chunk

    def action_compare(self) -> None:
        """Open the Compare view (F2 or /compare)."""
        if not self._guard():
            return
        if len(self.model_names) < 2:
            self.notify("Comparing needs at least two installed models (see /pull).", severity="warning")
            return
        self.push_screen(CompareScreen(list(self.model_names)))

    def action_help(self) -> None:
        width = max(len(c) for c in COMMANDS)
        body = "\n".join(f"{cmd.ljust(width)}  {text}" for cmd, text in COMMANDS.items())
        body += ("\n\nEnter  send    Ctrl+J  new line (Shift+Enter in some terminals)"
                 "\nTab  complete the /command or model name shown above the input"
                 "\nCtrl+B  side panel    Ctrl+N  new chat    Esc  stop reply    Ctrl+Q  quit"
                 "\nF2  compare models side by side (Ctrl+R runs, Esc stops, F2 goes back)")
        self.push_screen(InfoScreen("Commands", body))

    # ---- chat log -----------------------------------------------------------------------------

    async def _render_messages(self, messages: List[Dict[str, str]]) -> None:
        """Replace the visible conversation with these messages (or the welcome text if none)."""
        log = self.main_screen.query_one("#log", VerticalScroll)
        # Mounting/removing widgets is asynchronous in Textual, so these calls are awaited.
        await log.remove_children()
        if not messages:
            await log.mount(Static(self._welcome_text(), id="welcome"))
            return
        for m in messages:
            if m["role"] == "user":
                await log.mount(UserMessage(m["content"], classes="msg user"))
            elif m["role"] == "assistant":
                await log.mount(AssistantMessage(m["content"], classes="msg assistant"))
        log.scroll_end(animate=False)

    async def _submit_message(self, text: str) -> None:
        """Show the user's message and an empty reply, then start streaming the answer."""
        if not self._guard():
            return
        if not self.model:
            self.notify("No model selected. Pick one in the side panel or use /pull NAME.",
                        severity="warning")
            return
        log = self.main_screen.query_one("#log", VerticalScroll)
        for welcome in self.main_screen.query("#welcome"):
            await welcome.remove()
        # Attached files travel inside the message, so they are part of the history and the model
        # can refer back to them in later questions. They apply to this one message only.
        text = build_prompt_with_files(text, self.attachments)
        self.attachments = []
        await log.mount(UserMessage(text, classes="msg user"))
        reply = AssistantMessage("…", classes="msg assistant")  # placeholder until text arrives
        await log.mount(reply)
        log.scroll_end(animate=False)
        self.session.model = self.model
        self._cancel.clear()
        self._set_busy(True)
        self._stream_reply(text, reply)  # returns at once: the work happens in another thread

    # The reply is produced in three steps, on two threads:
    #   _stream_reply   (worker thread)  reads chunks from Ollama and hands text to the UI
    #   _show_partial   (UI thread)      paints the text received so far
    #   _reply_finished (UI thread)      saves the exchange and resets the state
    @work(thread=True, exit_on_error=False)
    def _stream_reply(self, text: str, widget: AssistantMessage) -> None:
        parts: List[str] = []
        error: Optional[str] = None
        stopped = False
        try:
            stream = self.session.send(text, stream=True)
            started = time.monotonic()  # monotonic: a clock that never jumps backwards
            last_paint = 0.0
            for chunk in stream:
                if self._cancel.is_set():
                    stream.close()  # also drops the unanswered message from the session
                    stopped = True
                    break
                parts.append(chunk)
                now = time.monotonic()
                # Repainting for every single token would be wasteful; at most ~20 times a second.
                if now - last_paint >= 0.05:
                    last_paint = now
                    # each streamed chunk is about one token
                    self.call_from_thread(self._show_partial, widget, "".join(parts),
                                          (len(parts), now - started))
        except Exception as e:
            # exit_on_error=False above means an exception here would not crash the app, but we
            # still catch it so the error can be shown in the chat.
            error = _describe(e)
        self.call_from_thread(self._reply_finished, widget, text, "".join(parts), error, stopped)

    async def _show_partial(self, widget: AssistantMessage, text: str, live: tuple) -> None:
        self._live = live
        self._update_status()
        await widget.update(text)
        self.main_screen.query_one("#log", VerticalScroll).scroll_end(animate=False)

    async def _reply_finished(self, widget: AssistantMessage, text: str, reply: str,
                              error: Optional[str], stopped: bool) -> None:
        if error:
            widget.add_class("error")
            await widget.update(f"**Error:** {error}")
        elif stopped:
            await widget.update((reply + "\n\n" if reply else "") + "*(stopped; not kept in history)*")
        else:
            # Only a complete reply is saved and counted.
            await widget.update(reply)
            self._save_exchange(text, reply)
            if self.session.last_stats:
                self.session_tokens += self.session.last_stats["reply_tokens"]
            self._fetch_context_total(self.model)
        self.main_screen.query_one("#log", VerticalScroll).scroll_end(animate=False)
        self._live = None
        self._set_busy(False)
        self.main_screen.query_one("#prompt").focus()

    @work(thread=True, exit_on_error=False, group="stats")
    def _fetch_context_total(self, model: Optional[str]) -> None:
        """Ask Ollama which context window the loaded model really runs with."""
        try:
            loaded = {m["name"]: m["context_length"] for m in self.client.running_models()}
        except Exception:
            return  # the number is a nicety; the bar just shows ctx without a total
        self.call_from_thread(self._set_context_total, model, loaded.get(model))

    def _set_context_total(self, model: Optional[str], total: Optional[int]) -> None:
        # The user may have switched model while the request was running; ignore stale answers.
        if model == self.model and total:
            self.context_total = total
            self._update_status()
            self._warn_if_context_full()

    def _warn_if_context_full(self) -> None:
        """Tell the user once per level (80%, 100%) before the model starts forgetting the start."""
        percent = context_percent(self.session.last_stats, self.context_total)
        if percent is None:
            return
        # _context_warned remembers the highest level already shown (0 none, 1 = 80%, 2 = full),
        # so each warning appears once per chat instead of after every reply.
        if percent >= CONTEXT_FULL_PERCENT and self._context_warned < 2:
            self._context_warned = 2
            self.notify("The context window is full. Ollama now drops the oldest messages, so the "
                        "model forgets the start of this chat. Use /new to start a fresh one.",
                        title="Context full", severity="error", timeout=12)
        elif percent >= CONTEXT_WARN_PERCENT and self._context_warned < 1:
            self._context_warned = 1
            self.notify(f"The context window is about {percent}% full. Very long chats make the "
                        "model forget the beginning; consider /new.",
                        title="Context filling up", severity="warning", timeout=10)

    def _save_exchange(self, user_text: str, reply: str) -> None:
        """Store a finished question/answer pair, creating the chat row on the first one."""
        # The chat is created only now (not when it was started) so empty chats are never saved.
        if self.chat_id is None:
            self.chat_id = self.store.create_chat(
                hide_attachments(user_text).replace("\n", " ")[:TITLE_LENGTH] or "Chat",
                self.model, self.session.system, self.session.temperature,
            )
        self.store.add_messages(self.chat_id, [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ])
        self._refresh_chats()

    # ---- prompt input and slash commands -------------------------------------------------------

    # @on(MessageClass) registers a method as the handler for that message type. An optional CSS
    # selector (see the Input.Submitted handlers below) limits it to one widget.
    @on(PromptArea.Submitted)
    async def _prompt_submitted(self, event: PromptArea.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return
        self.main_screen.query_one("#prompt", PromptArea).clear()
        if text.startswith("/") and "\n" not in text:  # pasted multi-line text is never a command
            await self._run_command(text)
        else:
            await self._submit_message(text)

    async def _run_command(self, line: str) -> None:
        """Dispatch a slash command such as "/model llama3"."""
        name, _, arg = line.partition(" ")
        name, arg = name.lower(), arg.strip()
        if name == "/help":
            self.action_help()
        elif name == "/quit":
            self.exit()
        elif name == "/new":
            await self.action_new_chat()
        elif name == "/model":
            await self._cmd_model(arg)
        elif name == "/pull":
            if self._has_arg(arg, "/pull NAME"):
                self._pull(arg)
        elif name == "/show":
            self._show(arg or self.model or "")
        elif name == "/delete":
            if self._has_arg(arg, "/delete NAME"):
                self._confirm_delete_model(arg)
        elif name == "/system":
            self._apply_system(arg)
        elif name == "/temp":
            self._apply_temperature(arg)
        elif name == "/file":
            self._cmd_file(arg)
        elif name == "/save":
            self._cmd_save(arg)
        elif name == "/compare":
            self.action_compare()
        elif name == "/rename":
            if self._has_arg(arg, "/rename TITLE"):
                self._rename_chat(arg)
        else:
            self.notify(f"Unknown command {name}. Try /help.", severity="warning")

    def _has_arg(self, arg: str, usage: str) -> bool:
        """True if the command got an argument; otherwise show its usage and return False."""
        if not arg:
            self.notify(f"Usage: {usage}", severity="warning")
        return bool(arg)

    def _cmd_file(self, arg: str) -> None:
        """/file PATH queues a file for the next message; /file lists; /file clear removes all."""
        if not arg:
            names = ", ".join(name for name, _ in self.attachments)
            self.notify(f"Attached: {names}" if names else "Usage: /file PATH (nothing attached yet)")
        elif arg == "clear":
            self.attachments = []
            self._update_status()
            self.notify("Attachments cleared.")
        else:
            try:
                name, text = read_text_file(arg)
            except FileError as e:
                self.notify(str(e), severity="error")
                return
            # Attaching the same file again replaces the earlier copy (e.g. after editing it).
            self.attachments = [a for a in self.attachments if a[0] != name] + [(name, text)]
            self._update_status()
            self.notify(f"Attached {name} ({len(text.splitlines())} lines). It is sent with your "
                        "next message.")

    def _cmd_save(self, arg: str) -> None:
        """/save [--code] [PATH]: write the last reply (or just its code blocks) to a file."""
        code_only = False
        if arg == "--code" or arg.startswith("--code "):
            code_only, arg = True, arg[len("--code"):].strip()
        replies = [m["content"] for m in self.session.messages if m["role"] == "assistant"]
        if not replies:
            self.notify("There is no reply to save yet.", severity="warning")
            return
        try:
            text = extract_code_blocks(replies[-1]) if code_only else replies[-1]
        except FileError as e:
            self.notify(str(e), severity="warning")
            return
        if not arg:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            arg = f"{'code' if code_only else 'reply'}-{stamp}.{'txt' if code_only else 'md'}"

        def save(overwrite: bool) -> None:
            try:
                path = write_text_file(arg, text, overwrite=overwrite)
            except FileError as e:
                self.notify(str(e), severity="error")
            else:
                self.notify(f"Saved to {path}")

        if Path(arg).expanduser().exists():
            # Never overwrite silently: ask first, and only write if the answer is yes.
            self.push_screen(ConfirmScreen(f"{arg} exists. Overwrite it?"),
                             lambda yes: save(True) if yes else None)
        else:
            save(False)

    async def _cmd_model(self, arg: str) -> None:
        if not arg:
            self.notify(f"Current model: {self.model or '(none)'}")
            return
        if not self._guard():
            return
        # Only complain when the list is known: an empty list may just mean "not loaded yet".
        if self.model_names and arg not in self.model_names:
            self.notify(f"{arg} is not installed. Use /pull {arg} to download it.", severity="warning")
            return
        self._set_model(arg)

    def _apply_system(self, text: str) -> None:
        """Set (or clear, with empty text) the system prompt, including in the running history."""
        if not self._guard():
            return
        # The system prompt is the first message of the history, so replace it there too.
        messages = self.session.messages
        if messages and messages[0]["role"] == "system":
            messages.pop(0)
        if text:
            messages.insert(0, {"role": "system", "content": text})
        self.session.system = text or None
        if self.chat_id is not None:
            self.store.update_chat(self.chat_id, system=self.session.system)
        self.main_screen.query_one("#setting-system", Input).value = text
        self._update_status()
        self.notify("System prompt set." if text else "System prompt cleared.")

    def _apply_temperature(self, text: str) -> None:
        if not text:
            self.notify(f"Temperature: {self.session.temperature:g}")
            return
        try:
            value = float(text)
            if not 0 <= value <= 2:  # chained comparison: 0 <= value and value <= 2
                raise ValueError
        except ValueError:  # covers both "not a number" and the range check above
            self.notify("Temperature must be a number between 0 and 2.", severity="warning")
            return
        self.session.temperature = value
        if self.chat_id is not None:
            self.store.update_chat(self.chat_id, temperature=value)
        self.main_screen.query_one("#setting-temp", Input).value = f"{value:g}"
        self._update_status()

    def _rename_chat(self, title: str) -> None:
        if self.chat_id is None:
            self.notify("Nothing to rename yet: send a message first.", severity="warning")
            return
        self.store.update_chat(self.chat_id, title=title[:TITLE_LENGTH])
        self._refresh_chats()

    # ---- settings tab -------------------------------------------------------------------------

    @on(Input.Submitted, "#setting-system")
    def _system_submitted(self, event: Input.Submitted) -> None:
        self._apply_system(event.value.strip())

    @on(Input.Submitted, "#setting-temp")
    def _temp_submitted(self, event: Input.Submitted) -> None:
        self._apply_temperature(event.value.strip())

    # ---- resources ----------------------------------------------------------------------------

    def _resources_visible(self) -> bool:
        return (self.main_screen.query_one(TabbedContent).active == "tab-resources"
                and not self.main_screen.query_one("#sidebar").has_class("hidden"))

    @on(TabbedContent.TabActivated)
    def _tab_activated(self) -> None:
        self._poll_resources()  # refresh at once when the tab is opened, not after up to 2 s

    def _poll_resources(self) -> None:
        """Refresh the Resources tab, but only while it is on screen and the last read is done."""
        # Skipping when hidden saves network calls and nvidia-smi runs; _polling prevents a slow
        # read from piling up behind the 2-second timer.
        if self._polling or not self._resources_visible():
            return
        self._polling = True
        self._read_resources()

    @work(thread=True, exit_on_error=False, group="resources")
    def _read_resources(self) -> None:
        loaded: Optional[List[Dict[str, Any]]] = None
        error: Optional[str] = None
        try:
            loaded = self.client.running_models()
        except Exception as e:
            error = _describe(e)
        host = read_host_stats() if self.is_local else None
        gpus = read_gpus() if self.is_local else None
        self.call_from_thread(self._show_resources, loaded, error, host, gpus)

    def _show_resources(self, loaded, error, host, gpus) -> None:
        self.resources_text = format_resources(loaded, error, self.is_local, host, gpus)
        self.main_screen.query_one("#resources", Static).update(Text(self.resources_text))
        self._polling = False

    # ---- models -------------------------------------------------------------------------------

    @work(thread=True, exit_on_error=False)
    def _load_models(self, select_first: bool = False) -> None:
        try:
            names = [m["name"] for m in self.client.list_models()]
        except Exception as e:
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._models_loaded, names, select_first)

    async def _models_loaded(self, names: List[str], select_first: bool) -> None:
        self.model_names = names
        self.models_loaded = True
        if select_first and names and not self.model:
            self._set_model(names[0])  # no model given on the command line: take the first one
        elif select_first and not names:
            self.notify("No models installed. Use /pull NAME to download one.", severity="warning")
        view = self.main_screen.query_one("#models", ListView)
        await view.clear()
        await view.extend(KeyedItem(n, n) for n in names)

    def _highlighted_model(self) -> Optional[str]:
        item = self.main_screen.query_one("#models", ListView).highlighted_child
        return item.key if isinstance(item, KeyedItem) else None

    @on(ListView.Selected, "#models")
    async def _model_selected(self, event: ListView.Selected) -> None:
        if self._guard():
            self._set_model(event.item.key)
            self.main_screen.query_one("#prompt").focus()

    @on(Button.Pressed, "#refresh-models")
    def _refresh_pressed(self) -> None:
        self._load_models()

    @on(Button.Pressed, "#show-model")
    def _show_pressed(self) -> None:
        name = self._highlighted_model()
        if name:
            self._show(name)
        else:
            self.notify("Select a model first.", severity="warning")

    @on(Button.Pressed, "#delete-model")
    def _delete_pressed(self) -> None:
        name = self._highlighted_model()
        if name:
            self._confirm_delete_model(name)
        else:
            self.notify("Select a model first.", severity="warning")

    # Stacked @on decorators: this one handler serves both the Pull button and Enter in the field.
    @on(Button.Pressed, "#pull-model")
    @on(Input.Submitted, "#pull-name")
    def _pull_pressed(self) -> None:
        field = self.main_screen.query_one("#pull-name", Input)
        if field.value.strip():
            self._pull(field.value.strip())
            field.value = ""

    # group="model-ops" labels the worker. Workers in a group can be listed or cancelled together.
    @work(thread=True, exit_on_error=False, group="model-ops")
    def _show(self, name: str) -> None:
        if not name:
            self.call_from_thread(self.notify, "No model selected.", severity="warning")
            return
        try:
            info = self.client.show_model(name)
        except Exception as e:
            self.call_from_thread(self._fail, e)
            return
        body = "\n".join(
            f"{key}: {', '.join(value) if isinstance(value, list) else value}"
            for key, value in info.items()
        )
        self.call_from_thread(self.push_screen, InfoScreen(name, body))

    def _pull(self, name: str) -> None:
        # Look the widgets up here on the UI thread and pass them in: the worker thread must not
        # query the widget tree itself.
        self._pull_worker(name, self.main_screen.query_one("#progress", ProgressBar),
                          self.main_screen.query_one("#pull-status", Label))

    @work(thread=True, exit_on_error=False, group="model-ops")
    def _pull_worker(self, name: str, bar: ProgressBar, label: Label) -> None:
        self.call_from_thread(bar.add_class, "visible")
        last_paint = 0.0
        try:
            for p in self.client.pull_model(name):
                now = time.monotonic()
                # A pull sends hundreds of progress events; repaint about 10 times a second, but
                # never skip the final one (completed == total).
                if now - last_paint < 0.1 and p["completed"] != p["total"]:
                    continue
                last_paint = now
                self.call_from_thread(label.update, f"{name}: {p['status']}")
                self.call_from_thread(bar.update, total=p["total"] or None, progress=p["completed"] or 0)
        except Exception as e:
            self.call_from_thread(self._fail, e)
        else:  # runs only when the loop finished without an exception
            self.call_from_thread(self.notify, f"Pulled {name}")
            self._load_models()
        finally:  # runs always, so the progress bar is hidden on success AND on failure
            self.call_from_thread(bar.remove_class, "visible")
            self.call_from_thread(label.update, "")

    def _confirm_delete_model(self, name: str) -> None:
        # push_screen(screen, callback): the callback receives whatever the dialog dismissed with.
        def answered(yes: Optional[bool]) -> None:
            if yes:
                self._delete_model(name)
        self.push_screen(ConfirmScreen(f"Delete model {name}?"), answered)

    @work(thread=True, exit_on_error=False, group="model-ops")
    def _delete_model(self, name: str) -> None:
        try:
            self.client.delete_model(name)
        except Exception as e:
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self.notify, f"Deleted {name}")
        if self.model == name:  # the active model is gone, so there is no model to chat with now
            self.model = None
            self.session.model = None
            self.call_from_thread(self._update_status)
        self._load_models()

    # ---- saved chats --------------------------------------------------------------------------

    def _refresh_chats(self) -> None:
        view = self.main_screen.query_one("#chats", ListView)
        chats = self.store.list_chats()
        view.clear()
        view.extend(KeyedItem(f"{c['title']}  ({c['updated_at'][:10]})", c["id"]) for c in chats)

    @on(ListView.Selected, "#chats")
    async def _chat_selected(self, event: ListView.Selected) -> None:
        if self._guard():
            await self._load_chat(event.item.key)

    async def _load_chat(self, chat_id: int) -> None:
        """Open a saved chat: restore its settings and history, and show the messages."""
        chat = self.store.get_chat(chat_id)
        if chat is None:
            self.notify("That chat no longer exists.", severity="warning")
            self._refresh_chats()
            return
        previous_model = self.model
        self.chat_id = chat_id
        self.attachments = []
        self.model = chat["model"] or self.model
        # Rebuild the session from the database. The stored messages are exactly what gets sent
        # to the model again with the next question.
        self.session = ChatSession(self.client, model=self.model, system=chat["system"],
                                   temperature=chat["temperature"])
        self.session.messages.extend(chat["messages"])
        self._clear_stats()
        self.main_screen.query_one("#setting-system", Input).value = chat["system"] or ""
        self.main_screen.query_one("#setting-temp", Input).value = f"{chat['temperature']:g}"
        await self._render_messages(chat["messages"])
        self._update_status()
        self.main_screen.query_one("#prompt").focus()
        if self.models_loaded and chat["model"] and chat["model"] not in self.model_names:
            self._ask_about_missing_model(chat["model"], previous_model)

    def _ask_about_missing_model(self, missing: str, previous: Optional[str]) -> None:
        """The chat was made with a model that is no longer installed: offer pull or switch."""
        choices = [(f"Pull {missing}", "pull")]
        fallback = previous if previous in self.model_names else (self.model_names or [None])[0]
        if fallback:
            choices.append((f"Switch to {fallback}", "switch"))
        choices.append(("Keep it (replies will fail until it is installed)", "keep"))

        def answered(choice: Optional[str]) -> None:
            if choice == "pull":
                self._pull(missing)
            # `self.model == missing` guards against the user having changed model meanwhile.
            elif choice == "switch" and fallback and self.model == missing:
                self._set_model(fallback)

        self.push_screen(ChoiceScreen(f"This chat used {missing}, which is not installed.", choices),
                         answered)

    @on(Button.Pressed, "#new-chat")
    async def _new_pressed(self) -> None:
        await self.action_new_chat()

    @on(Button.Pressed, "#delete-chat")
    def _delete_chat_pressed(self) -> None:
        item = self.main_screen.query_one("#chats", ListView).highlighted_child
        if not isinstance(item, KeyedItem):
            self.notify("Select a chat first.", severity="warning")
            return
        chat_id = item.key

        async def answered(yes: Optional[bool]) -> None:
            if not yes:
                return
            self.store.delete_chat(chat_id)
            # If the open chat was deleted, leave it too, so new messages do not go to a
            # chat that no longer exists.
            if chat_id == self.chat_id and self._guard():
                await self.action_new_chat()
            self._refresh_chats()

        self.push_screen(ConfirmScreen("Delete this chat?"), answered)


def build_parser(config: Optional[dict] = None) -> argparse.ArgumentParser:
    """Command line options. They mirror ollama_client.py, plus --db for the chat database.

    `config` is the validated config file (see ollama_client.load_config); it sets the defaults.
    """
    config = config or {}
    d = connection_defaults(config)
    db_default = Path(config["db"]).expanduser() if "db" in config else default_db_path()
    parser = argparse.ArgumentParser(description="Full-screen terminal chat for Ollama.")
    parser.add_argument("--host", default=d["host"],
                        help=f"Ollama server URL (env OLLAMA_HOST, config 'host'; default {d['host']})")
    parser.add_argument("-m", "--model", default=d["model"],
                        help="model name (env OLLAMA_MODEL, config 'model'; default: first installed model)")
    parser.add_argument("--timeout", type=float, default=d["timeout"],
                        help=f"seconds to wait for data from the server (default {d['timeout']:g})")
    parser.add_argument("--retries", type=int, default=d["retries"],
                        help=f"extra attempts after a transient error (default {d['retries']})")
    parser.add_argument("--db", type=Path, default=db_default,
                        help="chat database file (default: %(default)s)")
    parser.add_argument("--config", type=Path,
                        help=f"config file (default {default_config_path()})")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    try:
        config = load_config(config_path_from_argv(argv))
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    args = build_parser(config).parse_args(argv)
    client = OllamaClient(args.host, timeout=args.timeout, retries=args.retries)
    store = ChatStore(args.db)
    app = OllamaTui(client, store, model=args.model, system=config.get("system"),
                    temperature=float(config.get("temperature", 0.7)))
    try:
        app.run()  # blocks until the user quits
    finally:
        store.close()  # close the database even if the app crashed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
