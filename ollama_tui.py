"""Full-screen terminal chat for Ollama, built on Textual.

Run with `python3 ollama_tui.py`. Chats are saved automatically in a SQLite database.
"""
import argparse
import ipaddress
import os
import shutil
import socket
import sqlite3
import subprocess
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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem, ListView,
    Markdown, ProgressBar, Static, TabbedContent, TabPane,
)

try:
    import psutil  # optional: CPU and RAM of this machine
except ImportError:
    psutil = None

from ollama_client import (
    DEFAULT_HOST, DEFAULT_RETRIES, DEFAULT_TIMEOUT, ChatSession, OllamaClient,
)

APP_NAME = "ollama_tui"
TITLE_LENGTH = 50


def default_db_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / APP_NAME / "chats.db"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _describe(error: Exception) -> str:
    if isinstance(error, ollama.ResponseError):
        return f"Ollama error: {error.error or error}"
    if isinstance(error, (ConnectionError, httpx.TransportError)):
        return f"Cannot reach Ollama: {error or type(error).__name__}"
    return str(error) or type(error).__name__


class ChatStore:
    """Saves chats and their messages in SQLite. Use it from one thread only (the UI thread)."""

    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
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
        """)

    def close(self) -> None:
        self._db.close()

    def create_chat(self, title: str, model: Optional[str], system: Optional[str],
                    temperature: float) -> int:
        now = _now()
        with self._db:
            cur = self._db.execute(
                "INSERT INTO chats (title, model, system, temperature, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (title, model, system, temperature, now, now),
            )
        return cur.lastrowid

    def list_chats(self) -> List[Dict[str, Any]]:
        rows = self._db.execute(
            "SELECT id, title, model, updated_at FROM chats ORDER BY updated_at DESC, id DESC"
        )
        return [dict(r) for r in rows]

    def get_chat(self, chat_id: int) -> Optional[Dict[str, Any]]:
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
        now = _now()
        with self._db:
            self._db.executemany(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [(chat_id, m["role"], m["content"], now) for m in messages],
            )
            self._db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

    def update_chat(self, chat_id: int, **fields) -> None:
        allowed = {"title", "model", "system", "temperature"}
        if not fields or not set(fields) <= allowed:
            raise ValueError(f"can only update {sorted(allowed)}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._db:
            self._db.execute(f"UPDATE chats SET {assignments} WHERE id = ?",
                             (*fields.values(), chat_id))

    def delete_chat(self, chat_id: int) -> None:
        with self._db:
            self._db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def format_stats(last: Optional[Dict[str, Any]], context_total: Optional[int],
                 session_tokens: int, live: Optional[tuple] = None) -> str:
    """One line of numbers for the status bar. live is (chunks so far, seconds) while streaming."""
    if live is not None:
        chunks, seconds = live
        speed = f" · ~{chunks / seconds:.1f} tok/s" if seconds > 0 else ""
        return f"receiving: ~{chunks:,} tokens{speed}"
    if last is None:
        return "no replies yet"
    # prompt_tokens is what Ollama evaluated for this request (it can be lower than the whole
    # conversation when part of it was cached), so the context figure is approximate.
    used = last["prompt_tokens"] + last["reply_tokens"]
    if context_total:
        context = f"ctx ~{used:,}/{context_total:,} ({used * 100 // context_total}%)"
    else:
        context = f"ctx ~{used:,}"
    parts = [
        context,
        f"{last['tokens_per_second']:.1f} tok/s",
        f"last {last['prompt_tokens']:,} in / {last['reply_tokens']:,} out",
    ]
    if last["load_seconds"] >= 0.5:
        parts.append(f"load {last['load_seconds']:.1f}s")
    parts.append(f"session {session_tokens:,} out")
    return " · ".join(parts)


NVIDIA_SMI_QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"


def is_local_host(url: str) -> bool:
    """True when the Ollama URL points at this machine, so local CPU/RAM/GPU numbers apply."""
    host = urlparse(url if "//" in url else f"//{url}").hostname or ""
    if host in ("localhost", socket.gethostname()):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def read_host_stats() -> Optional[Dict[str, float]]:
    """CPU percent and RAM bytes of this machine, or None when psutil is not installed."""
    if psutil is None:
        return None
    memory = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(), "ram_used": memory.used, "ram_total": memory.total}


def parse_nvidia_smi(output: str) -> List[Dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=NVIDIA_SMI_QUERY --format=csv,noheader,nounits` (MiB values)."""
    def number(text: str) -> Optional[float]:
        try:
            return float(text)
        except ValueError:
            return None  # e.g. "[N/A]"

    gpus = []
    for line in output.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 5:
            continue
        used, total = number(fields[2]), number(fields[3])
        gpus.append({
            "name": fields[0],
            "util_percent": number(fields[1]),
            "vram_used": used * 2**20 if used is not None else None,
            "vram_total": total * 2**20 if total is not None else None,
            "temperature": number(fields[4]),
        })
    return gpus


def read_gpus() -> Optional[List[Dict[str, Any]]]:
    """NVIDIA GPUs of this machine, or None when nvidia-smi is missing or fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(result.stdout) or None


def format_bytes(size: float) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{size / 1e6:.0f} MB"


def _bar(percent: Optional[float], width: int = 10) -> str:
    if percent is None:
        return "?" * width
    filled = round(max(0, min(100, percent)) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _expires_in(expires_at: Optional[datetime]) -> str:
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
            gpu = in_vram * 100 // size
            where = f"{gpu}% GPU / {100 - gpu}% CPU ⚠"
        else:
            where = "size unknown"
        lines.append(f"    {format_bytes(size)} · {where}")
        extra = [f"ctx {m['context_length']:,}"] if m.get("context_length") else []
        extra.append(_expires_in(m.get("expires_at")))
        lines.append("    " + " · ".join(e for e in extra if e))
    lines.append("")

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


class CommandSuggester(Suggester):
    """Completes slash commands, and model names after /model, /show, /delete and /pull."""

    def __init__(self, commands: List[str], models):
        super().__init__(use_cache=False, case_sensitive=False)
        self.commands = commands
        self.models = models  # callable returning the current model names

    async def get_suggestion(self, value: str) -> Optional[str]:
        if not value.startswith("/"):
            return None
        command, space, rest = value.partition(" ")
        if not space:
            return next((c for c in self.commands if c.startswith(command.lower()) and c != command), None)
        if command.lower() in ("/model", "/show", "/delete") and rest:
            return next((f"{command} {m}" for m in self.models()
                         if m.lower().startswith(rest.lower()) and m != rest), None)
        return None


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
        self.text = markdown
        return super().update(markdown)


class UserMessage(Static):
    """A plain-text message that remembers its text."""

    def __init__(self, text: str, **kwargs):
        super().__init__(Text(text), **kwargs)
        self.text = text


class InfoScreen(ModalScreen[None]):
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
        self.dismiss(answer)


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
    "/quit": "exit",
}
COMMAND_NAMES = sorted({c.split()[0] for c in COMMANDS})


class OllamaTui(App):
    TITLE = "Ollama"

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
    #prompt { margin: 0 1; }
    #progress { display: none; margin: 1 0 0 0; }
    #progress.visible { display: block; }
    #pull-status { color: $text-muted; }
    .field-label { margin-top: 1; }
    .dialog { width: 70; max-width: 90%; height: auto; max-height: 90%; padding: 1 2;
              border: thick $primary; background: $surface; }
    InfoScreen, ConfirmScreen { align: center middle; }
    .dialog-title { text-style: bold; margin-bottom: 1; }
    .dialog .buttons { height: auto; margin-top: 1; }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Side panel"),
        Binding("ctrl+n", "new_chat", "New chat"),
        Binding("escape", "cancel", "Stop reply"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, client: OllamaClient, store: ChatStore, model: Optional[str] = None,
                 host_label: Optional[str] = None):
        super().__init__()
        self.client = client
        self.store = store
        self.model = model
        self.model_names: List[str] = []
        self.session = ChatSession(client)
        self.chat_id: Optional[int] = None
        self.busy = False
        self._cancel = threading.Event()
        self.context_total: Optional[int] = None  # context window of the loaded model, from /api/ps
        self.session_tokens = 0  # reply tokens received since the app started
        self._live: Optional[tuple] = None  # (chunks, seconds) while a reply streams in
        self.status_text = ""  # what the status bar shows (also read by the tests)
        self.is_local = is_local_host(client.base_url)
        self.resources_text = ""  # what the Resources tab shows (also read by the tests)
        self._polling = False
        self.sub_title = host_label or client.base_url

    # ---- layout -------------------------------------------------------------------------

    @property
    def main_screen(self):
        """The chat screen. App.query_one() only searches the screen on top, which is a dialog
        while one is open, so timers and background workers must look things up here."""
        return self.screen_stack[0]

    def compose(self) -> ComposeResult:
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
                yield Input(
                    placeholder="Message, or /help for commands",
                    id="prompt",
                    suggester=CommandSuggester(COMMAND_NAMES, lambda: self.model_names),
                )
        yield Footer()

    @staticmethod
    def _welcome_text() -> str:
        return ("Type a message and press Enter. Commands start with / (try /help).\n"
                "Ctrl+B hides the side panel, Esc stops a running reply, → accepts a suggestion.")

    def on_mount(self) -> None:
        if self.size.width < 100:
            self.main_screen.query_one("#sidebar").add_class("hidden")
        self.main_screen.query_one("#prompt").focus()
        self._refresh_chats()
        self._load_models(select_first=self.model is None)
        self._update_status()
        if psutil is not None:
            psutil.cpu_percent()  # the first call has no earlier sample to compare with
        self.set_interval(2.0, self._poll_resources)

    # ---- status and small helpers -------------------------------------------------------------

    def _update_status(self) -> None:
        state = "generating... (Esc to stop)" if self.busy else "ready"
        system = "set" if self.session.system else "none"
        stats = format_stats(self.session.last_stats, self.context_total,
                             self.session_tokens, self._live if self.busy else None)
        self.status_text = (f"model: {self.model or '(none)'} · temp {self.session.temperature:g}"
                            f" · system: {system} · {state}\n{stats}")
        self.main_screen.query_one("#status", Static).update(self.status_text)

    def _clear_stats(self) -> None:
        """Forget per-conversation numbers (new chat, loaded chat or another model)."""
        self.session.last_stats = None
        self.context_total = None
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
            self._clear_stats()
        self.model = name
        self.session.model = name
        if self.chat_id is not None:
            self.store.update_chat(self.chat_id, model=name)
        self._update_status()

    # ---- actions ------------------------------------------------------------------------------

    def action_toggle_sidebar(self) -> None:
        self.main_screen.query_one("#sidebar").toggle_class("hidden")

    async def action_new_chat(self) -> None:
        if not self._guard():
            return
        self.chat_id = None
        self.session = ChatSession(self.client, model=self.model, system=self.session.system,
                                   temperature=self.session.temperature)
        self._clear_stats()
        self._update_status()
        await self._render_messages([])
        self.main_screen.query_one("#prompt").focus()

    def action_cancel(self) -> None:
        if self.busy:
            self._cancel.set()

    def action_help(self) -> None:
        width = max(len(c) for c in COMMANDS)
        body = "\n".join(f"{cmd.ljust(width)}  {text}" for cmd, text in COMMANDS.items())
        body += ("\n\nCtrl+B  side panel    Ctrl+N  new chat    Esc  stop reply"
                 "\n→  accept the suggestion shown in the input    Ctrl+Q  quit")
        self.push_screen(InfoScreen("Commands", body))

    # ---- chat log -----------------------------------------------------------------------------

    async def _render_messages(self, messages: List[Dict[str, str]]) -> None:
        log = self.main_screen.query_one("#log", VerticalScroll)
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
        if not self._guard():
            return
        if not self.model:
            self.notify("No model selected. Pick one in the side panel or use /pull NAME.",
                        severity="warning")
            return
        log = self.main_screen.query_one("#log", VerticalScroll)
        for welcome in self.main_screen.query("#welcome"):
            await welcome.remove()
        await log.mount(UserMessage(text, classes="msg user"))
        reply = AssistantMessage("…", classes="msg assistant")
        await log.mount(reply)
        log.scroll_end(animate=False)
        self.session.model = self.model
        self._cancel.clear()
        self._set_busy(True)
        self._stream_reply(text, reply)

    @work(thread=True, exit_on_error=False)
    def _stream_reply(self, text: str, widget: AssistantMessage) -> None:
        parts: List[str] = []
        error: Optional[str] = None
        stopped = False
        try:
            stream = self.session.send(text, stream=True)
            started = time.monotonic()
            last_paint = 0.0
            for chunk in stream:
                if self._cancel.is_set():
                    stream.close()  # also drops the unanswered message from the session
                    stopped = True
                    break
                parts.append(chunk)
                now = time.monotonic()
                if now - last_paint >= 0.05:
                    last_paint = now
                    # each streamed chunk is about one token
                    self.call_from_thread(self._show_partial, widget, "".join(parts),
                                          (len(parts), now - started))
        except Exception as e:
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
        if model == self.model and total:
            self.context_total = total
            self._update_status()

    def _save_exchange(self, user_text: str, reply: str) -> None:
        if self.chat_id is None:
            self.chat_id = self.store.create_chat(
                user_text.strip().replace("\n", " ")[:TITLE_LENGTH] or "Chat",
                self.model, self.session.system, self.session.temperature,
            )
        self.store.add_messages(self.chat_id, [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ])
        self._refresh_chats()

    # ---- prompt input and slash commands -------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    async def _prompt_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if text.startswith("/"):
            await self._run_command(text)
        else:
            await self._submit_message(text)

    async def _run_command(self, line: str) -> None:
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
        elif name == "/rename":
            if self._has_arg(arg, "/rename TITLE"):
                self._rename_chat(arg)
        else:
            self.notify(f"Unknown command {name}. Try /help.", severity="warning")

    def _has_arg(self, arg: str, usage: str) -> bool:
        if not arg:
            self.notify(f"Usage: {usage}", severity="warning")
        return bool(arg)

    async def _cmd_model(self, arg: str) -> None:
        if not arg:
            self.notify(f"Current model: {self.model or '(none)'}")
            return
        if not self._guard():
            return
        if self.model_names and arg not in self.model_names:
            self.notify(f"{arg} is not installed. Use /pull {arg} to download it.", severity="warning")
            return
        self._set_model(arg)

    def _apply_system(self, text: str) -> None:
        if not self._guard():
            return
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
            if not 0 <= value <= 2:
                raise ValueError
        except ValueError:
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
        self._poll_resources()

    def _poll_resources(self) -> None:
        """Refresh the Resources tab, but only while it is on screen and the last read is done."""
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
        if select_first and names and not self.model:
            self._set_model(names[0])
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

    @on(Button.Pressed, "#pull-model")
    @on(Input.Submitted, "#pull-name")
    def _pull_pressed(self) -> None:
        field = self.main_screen.query_one("#pull-name", Input)
        if field.value.strip():
            self._pull(field.value.strip())
            field.value = ""

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
        self._pull_worker(name, self.main_screen.query_one("#progress", ProgressBar),
                          self.main_screen.query_one("#pull-status", Label))

    @work(thread=True, exit_on_error=False, group="model-ops")
    def _pull_worker(self, name: str, bar: ProgressBar, label: Label) -> None:
        self.call_from_thread(bar.add_class, "visible")
        last_paint = 0.0
        try:
            for p in self.client.pull_model(name):
                now = time.monotonic()
                if now - last_paint < 0.1 and p["completed"] != p["total"]:
                    continue
                last_paint = now
                self.call_from_thread(label.update, f"{name}: {p['status']}")
                self.call_from_thread(bar.update, total=p["total"] or None, progress=p["completed"] or 0)
        except Exception as e:
            self.call_from_thread(self._fail, e)
        else:
            self.call_from_thread(self.notify, f"Pulled {name}")
            self._load_models()
        finally:
            self.call_from_thread(bar.remove_class, "visible")
            self.call_from_thread(label.update, "")

    def _confirm_delete_model(self, name: str) -> None:
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
        if self.model == name:
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
        chat = self.store.get_chat(chat_id)
        if chat is None:
            self.notify("That chat no longer exists.", severity="warning")
            self._refresh_chats()
            return
        self.chat_id = chat_id
        self.model = chat["model"] or self.model
        self.session = ChatSession(self.client, model=self.model, system=chat["system"],
                                   temperature=chat["temperature"])
        self.session.messages.extend(chat["messages"])
        self._clear_stats()
        self.main_screen.query_one("#setting-system", Input).value = chat["system"] or ""
        self.main_screen.query_one("#setting-temp", Input).value = f"{chat['temperature']:g}"
        await self._render_messages(chat["messages"])
        self._update_status()
        self.main_screen.query_one("#prompt").focus()

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
            if chat_id == self.chat_id and self._guard():
                await self.action_new_chat()
            self._refresh_chats()

        self.push_screen(ConfirmScreen("Delete this chat?"), answered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-screen terminal chat for Ollama.")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
                        help=f"Ollama server URL (env OLLAMA_HOST, default {DEFAULT_HOST})")
    parser.add_argument("-m", "--model", default=os.environ.get("OLLAMA_MODEL"),
                        help="model name (env OLLAMA_MODEL, default: first installed model)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"seconds to wait for data from the server (default {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help=f"extra attempts after a transient error (default {DEFAULT_RETRIES})")
    parser.add_argument("--db", type=Path, default=default_db_path(),
                        help="chat database file (default: %(default)s)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaClient(args.host, timeout=args.timeout, retries=args.retries)
    store = ChatStore(args.db)
    try:
        OllamaTui(client, store, model=args.model).run()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
