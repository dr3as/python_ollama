"""Tests for ollama_tui.py.

Three kinds of test live here:
* plain unit tests of functions and ChatStore (TestChatStore, TestFormatStats, ...);
* app tests that start the real OllamaTui against a FakeOllama server and drive it like a user
  (TestTui, TestResourcesTab, ...), using Textual's "pilot";
* regression tests for bugs we fixed (look for "dialog" in the test names).
"""
import asyncio
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from textual import events
from textual.widgets import Button, Input, ListView, RadioButton, RadioSet, SelectionList, TabbedContent

from ollama_client import OllamaClient
from ollama_compare import ModelRun
import ollama_tui
from ollama_tui import (
    ChatStore, ChoiceScreen, CompareScreen, PromptArea, ConfirmScreen, InfoScreen, OllamaTui, format_bytes, format_resources, format_stats,
    context_percent, is_local_host, parse_nvidia_smi, read_gpus, read_host_stats,
)
from tests.fake_ollama import FakeOllama


class TestChatStore(unittest.TestCase):
    def setUp(self):
        self.store = ChatStore(":memory:")
        self.addCleanup(self.store.close)

    def test_create_and_get_chat_with_messages(self):
        chat_id = self.store.create_chat("Hello", "fake:1", "be brief", 0.3)
        self.store.add_messages(chat_id, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        chat = self.store.get_chat(chat_id)
        self.assertEqual((chat["title"], chat["model"], chat["system"], chat["temperature"]),
                         ("Hello", "fake:1", "be brief", 0.3))
        self.assertEqual([m["content"] for m in chat["messages"]], ["hi", "hello"])

    def test_get_missing_chat(self):
        self.assertIsNone(self.store.get_chat(99))

    def test_list_chats_most_recently_updated_first(self):
        first = self.store.create_chat("first", None, None, 0.7)
        second = self.store.create_chat("second", None, None, 0.7)
        self.assertEqual([c["id"] for c in self.store.list_chats()], [second, first])

    def test_update_chat(self):
        chat_id = self.store.create_chat("old", None, None, 0.7)
        self.store.update_chat(chat_id, title="new", temperature=1.0)
        chat = self.store.get_chat(chat_id)
        self.assertEqual((chat["title"], chat["temperature"]), ("new", 1.0))

    def test_update_rejects_unknown_fields(self):
        chat_id = self.store.create_chat("x", None, None, 0.7)
        with self.assertRaises(ValueError):
            self.store.update_chat(chat_id, id=5)

    def test_delete_chat_removes_messages(self):
        chat_id = self.store.create_chat("x", None, None, 0.7)
        self.store.add_messages(chat_id, [{"role": "user", "content": "hi"}])
        self.store.delete_chat(chat_id)
        self.assertIsNone(self.store.get_chat(chat_id))
        left = self.store._db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        self.assertEqual(left, 0)


class TestComparisonStore(unittest.TestCase):
    def setUp(self):
        self.store = ChatStore(":memory:")
        self.addCleanup(self.store.close)
        self.runs = [ModelRun(model="a:1", status="ok", answer="first", total_seconds=1.5),
                     ModelRun(model="b:1", status="error", error="boom")]

    def test_save_and_get_keep_runs_in_order(self):
        comparison_id = self.store.save_comparison("why?", "be brief", 0.2, "j:1", self.runs)
        data = self.store.get_comparison(comparison_id)
        self.assertEqual((data["prompt"], data["system"], data["temperature"], data["judge_model"]),
                         ("why?", "be brief", 0.2, "j:1"))
        self.assertIsNone(data["winner"])
        self.assertEqual([ModelRun.from_dict(r) for r in data["runs"]], self.runs)

    def test_get_missing_comparison(self):
        self.assertIsNone(self.store.get_comparison(99))

    def test_list_newest_first(self):
        first = self.store.save_comparison("one", None, 0.7, None, self.runs)
        second = self.store.save_comparison("two", None, 0.7, None, self.runs)
        self.assertEqual([c["id"] for c in self.store.list_comparisons()], [second, first])

    def test_winner_can_be_set_and_cleared(self):
        comparison_id = self.store.save_comparison("q", None, 0.7, None, self.runs)
        self.store.set_winner(comparison_id, "a:1")
        self.assertEqual(self.store.get_comparison(comparison_id)["winner"], "a:1")
        self.store.set_winner(comparison_id, None)
        self.assertIsNone(self.store.get_comparison(comparison_id)["winner"])

    def test_delete_removes_the_runs_too(self):
        comparison_id = self.store.save_comparison("q", None, 0.7, None, self.runs)
        self.store.delete_comparison(comparison_id)
        self.assertIsNone(self.store.get_comparison(comparison_id))
        left = self.store._db.execute("SELECT COUNT(*) FROM comparison_runs").fetchone()[0]
        self.assertEqual(left, 0)


class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    """Base for tests with `async def test_...` methods (they run inside an asyncio event loop)."""

    def _setupAsyncioRunner(self):
        # unittest starts its event loop in debug mode, which slows Textual down about 10x
        # (a test run goes from ~65 s to ~5 s). Override this hook to turn debug mode off.
        self._asyncioRunner = asyncio.Runner(debug=False)


# Sample data shared by the formatting tests below. LAST looks like ChatSession.last_stats.
LAST = {"prompt_tokens": 1200, "reply_tokens": 300, "tokens_per_second": 42.5,
        "load_seconds": 0.1, "total_seconds": 8.0}


class TestFormatStats(unittest.TestCase):
    def test_no_replies_yet(self):
        self.assertEqual(format_stats(None, None, 0), "no replies yet")

    def test_full_line(self):
        self.assertEqual(
            format_stats(LAST, 4096, 1500),
            "ctx ~1,500/4,096 (36%) · 42.5 tok/s · last 1,200 in / 300 out · session 1,500 out",
        )

    def test_without_context_total(self):
        self.assertTrue(format_stats(LAST, None, 300).startswith("ctx ~1,500 · "))

    def test_load_time_shown_only_when_noticeable(self):
        self.assertNotIn("load", format_stats(LAST, 4096, 0))
        self.assertIn("load 3.2s", format_stats({**LAST, "load_seconds": 3.2}, 4096, 0))

    def test_context_percent(self):
        self.assertEqual(context_percent(LAST, 4096), 36)
        self.assertIsNone(context_percent(LAST, None))
        self.assertIsNone(context_percent(None, 4096))

    def test_warning_mark_from_80_percent(self):
        self.assertNotIn("⚠", format_stats(LAST, 4096, 0))  # 36%
        near_full = {**LAST, "prompt_tokens": 1900, "reply_tokens": 100}  # 2000 of 2400 = 83%
        self.assertIn("(83%) ⚠", format_stats(near_full, 2400, 0))

    def test_live_numbers_while_streaming(self):
        self.assertEqual(format_stats(LAST, 4096, 0, live=(50, 2.0)),
                         "receiving: ~50 tokens · ~25.0 tok/s")

    def test_live_with_no_elapsed_time(self):
        self.assertEqual(format_stats(None, None, 0, live=(1, 0.0)), "receiving: ~1 tokens")


SMI_OUTPUT = "NVIDIA GeForce RTX 3070, 63, 6144, 8192, 58\n"
GPU = {"name": "NVIDIA GeForce RTX 3070", "util_percent": 63.0, "vram_used": 6144 * 2**20,
       "vram_total": 8192 * 2**20, "temperature": 58.0}
HOST = {"cpu_percent": 34.0, "ram_used": 12.1e9, "ram_total": 31.2e9}
MODEL = {"name": "llama3:8b", "size": 5_200_000_000, "size_vram": 5_200_000_000,
         "context_length": 4096, "expires_at": None}


class TestIsLocalHost(unittest.TestCase):
    def test_local_urls(self):
        for url in ("http://localhost:11434", "localhost", "http://127.0.0.1:11434",
                    "http://127.0.1.1", "http://[::1]:11434", "http://0.0.0.0:11434"):
            self.assertTrue(is_local_host(url), url)

    def test_remote_urls(self):
        for url in ("http://192.168.1.5:11434", "http://server:11434", "https://ollama.example.com"):
            self.assertFalse(is_local_host(url), url)

    def test_own_hostname_counts_as_local(self):
        with mock.patch("socket.gethostname", return_value="myserver"):
            self.assertTrue(is_local_host("http://myserver:11434"))


class TestHostReaders(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        self.assertEqual(parse_nvidia_smi(SMI_OUTPUT), [GPU])

    def test_parse_handles_na_values_and_bad_lines(self):
        gpus = parse_nvidia_smi("Tesla T4, [N/A], 100, 200, [N/A]\nbroken line\n")
        self.assertEqual(len(gpus), 1)
        self.assertIsNone(gpus[0]["util_percent"])
        self.assertIsNone(gpus[0]["temperature"])
        self.assertEqual(gpus[0]["vram_total"], 200 * 2**20)

    def test_parse_empty_output(self):
        self.assertEqual(parse_nvidia_smi(""), [])

    def test_read_gpus_without_nvidia_smi(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertIsNone(read_gpus())

    def test_read_gpus_parses_command_output(self):
        done = subprocess.CompletedProcess([], 0, stdout=SMI_OUTPUT)
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch("subprocess.run", return_value=done) as run:
            self.assertEqual(read_gpus(), [GPU])
        self.assertIn("--format=csv,noheader,nounits", run.call_args.args[0])

    def test_read_gpus_returns_none_when_command_fails(self):
        for error in (subprocess.CalledProcessError(9, "nvidia-smi"), FileNotFoundError(),
                      subprocess.TimeoutExpired("nvidia-smi", 3)):
            with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                    mock.patch("subprocess.run", side_effect=error):
                self.assertIsNone(read_gpus())

    def test_read_host_stats_without_psutil(self):
        with mock.patch.object(ollama_tui, "psutil", None):
            self.assertIsNone(read_host_stats())

    def test_read_host_stats_with_psutil(self):
        fake = SimpleNamespace(cpu_percent=lambda: 12.5,
                               virtual_memory=lambda: SimpleNamespace(used=4e9, total=16e9))
        with mock.patch.object(ollama_tui, "psutil", fake):
            self.assertEqual(read_host_stats(),
                             {"cpu_percent": 12.5, "ram_used": 4e9, "ram_total": 16e9})


class TestFormatResources(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(format_bytes(5.2e9), "5.2 GB")
        self.assertEqual(format_bytes(5e7), "50 MB")

    def test_model_fully_on_gpu(self):
        text = format_resources([MODEL], None, True, HOST, [GPU])
        self.assertIn("llama3:8b", text)
        self.assertIn("5.2 GB · 100% GPU", text)
        self.assertIn("ctx 4,096", text)
        self.assertNotIn("⚠", text)

    def test_model_spilling_onto_cpu_is_flagged(self):
        spilled = {**MODEL, "size": 10_000_000_000, "size_vram": 6_000_000_000}
        text = format_resources([spilled], None, False, None, None)
        self.assertIn("60% GPU / 40% CPU ⚠", text)

    def test_unload_countdown(self):
        soon = {**MODEL, "expires_at": datetime.now(timezone.utc) + timedelta(minutes=4, seconds=30)}
        self.assertIn("unloads in 4m", format_resources([soon], None, False, None, None))
        gone = {**MODEL, "expires_at": datetime.now(timezone.utc) - timedelta(seconds=5)}
        self.assertIn("unloading", format_resources([gone], None, False, None, None))

    def test_no_model_loaded_and_error(self):
        self.assertIn("none (a model loads", format_resources([], None, False, None, None))
        self.assertIn("Cannot reach Ollama", format_resources(None, "Cannot reach Ollama: x", False, None, None))

    def test_remote_host_shows_no_local_numbers(self):
        text = format_resources([MODEL], None, False, HOST, [GPU])
        self.assertIn("only", text)
        self.assertIn("localhost", text)
        self.assertNotIn("RAM  ", text)  # the meter line, not the word in the note
        self.assertNotIn("RTX", text)

    def test_local_numbers(self):
        text = format_resources([], None, True, HOST, [GPU])
        self.assertIn("CPU  ███░░░░░░░ 34%", text)
        self.assertIn("RAM  ████░░░░░░ 39%", text)
        self.assertIn("12.1 GB / 31.2 GB", text)
        self.assertIn("NVIDIA GeForce RTX 3070", text)
        self.assertIn("load ██████░░░░ 63%", text)
        self.assertIn("VRAM ████████░░ 75%", text)
        self.assertIn("6.4 GB / 8.6 GB", text)
        self.assertIn("temp 58 °C", text)

    def test_local_hints_when_tools_are_missing(self):
        text = format_resources([], None, True, None, None)
        self.assertIn("install psutil", text)
        self.assertIn("nvidia-smi not available", text)


class TuiTestCase(AsyncTestCase):
    """Base for app tests: a fake server, an in-memory database and an OllamaTui, not yet running.

    A test starts the app with `async with self.app.run_test(size=(120, 40)) as pilot:`. The
    "pilot" presses keys and clicks like a user, and the app runs headless (no real terminal).
    """

    async def asyncSetUp(self):
        self.server = FakeOllama().start()
        self.addCleanup(self.server.stop)
        self.store = ChatStore(":memory:")
        self.addCleanup(self.store.close)
        self.app = OllamaTui(OllamaClient(self.server.url, retry_delay=0.01), self.store)

    async def settle(self, pilot):
        """Wait until background work is done and the UI has processed its results."""
        # Workers run in threads and report back through the event loop, so a test has to wait
        # for both before it looks at the screen, otherwise it would check too early.
        await self.app.workers.wait_for_complete()
        await pilot.pause()

    async def say(self, pilot, text):
        """Type a message or /command into the prompt, press Enter and wait for the result."""
        self.app.query_one("#prompt", PromptArea).text = text
        await pilot.press("enter")
        await self.settle(pilot)

    def log_texts(self):
        """The text of every message currently shown in the chat log, in order."""
        # main_screen, not app.query: a dialog may be on top (see the bug fixed in c476a79).
        return [w.text
                for w in self.app.main_screen.query("#log .msg")]


class TestTui(TuiTestCase):
    async def test_first_installed_model_is_selected_and_listed(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            self.assertEqual(self.app.model, "fake:1")
            self.assertEqual(len(self.app.query_one("#models", ListView).children), 2)

    async def test_message_gets_reply_and_is_saved(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hello")
            self.assertEqual(self.log_texts(), ["hello", "msgs=1 last=hello"])
            chat = self.store.get_chat(self.app.chat_id)
            self.assertEqual(chat["title"], "hello")
            self.assertEqual([m["role"] for m in chat["messages"]], ["user", "assistant"])
            self.assertEqual(len(self.app.query_one("#chats", ListView).children), 1)

    async def test_status_bar_shows_stats_after_reply(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            self.assertIn("no replies yet", self.app.status_text)
            await self.say(pilot, "hello")  # reply "msgs=1 last=hello": 2 tokens in 0.5 s
            await self.settle(pilot)
            self.assertIn("ctx ~12/4,096 (0%)", self.app.status_text)
            self.assertIn("4.0 tok/s", self.app.status_text)
            self.assertIn("session 2 out", self.app.status_text)
            await self.say(pilot, "again")
            self.assertIn("session 4 out", self.app.status_text)

    async def test_stats_reset_for_new_chat_and_model_switch(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hello")
            await self.settle(pilot)
            await self.say(pilot, "/new")
            self.assertIn("no replies yet", self.app.status_text)
            await self.say(pilot, "hello")
            await self.settle(pilot)
            await self.say(pilot, "/model nomic-embed-text:latest")
            self.assertIn("no replies yet", self.app.status_text)

    async def test_reply_finishes_while_a_dialog_is_open(self):
        self.server.slow_seconds = 1.0
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            self.app.query_one("#prompt", PromptArea).text = "slow"
            await pilot.press("enter")
            await pilot.press("f1")  # open the help dialog while the reply is still coming
            self.assertIsInstance(self.app.screen, InfoScreen)
            await pilot.pause(1.5)
            await self.settle(pilot)
            await pilot.press("escape")
            await pilot.pause()
            self.assertFalse(self.app.busy)
            self.assertIn("ready", self.app.status_text)
            self.assertEqual(self.log_texts()[-1], "msgs=1 last=slow")

    async def test_prompt_tab_completes_command_and_model_name(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            prompt.text = "/mo"
            await pilot.pause()
            await pilot.press("tab")
            self.assertEqual(prompt.text, "/model")
            prompt.text = "/model nomic"
            await pilot.pause()
            await pilot.press("tab")
            self.assertEqual(prompt.text, "/model nomic-embed-text:latest")
            self.assertIs(self.app.focused, prompt)

    async def type_text(self, pilot, text):
        """Type like a user: printable characters as key presses, "\\n" as Ctrl+J."""
        for char in text:
            await pilot.press("ctrl+j" if char == "\n" else char)

    async def test_ctrl_j_adds_a_line_and_enter_sends_all_of_it(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            await self.type_text(pilot, "first\nsecond")
            self.assertEqual(prompt.text, "first\nsecond")
            self.assertEqual(self.log_texts(), [])  # nothing sent yet
            await pilot.press("enter")
            await self.settle(pilot)
            self.assertEqual(self.log_texts(), ["first\nsecond", "msgs=1 last=first\nsecond"])
            self.assertEqual(prompt.text, "")
            self.assertIs(self.app.focused, prompt)
            self.assertEqual(self.store.get_chat(self.app.chat_id)["title"], "first second")

    async def test_enter_on_empty_or_blank_prompt_sends_nothing(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            before = len(self.server.requests)
            await pilot.press("enter")
            await self.type_text(pilot, "  \n ")
            await pilot.press("enter")
            await self.settle(pilot)
            self.assertEqual(len(self.server.requests), before)
            self.assertEqual(self.log_texts(), [])

    async def test_pasted_multiline_text_is_kept(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            self.app.post_message(events.Paste("def f():\n    return 1\n"))  # as the terminal does
            await pilot.pause()
            self.assertEqual(prompt.text, "def f():\n    return 1\n")

    async def test_multiline_text_starting_with_slash_is_not_a_command(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.type_text(pilot, "/delete fake:1\nplease explain")
            await pilot.press("enter")
            await self.settle(pilot)
            self.assertNotIsInstance(self.app.screen, ConfirmScreen)
            self.assertEqual(self.server.requests[-1][1]["messages"][-1]["content"],
                             "/delete fake:1\nplease explain")

    async def test_prompt_grows_with_text_up_to_a_limit_and_shrinks_after_sending(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            self.assertEqual(prompt.outer_size.height, 3)
            await self.type_text(pilot, "a\nb\nc")
            await pilot.pause()
            self.assertEqual(prompt.outer_size.height, 5)  # 3 lines + border
            await self.type_text(pilot, "\n" * 20)
            await pilot.pause()
            self.assertEqual(prompt.outer_size.height, PromptArea.MAX_ROWS + 2)
            await pilot.press("enter")
            await self.settle(pilot)
            self.assertEqual(prompt.outer_size.height, 3)

    async def test_hint_shows_pending_completion(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            self.assertIn("Ctrl+J", prompt.hint_text)
            await self.type_text(pilot, "/mo")
            self.assertEqual(prompt.hint_text, "Tab → /model")
            await pilot.press("tab")
            self.assertEqual(prompt.text, "/model")
            self.assertIn("Ctrl+J", prompt.hint_text)  # fully typed: nothing left to suggest

    async def test_tab_does_not_complete_multiline_text(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            await self.type_text(pilot, "/mo\nx")
            await pilot.press("tab")
            self.assertEqual(prompt.text, "/mo\nx")
            self.assertIsNot(self.app.focused, prompt)

    async def test_tab_moves_focus_when_there_is_nothing_to_complete(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            prompt = self.app.query_one("#prompt", PromptArea)
            prompt.text = "hello"
            await pilot.pause()
            await pilot.press("tab")
            self.assertEqual(prompt.text, "hello")
            self.assertIsNot(self.app.focused, prompt)

    async def test_warns_once_per_level_as_context_fills(self):
        self.server.context_length = 15  # replies use 12, then 32 tokens
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            with mock.patch.object(OllamaTui, "notify") as notify:
                await self.say(pilot, "one")
                await self.settle(pilot)
                self.assertEqual([c.kwargs["severity"] for c in notify.call_args_list], ["warning"])  # 80%
                self.assertIn("⚠", self.app.status_text)
                await self.say(pilot, "two")
                await self.settle(pilot)
                self.assertEqual([c.kwargs["severity"] for c in notify.call_args_list],
                                 ["warning", "error"])  # full
                await self.say(pilot, "three")
                await self.settle(pilot)
                self.assertEqual(notify.call_count, 2)  # no repeats
                await self.say(pilot, "/new")
                await self.say(pilot, "again")
                await self.settle(pilot)
                self.assertEqual(notify.call_count, 3)  # a new chat warns again

    async def test_no_warning_when_context_is_roomy(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            with mock.patch.object(OllamaTui, "notify") as notify:
                await self.say(pilot, "one")
                await self.settle(pilot)
                notify.assert_not_called()

    async def open_chat_with_model(self, pilot, model):
        chat_id = self.store.create_chat("old chat", model, None, 0.7)
        self.store.add_messages(chat_id, [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "hello"}])
        await self.app._load_chat(chat_id)
        await pilot.pause()
        return chat_id

    async def test_missing_model_on_load_offers_pull_switch_or_keep(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.open_chat_with_model(pilot, "gone:1")
            self.assertIsInstance(self.app.screen, ChoiceScreen)
            self.assertEqual([v for _, v in self.app.screen.choices], ["pull", "switch", "keep"])
            self.assertEqual(self.log_texts(), ["hi", "hello"])  # history is shown regardless

    async def test_missing_model_pull_choice(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.open_chat_with_model(pilot, "gone:1")
            self.app.screen.query_one("#choice-0").press()
            await pilot.pause()
            await self.settle(pilot)
            pulls = [body["model"] for path, body in self.server.requests if path == "/api/pull"]
            self.assertEqual(pulls, ["gone:1"])
            self.assertIn("gone:1", self.app.model_names)
            self.assertEqual(self.app.model, "gone:1")

    async def test_missing_model_switch_choice_keeps_chat_and_saves_model(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            chat_id = await self.open_chat_with_model(pilot, "gone:1")
            self.app.screen.query_one("#choice-1").press()
            await pilot.pause()
            self.assertEqual(self.app.model, "fake:1")
            self.assertEqual(self.store.get_chat(chat_id)["model"], "fake:1")
            await self.say(pilot, "next")
            self.assertEqual(self.server.requests[-1][1]["model"], "fake:1")

    async def test_missing_model_keep_choice_changes_nothing(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.open_chat_with_model(pilot, "gone:1")
            await pilot.press("escape")
            await pilot.pause()
            self.assertEqual(self.app.model, "gone:1")
            self.assertFalse([r for r in self.server.requests if r[0] == "/api/pull"])

    async def test_installed_model_on_load_shows_no_dialog(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.open_chat_with_model(pilot, "nomic-embed-text:latest")
            self.assertNotIsInstance(self.app.screen, ChoiceScreen)

    async def test_history_is_sent_with_follow_up(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "one")
            await self.say(pilot, "two")
            self.assertEqual(self.log_texts()[-1], "msgs=3 last=two")

    async def test_server_error_is_shown_and_not_saved(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "boom")
            self.assertIn("model exploded", self.log_texts()[-1])
            self.assertIsNone(self.app.chat_id)
            await self.say(pilot, "ok")
            self.assertEqual(self.log_texts()[-1], "msgs=1 last=ok")

    async def test_escape_stops_reply_and_keeps_history_clean(self):
        self.server.slow_seconds = 1.5
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            self.app.query_one("#prompt", PromptArea).text = "slow"
            await pilot.press("enter")
            await pilot.pause()
            self.assertTrue(self.app.busy)
            await pilot.press("escape")
            await self.settle(pilot)
            self.assertFalse(self.app.busy)
            self.assertIn("stopped", self.log_texts()[-1])
            self.assertIsNone(self.app.chat_id)
            self.assertEqual(self.app.session.messages, [])

    async def test_new_chat_and_load_saved_chat(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "first chat")
            saved_id = self.app.chat_id
            await self.say(pilot, "/new")
            self.assertIsNone(self.app.chat_id)
            self.assertEqual(self.log_texts(), [])
            await self.app._load_chat(saved_id)
            await pilot.pause()
            self.assertEqual(self.log_texts(), ["first chat", "msgs=1 last=first chat"])
            await self.say(pilot, "again")
            self.assertEqual(self.log_texts()[-1], "msgs=3 last=again")
            self.assertEqual(len(self.store.get_chat(saved_id)["messages"]), 4)

    async def test_system_prompt_command_applies_and_persists(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/system be brief")
            await self.say(pilot, "hi")
            self.assertEqual(self.log_texts()[-1], "msgs=2 last=hi")
            self.assertEqual(self.store.get_chat(self.app.chat_id)["system"], "be brief")
            await self.say(pilot, "/system")
            await self.say(pilot, "again")
            self.assertEqual(self.log_texts()[-1], "msgs=3 last=again")  # system message removed

    async def test_temperature_validation_and_use(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/temp 5")
            self.assertEqual(self.app.session.temperature, 0.7)
            await self.say(pilot, "/temp 0.2")
            await self.say(pilot, "hi")
            self.assertEqual(self.server.requests[-1][1]["options"]["temperature"], 0.2)

    async def test_model_command(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/model nomic-embed-text:latest")
            self.assertEqual(self.app.model, "nomic-embed-text:latest")
            await self.say(pilot, "/model nope")
            self.assertEqual(self.app.model, "nomic-embed-text:latest")
            await self.say(pilot, "hi")
            self.assertEqual(self.server.requests[-1][1]["model"], "nomic-embed-text:latest")

    async def test_rename_chat(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hi")
            await self.say(pilot, "/rename My title")
            self.assertEqual(self.store.get_chat(self.app.chat_id)["title"], "My title")

    async def test_unknown_command_is_not_sent_to_model(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            before = len(self.server.requests)
            await self.say(pilot, "/bogus")
            self.assertEqual(len(self.server.requests), before)

    async def test_help_opens_and_closes(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/help")
            self.assertIsInstance(self.app.screen, InfoScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(self.app.screen, InfoScreen)

    async def test_show_opens_details(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/show")
            await pilot.pause()
            self.assertIsInstance(self.app.screen, InfoScreen)
            self.assertIn("parameter_size: 7B", self.app.screen.body)

    async def test_pull_adds_model_to_list(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/pull new:1")
            await self.settle(pilot)
            self.assertIn("new:1", self.app.model_names)
            self.assertEqual(len(self.app.query_one("#models", ListView).children), 3)

    async def test_delete_model_asks_for_confirmation(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/delete nomic-embed-text:latest")
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("n")
            await self.settle(pilot)
            self.assertIn("nomic-embed-text:latest", self.server.models)
            await self.say(pilot, "/delete nomic-embed-text:latest")
            await pilot.press("y")
            await self.settle(pilot)
            self.assertNotIn("nomic-embed-text:latest", self.server.models)
            self.assertEqual(self.app.model_names, ["fake:1"])

    async def test_delete_chat_button(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hi")
            chats = self.app.query_one("#chats", ListView)
            chats.index = 0
            await pilot.pause()
            self.app.query_one("#delete-chat").press()
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            self.assertEqual(self.store.list_chats(), [])
            self.assertIsNone(self.app.chat_id)

    async def test_sidebar_toggle(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            sidebar = self.app.query_one("#sidebar")
            self.assertFalse(sidebar.has_class("hidden"))
            await pilot.press("ctrl+b")
            self.assertTrue(sidebar.has_class("hidden"))

    async def test_sidebar_starts_hidden_on_narrow_terminal(self):
        async with self.app.run_test(size=(80, 30)) as pilot:
            self.assertTrue(self.app.query_one("#sidebar").has_class("hidden"))


class TestResourcesTab(TuiTestCase):
    FAKE_PSUTIL = SimpleNamespace(cpu_percent=lambda: 34.0,
                                  virtual_memory=lambda: SimpleNamespace(used=12.1e9, total=31.2e9))

    async def show_tab(self, pilot):
        self.app.query_one(TabbedContent).active = "tab-resources"
        await pilot.pause()
        await self.settle(pilot)

    async def test_local_host_shows_ollama_and_machine_numbers(self):
        with mock.patch.object(ollama_tui, "psutil", self.FAKE_PSUTIL), \
                mock.patch.object(ollama_tui, "read_gpus", return_value=[GPU]):
            async with self.app.run_test(size=(120, 40)) as pilot:
                await self.settle(pilot)
                self.assertEqual(self.app.resources_text, "")  # nothing polled while the tab is hidden
                await self.show_tab(pilot)
                text = self.app.resources_text
        self.assertIn("fake:1", text)
        self.assertIn("75% GPU / 25% CPU", text)
        self.assertIn("ctx 4,096", text)
        self.assertIn("CPU  ███░░░░░░░ 34%", text)
        self.assertIn("RTX 3070", text)

    async def test_remote_host_hides_machine_numbers(self):
        self.app.is_local = False
        with mock.patch.object(ollama_tui, "read_gpus", side_effect=AssertionError("must not run")):
            async with self.app.run_test(size=(120, 40)) as pilot:
                await self.settle(pilot)
                await self.show_tab(pilot)
                text = self.app.resources_text
        self.assertIn("fake:1", text)
        self.assertIn("only shown when Ollama runs on this", text.replace("\n  ", " "))
        self.assertNotIn("RAM  ", text)
        self.assertIn("unloads in 4m", text)

    async def test_refreshes_while_visible(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.show_tab(pilot)
            self.assertIn("fake:1", self.app.resources_text)
            self.server.running.clear()
            await pilot.pause(2.5)
            await self.settle(pilot)
            self.assertIn("none (a model loads", self.app.resources_text)

    async def test_no_polling_when_sidebar_hidden(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.show_tab(pilot)
            await pilot.press("ctrl+b")
            self.server.running.clear()
            await pilot.pause(2.5)
            await self.settle(pilot)
            self.assertIn("fake:1", self.app.resources_text)  # stale: hidden panel is not refreshed

    async def test_polling_keeps_working_while_a_dialog_is_open(self):
        # App.query_one searches the screen on top, so a dialog used to hide the whole UI from the timer
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.show_tab(pilot)
            await self.say(pilot, "/delete fake:1")
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            self.server.running.clear()
            await pilot.pause(2.5)  # the 2 s poll timer fires while the dialog is open
            await self.settle(pilot)
            self.assertIn("none (a model loads", self.app.resources_text)
            await pilot.press("n")

    async def test_server_down_shows_error_in_tab(self):
        from tests.test_ollama_client import unused_url
        from ollama_client import OllamaClient
        app = OllamaTui(OllamaClient(unused_url(), retries=0), self.store)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.workers.wait_for_complete()
            app.query_one(TabbedContent).active = "tab-resources"
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIn("Cannot reach Ollama", app.resources_text)


def table_rows(screen):
    """The Compare table as {row label: [cell, ...]}."""
    table = screen.query_one("#cmp-table")
    return {row[0]: list(row[1:]) for row in (table.get_row_at(i) for i in range(table.row_count))}


class TestCompareView(TuiTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.server.models[:] = ["a:1", "b:1", "c:1"]
        self.server.running[:] = ["a:1"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    async def open_compare(self, pilot):
        await self.settle(pilot)
        await pilot.press("f2")
        await pilot.pause()
        self.assertIsInstance(self.app.screen, CompareScreen)
        return self.app.screen

    async def wait_for_run(self, pilot, screen, timeout=20.0):
        for _ in range(int(timeout / 0.05)):
            await pilot.pause(0.05)
            if not screen.running:
                break
        await self.settle(pilot)

    async def run_compare(self, pilot, screen, models=("a:1", "b:1"), prompt="hello", judge=None):
        selection = screen.query_one("#cmp-models", SelectionList)
        for name in models:
            selection.select(name)
        screen.query_one("#cmp-prompt").text = prompt
        if judge:
            screen.query_one("#cmp-judge").query(RadioButton)[screen.models.index(judge) + 1].value = True
        await pilot.press("ctrl+r")
        await self.wait_for_run(pilot, screen)

    async def test_f2_opens_and_closes_the_view(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await pilot.press("f2")
            await pilot.pause()
            self.assertNotIsInstance(self.app.screen, CompareScreen)

    async def test_compare_command_opens_the_view(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/compare")
            await pilot.pause()
            self.assertIsInstance(self.app.screen, CompareScreen)

    async def test_needs_two_installed_models(self):
        self.server.models[:] = ["only:1"]
        async with self.app.run_test(size=(140, 45)) as pilot:
            await self.settle(pilot)
            with mock.patch.object(OllamaTui, "notify") as notify:
                await pilot.press("f2")
                await pilot.pause()
            self.assertNotIsInstance(self.app.screen, CompareScreen)
            self.assertIn("at least two", notify.call_args.args[0])

    async def test_cannot_open_while_a_chat_reply_is_streaming(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            await self.settle(pilot)
            self.app.busy = True
            await pilot.press("f2")
            await pilot.pause()
            self.assertNotIsInstance(self.app.screen, CompareScreen)

    async def test_run_shows_answers_and_numbers_side_by_side(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen)
            self.assertEqual(list(screen.columns), ["a:1", "b:1"])
            for column in screen.columns.values():
                self.assertEqual(column.answer_text, "msgs=1 last=hello")
                self.assertTrue(column.status_text.startswith("ok"))
            rows = table_rows(screen)
            self.assertEqual(rows["Status"], ["ok", "ok"])
            self.assertEqual(rows["Reply tokens"], ["2", "2"])
            self.assertEqual(rows["Model memory"], ["4.0 GB · 75% GPU ⚠"] * 2)
            self.assertIn("Done", screen.banner_text)
            self.assertFalse(screen.running)

    async def test_models_never_share_memory_and_nothing_is_left_loaded(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen, models=("a:1", "b:1", "c:1"))
            self.assertEqual(self.server.max_loaded, 1)
            self.assertEqual(self.server.running, [])

    async def test_result_is_saved_and_listed_in_history(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen, prompt="why is the sky blue")
            saved = self.store.get_comparison(screen.comparison_id)
            self.assertEqual(saved["prompt"], "why is the sky blue")
            self.assertEqual([r["model"] for r in saved["runs"]], ["a:1", "b:1"])
            self.assertEqual(len(screen.query_one("#cmp-history", ListView).children), 1)

    async def test_uses_the_chats_system_prompt_and_temperature(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "/system be brief")
            await self.say(pilot, "/temp 0.3")
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen)
            body = [b for p, b in self.server.requests if p == "/api/chat"][-1]
            self.assertEqual(body["messages"][0], {"role": "system", "content": "be brief"})
            self.assertEqual(body["options"]["temperature"], 0.3)

    async def test_run_needs_two_models_and_a_prompt(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            before = len(self.server.requests)
            with mock.patch.object(CompareScreen, "notify") as notify:
                screen.query_one("#cmp-prompt").text = "hello"
                await pilot.press("ctrl+r")
                self.assertIn("at least two", notify.call_args.args[0])
                screen.query_one("#cmp-models", SelectionList).select_all()
                screen.query_one("#cmp-prompt").text = "  "
                await pilot.press("ctrl+r")
                self.assertIn("prompt", notify.call_args.args[0])
            self.assertEqual(len(self.server.requests), before)

    async def test_one_failing_model_still_shows_the_others(self):
        self.server.bad_models.add("b:1")
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen, models=("a:1", "b:1", "c:1"))
            self.assertEqual(table_rows(screen)["Status"][0], "ok")
            self.assertTrue(table_rows(screen)["Status"][1].startswith("error:"))
            self.assertEqual(table_rows(screen)["Status"][2], "ok")
            self.assertIn("model failed to load", screen.columns["b:1"].status_text)

    async def test_cpu_spill_is_flagged_in_the_table(self):
        self.server.sizes["b:1"] = (10_000_000_000, 6_000_000_000)
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen)
            self.assertEqual(table_rows(screen)["Model memory"][1], "10.0 GB · 60% GPU ⚠")

    async def test_esc_stops_the_run_and_skips_the_rest(self):
        self.server.delays["a:1"] = 0.6
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            selection = screen.query_one("#cmp-models", SelectionList)
            selection.select("a:1")
            selection.select("b:1")
            screen.query_one("#cmp-prompt").text = "hello"
            await pilot.press("ctrl+r")
            await pilot.pause(0.2)
            self.assertTrue(screen.running)
            await pilot.press("escape")
            await self.wait_for_run(pilot, screen)
            self.assertEqual(table_rows(screen)["Status"], ["stopped", "skipped"])
            self.assertIn("Stopped", screen.banner_text)
            self.assertEqual(self.server.running, [])

    async def test_cannot_leave_while_running(self):
        self.server.delays["a:1"] = 0.6
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            selection = screen.query_one("#cmp-models", SelectionList)
            selection.select("a:1")
            selection.select("b:1")
            screen.query_one("#cmp-prompt").text = "hello"
            await pilot.press("ctrl+r")
            await pilot.pause(0.2)
            await pilot.press("f2")
            await pilot.pause()
            self.assertIs(self.app.screen, screen)
            await pilot.press("escape")
            await self.wait_for_run(pilot, screen)

    async def test_pick_a_winner_and_change_your_mind(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen)
            screen.columns["b:1"].query_one(".cmp-pick", Button).press()
            await pilot.pause()
            self.assertTrue(screen.columns["b:1"].has_class("winner"))
            self.assertFalse(screen.columns["a:1"].has_class("winner"))
            self.assertEqual(self.store.get_comparison(screen.comparison_id)["winner"], "b:1")
            screen.columns["b:1"].query_one(".cmp-pick", Button).press()
            await pilot.pause()
            self.assertFalse(screen.columns["b:1"].has_class("winner"))
            self.assertIsNone(self.store.get_comparison(screen.comparison_id)["winner"])

    async def test_judge_scores_appear_in_columns_and_table(self):
        self.server.judge_reply = ('{"A": {"score": 9, "reason": "clear"},'
                                   ' "B": {"score": 2, "reason": "vague"}}')
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen, judge="c:1")
            scores = sorted(run.judge_score for run in screen.runs)
            self.assertEqual(scores, [2.0, 9.0])
            self.assertIn("Judge score", table_rows(screen))
            self.assertTrue(any("judge:" in c.status_text for c in screen.columns.values()))
            self.assertEqual(self.store.get_comparison(screen.comparison_id)["judge_model"], "c:1")
            self.assertEqual(self.server.max_loaded, 1)

    async def test_unreadable_judge_reply_is_reported(self):
        self.server.judge_reply = "no idea"
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            with mock.patch.object(CompareScreen, "notify") as notify:
                await self.run_compare(pilot, screen, judge="c:1")
            self.assertIn("Judging did not work", notify.call_args.args[0])
            self.assertNotIn("Judge score", table_rows(screen))

    async def test_gpu_numbers_are_added_on_this_machine(self):
        calls = []

        def reader():
            calls.append(1)
            return [{"name": "GPU", "util_percent": 70.0, "vram_used": 1_000_000_000 * len(calls),
                     "vram_total": 8_000_000_000, "temperature": 50.0}]

        with mock.patch.object(ollama_tui, "read_gpus", reader):
            async with self.app.run_test(size=(140, 45)) as pilot:
                screen = await self.open_compare(pilot)
                await self.run_compare(pilot, screen)
                rows = table_rows(screen)
        self.assertIn("GPU memory used", rows)
        self.assertEqual(rows["GPU peak load"], ["70%", "70%"])

    async def test_no_gpu_numbers_when_ollama_is_remote(self):
        self.app.is_local = False
        with mock.patch.object(ollama_tui, "read_gpus", side_effect=AssertionError("must not run")):
            async with self.app.run_test(size=(140, 45)) as pilot:
                screen = await self.open_compare(pilot)
                await self.run_compare(pilot, screen)
                self.assertNotIn("GPU memory used", table_rows(screen))

    async def select_history_item(self, pilot, screen):
        screen.query_one(TabbedContent).active = "cmp-tab-history"
        await pilot.pause()
        history = screen.query_one("#cmp-history", ListView)
        history.focus()
        history.index = 0
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    async def test_old_comparison_can_be_reopened_from_history(self):
        runs = [ModelRun(model="x:1", status="ok", answer="old answer", total_seconds=2.0,
                         tokens_per_second=10.0),
                ModelRun(model="y:1", status="ok", answer="other answer", total_seconds=3.0,
                         tokens_per_second=8.0)]
        comparison_id = self.store.save_comparison("old question", None, 0.7, None, runs)
        self.store.set_winner(comparison_id, "y:1")
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.select_history_item(pilot, screen)
            self.assertEqual(screen.query_one("#cmp-prompt").text, "old question")
            self.assertEqual(screen.columns["x:1"].answer_text, "old answer")
            self.assertTrue(screen.columns["y:1"].has_class("winner"))
            self.assertEqual(table_rows(screen)["Total time"], ["2.0 s ★", "3.0 s"])

    async def test_export_writes_markdown(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            screen.export_dir = Path(self.tmp.name)
            await self.run_compare(pilot, screen, prompt="export me")
            screen.query_one("#cmp-export", Button).press()
            await pilot.pause()
            text = (Path(self.tmp.name) / f"compare-{screen.comparison_id}.md").read_text()
            self.assertIn("> export me", text)
            self.assertIn("### a:1", text)
            self.assertIn("| Total time |", text)

    async def test_export_without_a_comparison_explains_itself(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            screen.export_dir = Path(self.tmp.name)
            with mock.patch.object(CompareScreen, "notify") as notify:
                screen.query_one("#cmp-export", Button).press()
                await pilot.pause()
            self.assertIn("Run a comparison", notify.call_args.args[0])
            self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    async def test_delete_asks_first_then_removes_the_comparison(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen)
            screen.query_one(TabbedContent).active = "cmp-tab-history"
            await pilot.pause()
            history = screen.query_one("#cmp-history", ListView)
            history.index = 0
            await pilot.pause()
            screen.query_one("#cmp-delete", Button).press()
            await pilot.pause()
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            self.assertEqual(len(self.store.list_comparisons()), 1)
            screen.query_one("#cmp-delete", Button).press()
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            self.assertEqual(self.store.list_comparisons(), [])
            self.assertEqual(screen.runs, [])

    async def test_chat_still_works_after_a_comparison_unloaded_everything(self):
        async with self.app.run_test(size=(140, 45)) as pilot:
            screen = await self.open_compare(pilot)
            await self.run_compare(pilot, screen)
            await pilot.press("f2")
            await pilot.pause()
            self.assertNotIsInstance(self.app.screen, CompareScreen)
            await self.say(pilot, "still there?")
            self.assertEqual(self.log_texts()[-1], "msgs=1 last=still there?")


class TestConfigDefaults(unittest.TestCase):
    """The config file supplies defaults for the TUI's command line and first chat."""

    def parse(self, config, *argv):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": "", "OLLAMA_MODEL": ""}):
            return ollama_tui.build_parser(config).parse_args(list(argv))

    def test_config_sets_defaults(self):
        args = self.parse({"host": "http://srv:1", "retries": 4, "db": "~/x.db"})
        self.assertEqual((args.host, args.retries), ("http://srv:1", 4))
        self.assertEqual(args.db, Path("~/x.db").expanduser())

    def test_command_line_beats_config(self):
        args = self.parse({"host": "http://srv:1"}, "--host", "http://cli:1")
        self.assertEqual(args.host, "http://cli:1")

    def test_default_db_without_config(self):
        self.assertEqual(self.parse({}).db, ollama_tui.default_db_path())

    def test_app_starts_with_configured_system_and_temperature(self):
        store = ChatStore(":memory:")
        self.addCleanup(store.close)
        app = OllamaTui(OllamaClient("http://127.0.0.1:1"), store, system="be brief",
                        temperature=0.2)
        self.assertEqual((app.session.system, app.session.temperature), ("be brief", 0.2))

    def test_main_reports_bad_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("bogus = 1\n")
            with mock.patch("sys.stderr") as err:
                rc = ollama_tui.main(["--config", str(path)])
        self.assertEqual(rc, 2)
        self.assertIn("Config error", "".join(c.args[0] for c in err.write.call_args_list))


class TestAttachments(TuiTestCase):
    """/file attaches a text file to the next message; /save writes a reply to a file."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "a.log").write_text("failed login root\nfailed login admin\n")

    def sent(self):
        """The content of the last user message the fake server received."""
        return [b for p, b in self.server.requests if p == "/api/chat"][-1]["messages"][-1]["content"]

    async def test_file_goes_with_the_next_message_only(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            self.assertEqual([a[0] for a in self.app.attachments], ["a.log"])
            self.assertIn("📎 1", self.app.status_text)
            await self.say(pilot, "Summarise")
            self.assertIn("----- file: a.log -----\nfailed login root", self.sent())
            self.assertEqual(self.app.attachments, [])
            self.assertNotIn("📎", self.app.status_text)
            await self.say(pilot, "and now?")
            self.assertEqual(self.sent(), "and now?")

    async def test_chat_shows_a_summary_not_the_file(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            await self.say(pilot, "Summarise")
            shown = str(self.app.main_screen.query(".msg.user").first().render())
            self.assertEqual(shown, "Summarise\n📎 a.log (2 lines)")
            self.assertEqual(self.store.get_chat(self.app.chat_id)["title"],
                             "Summarise 📎 a.log (2 lines)")

    async def test_reloaded_chat_shows_summary_and_keeps_file_in_history(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            await self.say(pilot, "Summarise")
            chat_id = self.app.chat_id
            await self.say(pilot, "/new")
            await self.app._load_chat(chat_id)
            await self.settle(pilot)
            shown = str(self.app.main_screen.query(".msg.user").first().render())
            self.assertIn("📎 a.log", shown)
            self.assertNotIn("failed login", shown)
            self.assertIn("failed login root", self.app.session.messages[0]["content"])

    async def test_attaching_same_file_twice_keeps_one_copy(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            (self.dir / "a.log").write_text("changed")
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            self.assertEqual(self.app.attachments, [("a.log", "changed")])

    async def test_file_clear_and_bad_file(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            await self.say(pilot, "/file clear")
            self.assertEqual(self.app.attachments, [])
            await self.say(pilot, f"/file {self.dir / 'missing'}")
            self.assertEqual(self.app.attachments, [])

    async def test_new_chat_drops_attachments(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/file {self.dir / 'a.log'}")
            await self.say(pilot, "/new")
            self.assertEqual(self.app.attachments, [])

    async def test_save_writes_last_reply(self):
        target = self.dir / "reply.md"
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hello")
            await self.say(pilot, f"/save {target}")
            self.assertEqual(target.read_text(), "msgs=1 last=hello\n")

    async def test_save_code_writes_only_code_blocks(self):
        target = self.dir / "code.py"
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "```\nprint(1)\n```")  # the fake echoes this back
            await self.say(pilot, f"/save --code {target}")
            self.assertEqual(target.read_text(), "print(1)\n")

    async def test_save_without_reply_or_without_code_writes_nothing(self):
        target = self.dir / "x.txt"
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, f"/save {target}")
            await self.say(pilot, "hello")
            await self.say(pilot, f"/save --code {target}")
            self.assertFalse(target.exists())

    async def test_save_asks_before_overwriting(self):
        target = self.dir / "reply.md"
        target.write_text("precious")
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hello")
            await self.say(pilot, f"/save {target}")
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            self.assertEqual(target.read_text(), "precious")
            await self.say(pilot, f"/save {target}")
            await pilot.press("y")
            await pilot.pause()
            self.assertEqual(target.read_text(), "msgs=1 last=hello\n")

    async def test_save_without_path_uses_a_timestamped_name_in_cwd(self):
        async with self.app.run_test(size=(120, 40)) as pilot:
            await self.settle(pilot)
            await self.say(pilot, "hello")
            old = Path.cwd()
            os.chdir(self.dir)
            try:
                await self.say(pilot, "/save")
            finally:
                os.chdir(old)
            files = list(self.dir.glob("reply-*.md"))
            self.assertEqual(len(files), 1)


class TestTuiOffline(AsyncTestCase):
    async def test_unreachable_server_does_not_crash(self):
        from tests.test_ollama_client import unused_url
        store = ChatStore(":memory:")
        self.addCleanup(store.close)
        app = OllamaTui(OllamaClient(unused_url(), retries=0), store)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIsNone(app.model)
            app.query_one("#prompt", PromptArea).text = "hi"
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(list(app.query("#log .msg")), [])  # refused: no model selected


if __name__ == "__main__":
    unittest.main()
