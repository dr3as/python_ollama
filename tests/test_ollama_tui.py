import asyncio
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from textual.widgets import Input, ListView, TabbedContent

from ollama_client import OllamaClient
import ollama_tui
from ollama_tui import (
    ChatStore, ConfirmScreen, InfoScreen, OllamaTui, format_bytes, format_resources, format_stats,
    is_local_host, parse_nvidia_smi, read_gpus, read_host_stats,
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


class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        # unittest starts its event loop in debug mode, which slows Textual down about 10x
        # (a test run goes from ~65 s to ~5 s). Override this hook to turn debug mode off.
        self._asyncioRunner = asyncio.Runner(debug=False)


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
    async def asyncSetUp(self):
        self.server = FakeOllama().start()
        self.addCleanup(self.server.stop)
        self.store = ChatStore(":memory:")
        self.addCleanup(self.store.close)
        self.app = OllamaTui(OllamaClient(self.server.url, retry_delay=0.01), self.store)

    async def settle(self, pilot):
        await self.app.workers.wait_for_complete()
        await pilot.pause()

    async def say(self, pilot, text):
        self.app.query_one("#prompt", Input).value = text
        await pilot.press("enter")
        await self.settle(pilot)

    def log_texts(self):
        return [w.text
                for w in self.app.query("#log .msg")]


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
            self.app.query_one("#prompt", Input).value = "slow"
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
            app.query_one("#prompt", Input).value = "hi"
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(list(app.query("#log .msg")), [])  # refused: no model selected


if __name__ == "__main__":
    unittest.main()
