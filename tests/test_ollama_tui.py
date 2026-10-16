import asyncio
import unittest

from textual.widgets import Input, ListView

from ollama_client import OllamaClient
from ollama_tui import ChatStore, ConfirmScreen, InfoScreen, OllamaTui
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
