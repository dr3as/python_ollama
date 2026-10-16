import contextlib
import io
import json
import os
import socket
import unittest
from unittest import mock

import ollama_client
from ollama_client import ChatSession, OllamaClient
from tests.fake_ollama import FakeOllama


def unused_url() -> str:
    """A localhost URL where nothing is listening."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class FakeServerTestCase(unittest.TestCase):
    def setUp(self):
        self.server = FakeOllama().start()
        self.addCleanup(self.server.stop)
        self.client = OllamaClient(self.server.url, default_model="fake:1")


class TestClient(FakeServerTestCase):
    def test_check_connection(self):
        self.assertTrue(self.client.check_connection())

    def test_check_connection_down(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(OllamaClient(unused_url()).check_connection())

    def test_trailing_slash_in_base_url_is_stripped(self):
        self.assertEqual(OllamaClient(self.server.url + "/").base_url, self.server.url)

    def test_list_models(self):
        names = [m["name"] for m in self.client.list_models()]
        self.assertEqual(names, ["fake:1", "nomic-embed-text:latest"])

    def test_list_embeddings_models(self):
        self.assertEqual(self.client.list_embeddings_models(), ["nomic-embed-text:latest"])

    def test_list_embeddings_models_returns_empty_when_down(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(OllamaClient(unused_url()).list_embeddings_models(), [])

    def test_generate_returns_text(self):
        self.assertEqual(self.client.generate("hello"), "msgs=1 last=hello")

    def test_generate_sends_options_and_default_model(self):
        self.client.generate("hello", temperature=0.2, max_tokens=50)
        _, body = self.server.requests[-1]
        self.assertEqual(body["model"], "fake:1")
        self.assertEqual(body["options"]["temperature"], 0.2)
        self.assertEqual(body["options"]["num_predict"], 50)

    def test_generate_omits_num_predict_by_default(self):
        self.client.generate("hello")
        _, body = self.server.requests[-1]
        self.assertNotIn("num_predict", body["options"])

    def test_generate_stream_yields_chunks(self):
        chunks = list(self.client.generate("hello", stream=True))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), "msgs=1 last=hello")

    def test_system_message_sends_system_and_user(self):
        reply = self.client.system_message("be nice", "hi")
        _, body = self.server.requests[-1]
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(reply, "msgs=2 last=hi")

    def test_run_agent_uses_prompt(self):
        self.assertEqual(self.client.run_agent("rules", "do it"), "msgs=2 last=do it")

    def test_run_agent_returns_none_on_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(self.client.run_agent("rules", "boom"))

    def test_embed(self):
        self.assertEqual(self.client.embed("text"), [0.1, 0.2, 0.3])
        path, body = self.server.requests[-1]
        self.assertEqual(path, "/api/embed")
        self.assertEqual(body["model"], "nomic-embed-text")
        self.assertEqual(body["input"], "text")

    def test_batch_process_keys_and_errors(self):
        results = self.client.batch_process(["a", "boom", "c"])
        self.assertEqual(list(results), [0, 1, 2])
        self.assertEqual(results[0], {"status": "success", "response": "msgs=1 last=a"})
        self.assertEqual(results[1]["status"], "error")
        self.assertIn("model exploded", results[1]["error"])
        self.assertEqual(results[2]["status"], "success")

    def test_batch_process_empty(self):
        self.assertEqual(self.client.batch_process([]), {})


class TestChatSession(FakeServerTestCase):
    def test_history_grows_between_turns(self):
        session = ChatSession(self.client)
        self.assertEqual(session.send("one"), "msgs=1 last=one")
        self.assertEqual(session.send("two"), "msgs=3 last=two")
        self.assertEqual([m["role"] for m in session.messages],
                         ["user", "assistant", "user", "assistant"])

    def test_system_prompt_is_first_message(self):
        session = ChatSession(self.client, system="be brief")
        self.assertEqual(session.send("hi"), "msgs=2 last=hi")
        self.assertEqual(session.messages[0], {"role": "system", "content": "be brief"})

    def test_reset_keeps_system_prompt(self):
        session = ChatSession(self.client, system="be brief")
        session.send("hi")
        session.reset()
        self.assertEqual(session.messages, [{"role": "system", "content": "be brief"}])

    def test_reset_without_system_prompt_is_empty(self):
        session = ChatSession(self.client)
        session.send("hi")
        session.reset()
        self.assertEqual(session.messages, [])

    def test_streamed_reply_is_added_after_completion(self):
        session = ChatSession(self.client)
        stream = session.send("one", stream=True)
        self.assertEqual(len(session.messages), 1)  # reply not stored until consumed
        text = "".join(stream)
        self.assertEqual(text, "msgs=1 last=one")
        self.assertEqual(session.messages[-1], {"role": "assistant", "content": text})

    def test_failed_send_leaves_history_unchanged(self):
        session = ChatSession(self.client)
        session.send("one")
        with self.assertRaises(Exception):
            session.send("boom")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.send("two"), "msgs=3 last=two")

    def test_failed_stream_leaves_history_unchanged(self):
        session = ChatSession(self.client)
        stream = session.send("boom", stream=True)  # the HTTP error only surfaces on iteration
        with self.assertRaises(Exception):
            list(stream)
        self.assertEqual(session.messages, [])

    def test_abandoned_stream_drops_user_message(self):
        session = ChatSession(self.client)
        stream = session.send("one", stream=True)
        next(stream)
        stream.close()
        self.assertEqual(session.messages, [])


class TestCli(FakeServerTestCase):
    def run_cli(self, *argv, stdin=None, inputs=None):
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO(stdin if stdin is not None else "")
        env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_HOST", "OLLAMA_MODEL")}
        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.stdin", stdin),
        ]
        if inputs is not None:
            patches.append(mock.patch("builtins.input", side_effect=inputs))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            try:
                rc = ollama_client.main(["--host", self.server.url, *argv])
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue(), err.getvalue()

    def test_list(self):
        rc, out, _ = self.run_cli("list")
        self.assertEqual(rc, 0)
        self.assertEqual(out.split(), ["fake:1", "nomic-embed-text:latest"])

    def test_ask_with_prompt_uses_first_installed_model(self):
        rc, out, _ = self.run_cli("ask", "hello")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "msgs=1 last=hello")
        self.assertEqual(self.server.requests[-1][1]["model"], "fake:1")

    def test_ask_respects_model_option(self):
        self.run_cli("-m", "other:2", "ask", "hello")
        self.assertEqual(self.server.requests[-1][1]["model"], "other:2")

    def test_ask_system_prompt(self):
        rc, out, _ = self.run_cli("ask", "hello", "-s", "be brief")
        self.assertEqual(out.strip(), "msgs=2 last=hello")

    def test_ask_appends_piped_stdin(self):
        rc, out, _ = self.run_cli("ask", "summarise:", stdin="log line\n")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "msgs=1 last=summarise:\n\nlog line")

    def test_ask_stdin_only(self):
        rc, out, _ = self.run_cli("ask", stdin="just piped")
        self.assertEqual(out.strip(), "msgs=1 last=just piped")

    def test_ask_nothing_to_ask(self):
        rc, out, err = self.run_cli("ask")
        self.assertEqual(rc, 2)
        self.assertIn("Nothing to ask", err)

    def test_embed_prints_json(self):
        rc, out, _ = self.run_cli("embed", "hi")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [0.1, 0.2, 0.3])

    def test_chat_keeps_history_and_reset_clears_it(self):
        rc, out, _ = self.run_cli("chat", inputs=["one", "two", "/reset", "three", "/exit"])
        self.assertEqual(rc, 0)
        self.assertIn("msgs=1 last=one", out)
        self.assertIn("msgs=3 last=two", out)
        self.assertIn("History cleared.", out)
        self.assertIn("msgs=1 last=three", out)

    def test_chat_ends_on_eof(self):
        rc, _, _ = self.run_cli("chat", inputs=EOFError)
        self.assertEqual(rc, 0)

    def test_chat_survives_server_error(self):
        rc, out, err = self.run_cli("chat", inputs=["boom", "ok", "/exit"])
        self.assertEqual(rc, 0)
        self.assertIn("model exploded", err)
        self.assertIn("msgs=1 last=ok", out)

    def test_connection_error_exit_code(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch("sys.stdin", io.StringIO("")):
            rc = ollama_client.main(["--host", unused_url(), "-m", "x", "ask", "hi"])
        self.assertEqual(rc, 1)
        self.assertIn("Cannot reach Ollama", err.getvalue())

    def test_no_models_installed(self):
        self.server.models.clear()
        rc, _, _ = self.run_cli("ask", "hi")
        self.assertIn("No models installed", str(rc))

    def test_command_is_required(self):
        rc, _, err = self.run_cli()
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
