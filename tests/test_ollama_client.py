"""Tests for ollama_client.py: the client, chat sessions, retries/timeouts and the CLI.

Run everything with `python3 -m unittest`. Every test talks to a FakeOllama server (see
tests/fake_ollama.py), never to a real Ollama.
"""
import contextlib
import io
import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

import ollama_client
from ollama_client import ChatSession, OllamaClient
from tests.fake_ollama import FakeOllama


def unused_url() -> str:
    """A localhost URL where nothing is listening."""
    # Ask the OS for a free port, note the number and release it again: for the moment after
    # that, connecting to it is refused. That is how the "server is down" tests get a dead URL.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class FakeServerTestCase(unittest.TestCase):
    """Base class: every test gets its own fake server and a client pointed at it."""

    def setUp(self):  # unittest runs this before EACH test method
        self.server = FakeOllama().start()
        self.addCleanup(self.server.stop)  # cleanups run after the test, even if it failed
        # retry_delay=0.01 keeps the retry tests fast (the real default is 1 second).
        self.client = OllamaClient(self.server.url, default_model="fake:1", retry_delay=0.01)


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


class TestRetriesAndTimeouts(FakeServerTestCase):
    def chat_requests(self):
        return [r for r in self.server.requests if r[0] == "/api/chat"]

    def test_transient_error_is_retried(self):
        self.server.fail_next = 2
        self.assertEqual(self.client.generate("hi"), "msgs=1 last=hi")
        self.assertEqual(len(self.chat_requests()), 3)

    def test_gives_up_after_retries(self):
        self.server.fail_next = 10
        with self.assertRaises(ollama_client.ollama.ResponseError):
            self.client.generate("hi")
        self.assertEqual(len(self.chat_requests()), 3)  # 1 try + 2 retries

    def test_server_error_500_is_not_retried(self):
        with self.assertRaises(ollama_client.ollama.ResponseError):
            self.client.generate("boom")
        self.assertEqual(len(self.chat_requests()), 1)

    def test_retries_zero_disables_retrying(self):
        client = OllamaClient(self.server.url, retries=0)
        self.server.fail_next = 1
        with self.assertRaises(ollama_client.ollama.ResponseError):
            client.generate("hi", model="fake:1")
        self.assertEqual(len(self.chat_requests()), 1)

    def test_system_message_is_retried(self):
        self.server.fail_next = 1
        self.assertEqual(self.client.system_message("rules", "hi"), "msgs=2 last=hi")

    def test_streaming_is_not_retried(self):
        self.server.fail_next = 1
        with self.assertRaises(ollama_client.ollama.ResponseError):
            list(self.client.generate("hi", stream=True))
        self.assertEqual(len(self.chat_requests()), 1)

    def test_batch_process_retries_each_prompt(self):
        self.server.fail_next = 1
        results = self.client.batch_process(["a", "b"], max_workers=1)
        self.assertEqual([r["status"] for r in results.values()], ["success", "success"])

    def test_timeout_stops_a_hung_request(self):
        self.server.slow_seconds = 1.0
        client = OllamaClient(self.server.url, timeout=0.2, retries=0)
        start = time.monotonic()
        with self.assertRaises(httpx.TimeoutException):
            client.generate("slow", model="fake:1")
        self.assertLess(time.monotonic() - start, 0.9)

    def test_batch_process_reports_timeout_without_blocking_others(self):
        self.server.slow_seconds = 1.0
        client = OllamaClient(self.server.url, default_model="fake:1", timeout=0.2, retries=0)
        results = client.batch_process(["slow", "ok"], max_workers=2)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[1]["status"], "success")


class TestModelManagement(FakeServerTestCase):
    def test_pull_model_yields_progress(self):
        events = list(self.client.pull_model("new:1"))
        self.assertEqual(events[0]["status"], "pulling manifest")
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual((events[1]["completed"], events[1]["total"]), (50, 200))
        self.assertIn("new:1", [m["name"] for m in self.client.list_models()])

    def test_pull_missing_model_raises(self):
        with self.assertRaises(ollama_client.ollama.ResponseError):
            list(self.client.pull_model("missing"))

    def test_show_model(self):
        info = self.client.show_model("fake:1")
        self.assertEqual(info["family"], "llama")
        self.assertEqual(info["parameter_size"], "7B")
        self.assertEqual(info["quantization_level"], "Q4_0")
        self.assertEqual(info["capabilities"], ["completion"])

    def test_show_defaults_to_default_model(self):
        self.client.show_model()
        self.assertEqual(self.server.requests[-1][1]["model"], "fake:1")

    def test_delete_model(self):
        self.client.delete_model("fake:1")
        self.assertEqual([m["name"] for m in self.client.list_models()], ["nomic-embed-text:latest"])

    def test_delete_missing_model_raises(self):
        with self.assertRaises(ollama_client.ollama.ResponseError):
            self.client.delete_model("missing")


class TestStats(FakeServerTestCase):
    EXPECTED = {"prompt_tokens": 10, "reply_tokens": 2, "tokens_per_second": 4.0,
                "load_seconds": 0.1, "total_seconds": 0.7}

    def test_no_stats_before_first_reply(self):
        self.assertIsNone(ChatSession(self.client).last_stats)

    def test_stats_after_plain_reply(self):
        session = ChatSession(self.client)
        session.send("one")  # "msgs=1 last=one" is 2 words
        self.assertEqual(session.last_stats, self.EXPECTED)

    def test_stats_after_streamed_reply(self):
        session = ChatSession(self.client)
        "".join(session.send("one", stream=True))
        self.assertEqual(session.last_stats, self.EXPECTED)

    def test_stats_grow_with_history(self):
        session = ChatSession(self.client)
        session.send("one")
        session.send("two")
        self.assertEqual(session.last_stats["prompt_tokens"], 30)  # 3 messages

    def test_abandoned_stream_keeps_previous_stats(self):
        session = ChatSession(self.client)
        session.send("one")
        stream = session.send("two", stream=True)
        next(stream)
        stream.close()
        self.assertEqual(session.last_stats, self.EXPECTED)

    def test_chat_stats_treats_missing_values_as_zero(self):
        empty = ollama_client.ollama.ChatResponse(message=ollama_client.ollama.Message(role="assistant"))
        stats = ollama_client.chat_stats(empty)
        self.assertEqual((stats["reply_tokens"], stats["tokens_per_second"]), (0, 0.0))

    def test_running_models(self):
        loaded = self.client.running_models()
        self.assertEqual([m["name"] for m in loaded], ["fake:1"])
        self.assertEqual(loaded[0]["context_length"], 4096)
        self.assertEqual((loaded[0]["size"], loaded[0]["size_vram"]), (4_000_000_000, 3_000_000_000))


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
        """Run ollama_client.main() as if typed in a shell; return (exit code, stdout, stderr).

        stdin is the piped text; inputs are the answers to input() calls (for `chat`).
        """
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO(stdin if stdin is not None else "")
        # Remove the user's own OLLAMA_* settings so they cannot change the test outcome.
        env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_HOST", "OLLAMA_MODEL")}
        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.stdin", stdin),
        ]
        if inputs is not None:
            patches.append(mock.patch("builtins.input", side_effect=inputs))
        # ExitStack enters a list of context managers (the patches and redirects) and undoes
        # them all, in reverse order, when the block ends.
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            try:
                rc = ollama_client.main(["--host", self.server.url, *argv])
            except SystemExit as e:  # argparse errors and _resolve_model exit this way
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

    def test_pull_prints_progress(self):
        rc, out, _ = self.run_cli("pull", "new:1")
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines(),
                         ["pulling manifest", "pulling abc: 25%", "pulling abc: 100%", "success"])

    def test_pull_missing_model_exit_code(self):
        rc, _, err = self.run_cli("pull", "missing")
        self.assertEqual(rc, 1)
        self.assertIn("does not exist", err)

    def test_show(self):
        rc, out, _ = self.run_cli("show", "fake:1")
        self.assertEqual(rc, 0)
        self.assertIn("parameter_size: 7B", out)
        self.assertIn("capabilities: completion", out)

    def test_show_without_name_uses_first_installed_model(self):
        self.run_cli("show")
        self.assertEqual(self.server.requests[-1][1]["model"], "fake:1")

    def test_delete_with_yes(self):
        rc, out, _ = self.run_cli("delete", "fake:1", "--yes")
        self.assertEqual(rc, 0)
        self.assertNotIn("fake:1", self.server.models)

    def test_delete_without_yes_refuses_when_not_interactive(self):
        rc, _, err = self.run_cli("delete", "fake:1")
        self.assertEqual(rc, 2)
        self.assertIn("fake:1", self.server.models)

    def test_delete_confirmation_prompt(self):
        stdin = io.StringIO("")
        stdin.isatty = lambda: True
        with mock.patch("sys.stdin", stdin), mock.patch("builtins.input", return_value="n"), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = ollama_client.main(["--host", self.server.url, "delete", "fake:1"])
        self.assertEqual(rc, 1)
        self.assertIn("fake:1", self.server.models)

    def test_delete_missing_model_exit_code(self):
        rc, _, err = self.run_cli("delete", "missing", "-y")
        self.assertEqual(rc, 1)

    def test_timeout_and_retries_options_are_applied(self):
        with mock.patch.object(ollama_client, "OllamaClient", wraps=OllamaClient) as cls:
            self.run_cli("--timeout", "5", "--retries", "0", "list")
        _, kwargs = cls.call_args
        self.assertEqual((kwargs["timeout"], kwargs["retries"]), (5.0, 0))

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


class TestCliFiles(FakeServerTestCase):
    """File attachments and --output on the command line."""

    run_cli = TestCli.run_cli  # reuse the helper that runs main() with captured output

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "a.log").write_text("failed login root\nfailed login admin\n")

    def chat_requests(self):
        return [body for path, body in self.server.requests if path == "/api/chat"]

    def test_file_is_sent_with_the_prompt(self):
        rc, out, _ = self.run_cli("ask", "Summarise", "--file", str(self.dir / "a.log"))
        self.assertEqual(rc, 0)
        sent = self.chat_requests()[-1]["messages"][-1]["content"]
        self.assertTrue(sent.startswith("Summarise\n\n----- file: a.log -----\nfailed login root"))

    def test_file_without_prompt_is_enough(self):
        rc, _, _ = self.run_cli("ask", "-f", str(self.dir / "a.log"))
        self.assertEqual(rc, 0)

    def test_several_files(self):
        (self.dir / "b.txt").write_text("second")
        self.run_cli("ask", "x", "-f", str(self.dir / "a.log"), "-f", str(self.dir / "b.txt"))
        sent = self.chat_requests()[-1]["messages"][-1]["content"]
        self.assertIn("file: a.log", sent)
        self.assertIn("file: b.txt", sent)

    def test_bad_file_stops_before_asking_the_model(self):
        rc, _, err = self.run_cli("ask", "x", "-f", str(self.dir / "missing"))
        self.assertEqual(rc, 1)
        self.assertIn("not a file", err)
        self.assertEqual(self.chat_requests(), [])

    def test_output_writes_file_and_prints_no_answer(self):
        target = self.dir / "answer.md"
        rc, out, err = self.run_cli("ask", "hello", "-o", str(target))
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("Saved", err)
        self.assertEqual(target.read_text(), "msgs=1 last=hello\n")

    def test_output_refuses_existing_file_before_asking(self):
        target = self.dir / "answer.md"
        target.write_text("precious")
        rc, _, err = self.run_cli("ask", "hello", "-o", str(target))
        self.assertEqual(rc, 1)
        self.assertIn("--overwrite", err)
        self.assertEqual(target.read_text(), "precious")
        self.assertEqual(self.chat_requests(), [])

    def test_overwrite_replaces_file(self):
        target = self.dir / "answer.md"
        target.write_text("old")
        rc, _, _ = self.run_cli("ask", "hello", "-o", str(target), "--overwrite")
        self.assertEqual(rc, 0)
        self.assertEqual(target.read_text(), "msgs=1 last=hello\n")

    def test_code_saves_only_code_blocks(self):
        # The fake echoes the last message, so the prompt itself carries the fenced block.
        target = self.dir / "code.py"
        rc, _, _ = self.run_cli("ask", "```\nprint(1)\n```", "-o", str(target), "--code")
        self.assertEqual(rc, 0)
        self.assertEqual(target.read_text(), "print(1)\n")

    def test_code_without_a_code_block_writes_nothing(self):
        target = self.dir / "code.py"
        rc, _, err = self.run_cli("ask", "hello", "-o", str(target), "--code")
        self.assertEqual(rc, 1)
        self.assertIn("no code block", err)
        self.assertFalse(target.exists())

    def test_code_needs_output(self):
        rc, _, err = self.run_cli("ask", "hello", "--code")
        self.assertEqual(rc, 2)

    def test_chat_file_and_save_commands(self):
        target = self.dir / "saved.md"
        rc, out, err = self.run_cli(
            "chat", inputs=[f"/file {self.dir / 'a.log'}", "check this", f"/save {target}", "/exit"])
        self.assertEqual(rc, 0)
        self.assertIn("Attached a.log", out)
        self.assertIn("failed login root", self.chat_requests()[-1]["messages"][-1]["content"])
        self.assertIn("msgs=1 last=", target.read_text())

    def test_chat_attachment_applies_to_one_message_only(self):
        self.run_cli("chat", inputs=[f"/file {self.dir / 'a.log'}", "one", "two", "/exit"])
        first, second = [r["messages"][-1]["content"] for r in self.chat_requests()]
        self.assertIn("file: a.log", first)
        self.assertEqual(second, "two")

    def test_grep_and_tail_options_filter_the_file(self):
        log = self.dir / "auth.log"
        log.write_text("\n".join(f"{i} {'bad' if i % 2 else 'ok'}" for i in range(20)) + "\n")
        rc, _, _ = self.run_cli("ask", "x", "-f", str(log), "--grep", "bad", "--tail", "2")
        self.assertEqual(rc, 0)
        sent = self.chat_requests()[-1]["messages"][-1]["content"]
        self.assertIn("auth.log (grep 'bad', last 2; 2 of 20 lines)", sent)
        self.assertIn("17 bad\n19 bad", sent)
        self.assertNotIn("15 bad", sent)

    def test_grep_without_file_is_a_usage_error(self):
        rc, _, err = self.run_cli("ask", "x", "--grep", "bad")
        self.assertEqual(rc, 2)

    def test_chat_file_accepts_filters(self):
        log = self.dir / "auth.log"
        log.write_text("a\nb bad\nc\n")
        self.run_cli("chat", inputs=[f"/file {log} --grep bad", "go", "/exit"])
        sent = self.chat_requests()[-1]["messages"][-1]["content"]
        self.assertIn("b bad", sent)
        self.assertNotIn("\nc", sent)

    def test_chat_save_before_any_answer(self):
        rc, out, err = self.run_cli("chat", inputs=[f"/save {self.dir / 'x'}", "/exit"])
        self.assertIn("Nothing to save", err)
        self.assertFalse((self.dir / "x").exists())


    def big_file(self, chars=40000):
        path = self.dir / "big.log"
        path.write_text("a" * chars)
        return str(path)

    def test_too_big_prompt_is_refused_before_asking_the_model(self):
        rc, _, err = self.run_cli("ask", "x", "-f", self.big_file())
        self.assertEqual(rc, 1)
        self.assertIn("would be sent", err)
        self.assertEqual(self.chat_requests(), [])

    def test_piped_text_is_checked_too(self):
        rc, _, err = self.run_cli("ask", "x", stdin="a" * 40000)
        self.assertEqual(rc, 1)
        self.assertEqual(self.chat_requests(), [])

    def test_num_ctx_option_makes_it_fit_and_is_sent(self):
        rc, _, _ = self.run_cli("--num-ctx", "32768", "ask", "x", "-f", self.big_file())
        self.assertEqual(rc, 0)
        self.assertEqual(self.chat_requests()[-1]["options"]["num_ctx"], 32768)

    def test_chat_refuses_file_that_cannot_fit(self):
        rc, out, err = self.run_cli("chat", inputs=[f"/file {self.big_file()}", "/exit"])
        self.assertIn("would be sent", err)
        self.assertNotIn("Attached", out)

    def test_chat_refuses_file_plus_history_that_cannot_fit(self):
        medium = self.dir / "medium.log"
        medium.write_text("a" * 12000)  # alone ~4k tokens: fits 6,144
        self.run_cli("chat", inputs=[f"/file {medium}", "one", f"/file {medium}", "two", "/exit"])
        sent = [r["messages"][-1]["content"] for r in self.chat_requests()]
        self.assertEqual(len(sent), 1)  # the second one was refused

    def test_config_sets_num_ctx_default(self):
        args = ollama_client.build_parser({"num_ctx": 4096}).parse_args(["list"])
        self.assertEqual(args.num_ctx, 4096)
        self.assertEqual(ollama_client.build_parser().parse_args(["list"]).num_ctx, 8192)



class TestContextWindow(FakeServerTestCase):
    """num_ctx is sent with every chat request, and over-long prompts are caught beforehand."""

    def last_options(self):
        return [b for _, b in self.server.requests if b][-1]["options"]

    def test_default_window_is_8192(self):
        self.assertEqual(self.client.num_ctx, 8192)
        self.client.generate("hi")
        self.assertEqual(self.last_options()["num_ctx"], 8192)

    def test_every_chat_call_sends_it(self):
        self.client.num_ctx = 16384
        list(self.client.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(self.last_options()["num_ctx"], 16384)
        self.client.chat_once([{"role": "user", "content": "hi"}])
        self.assertEqual(self.last_options()["num_ctx"], 16384)
        ChatSession(self.client).send("hi")
        self.assertEqual(self.last_options()["num_ctx"], 16384)

    def test_none_leaves_it_to_the_server(self):
        self.client.num_ctx = None
        self.client.generate("hi")
        self.assertNotIn("num_ctx", self.last_options())

    def test_estimate_and_size_problem(self):
        self.assertEqual(ollama_client.estimate_tokens("x" * 300), 101)
        self.assertIsNone(ollama_client.prompt_size_problem(["x" * 300], 8192))
        self.assertIsNone(ollama_client.prompt_size_problem(["x" * 10**6], None))
        message = ollama_client.prompt_size_problem(["x" * 30000, "y"], 8192)
        self.assertIn("10,002 tokens", message)
        self.assertIn("6,144", message)  # 75% of the window; the rest is room for the answer


class TestFileFilters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "auth.log"
        self.path.write_text("\n".join(
            f"line {i} {'Failed password' if i % 2 == 0 else 'Accepted key'}" for i in range(10)) + "\n")

    def test_grep_keeps_matching_lines_and_labels_the_excerpt(self):
        label, text = ollama_client.read_text_file(self.path, grep="Failed")
        self.assertEqual(text.splitlines()[0], "line 0 Failed password")
        self.assertEqual(len(text.splitlines()), 5)
        self.assertEqual(label, "auth.log (grep 'Failed'; 5 of 10 lines)")

    def test_tail_keeps_last_lines(self):
        label, text = ollama_client.read_text_file(self.path, tail=3)
        self.assertEqual(text.splitlines(), ["line 7 Accepted key", "line 8 Failed password",
                                             "line 9 Accepted key"])
        self.assertEqual(label, "auth.log (last 3; 3 of 10 lines)")

    def test_grep_then_tail(self):
        label, text = ollama_client.read_text_file(self.path, grep="Failed", tail=2)
        self.assertEqual(text.splitlines(), ["line 6 Failed password", "line 8 Failed password"])
        self.assertIn("grep 'Failed', last 2; 2 of 10", label)

    def test_bad_filters_are_file_errors(self):
        for kwargs, message in [({"grep": "("}, "invalid --grep"), ({"grep": "nomatch"}, "no lines"),
                                ({"tail": 0}, "1 or more")]:
            with self.subTest(kwargs), self.assertRaisesRegex(ollama_client.FileError, message):
                ollama_client.read_text_file(self.path, **kwargs)

    def test_filter_allows_a_file_that_is_too_big_unfiltered(self):
        with mock.patch.object(ollama_client, "MAX_FILE_BYTES", 100):
            with self.assertRaisesRegex(ollama_client.FileError, "--grep / --tail"):
                ollama_client.read_text_file(self.path)  # about 200 bytes
            label, text = ollama_client.read_text_file(self.path, tail=2)
            self.assertEqual(len(text.splitlines()), 2)

    def test_excerpt_that_is_still_too_big_is_refused(self):
        with mock.patch.object(ollama_client, "MAX_FILE_BYTES", 100):
            with self.assertRaisesRegex(ollama_client.FileError, "still"):
                ollama_client.read_text_file(self.path, tail=9)

    def test_parse_file_args(self):
        parse = ollama_client.parse_file_args
        self.assertEqual(parse("my file.log"), ("my file.log", None, None))
        self.assertEqual(parse("auth.log --grep 'Failed password' --tail 50"),
                         ("auth.log", "Failed password", 50))
        self.assertEqual(parse("--tail 5 auth.log"), ("auth.log", None, 5))
        self.assertEqual(parse("my file.log --tail 5"), ("my file.log", None, 5))

    def test_parse_file_args_errors(self):
        for text in ("auth.log --grep", "auth.log --tail x", "--tail 5", "a --grep 'unclosed"):
            with self.subTest(text), self.assertRaises(ollama_client.FileError):
                ollama_client.parse_file_args(text)


class TestFileHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_read_text_file(self):
        (self.dir / "a.log").write_text("line1\nline2\n")
        self.assertEqual(ollama_client.read_text_file(self.dir / "a.log"), ("a.log", "line1\nline2"))

    def test_read_refuses_missing_big_binary_and_non_utf8(self):
        (self.dir / "bin").write_bytes(b"ab\0cd")
        (self.dir / "latin").write_bytes("blåbær".encode("latin-1"))
        (self.dir / "big").write_text("x" * 50)
        with mock.patch.object(ollama_client, "MAX_FILE_BYTES", 10):
            for name, message in [("nope", "not a file"), ("bin", "binary"),
                                  ("latin", "not UTF-8"), ("big", "too large")]:
                with self.subTest(name), self.assertRaisesRegex(ollama_client.FileError, message):
                    ollama_client.read_text_file(self.dir / name)

    def test_build_and_hide_roundtrip(self):
        prompt = ollama_client.build_prompt_with_files(
            "Summarise", [("a.log", "x\ny"), ("b.txt", "z")])
        self.assertIn("----- file: a.log -----\nx\ny\n----- end of file -----", prompt)
        self.assertEqual(ollama_client.hide_attachments(prompt),
                         "Summarise\n📎 a.log (2 lines)\n📎 b.txt (1 line)")

    def test_prompt_without_text_and_message_without_files(self):
        self.assertEqual(ollama_client.build_prompt_with_files("hi", []), "hi")
        self.assertTrue(ollama_client.build_prompt_with_files("", [("a", "b")]).startswith("-----"))
        self.assertEqual(ollama_client.hide_attachments("plain"), "plain")

    def test_extract_code_blocks(self):
        reply = "Intro\n```python\nprint(1)\n```\ntext\n```\nls\n```\n"
        self.assertEqual(ollama_client.extract_code_blocks(reply), "print(1)\n\nls\n")
        with self.assertRaises(ollama_client.FileError):
            ollama_client.extract_code_blocks("no code here")

    def test_write_text_file_never_overwrites_by_default(self):
        path = self.dir / "out.txt"
        ollama_client.write_text_file(path, "one")
        self.assertEqual(path.read_text(), "one\n")
        with self.assertRaisesRegex(ollama_client.FileError, "already exists"):
            ollama_client.write_text_file(path, "two")
        self.assertEqual(path.read_text(), "one\n")
        ollama_client.write_text_file(path, "two", overwrite=True)
        self.assertEqual(path.read_text(), "two\n")

    def test_write_to_missing_folder_is_a_file_error(self):
        with self.assertRaises(ollama_client.FileError):
            ollama_client.write_text_file(self.dir / "no" / "dir.txt", "x")


class TestConfigFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Remove OLLAMA_* so the user's own environment cannot change the outcome.
        env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_HOST", "OLLAMA_MODEL")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, text: str) -> Path:
        path = Path(self.tmp.name) / "config.toml"
        path.write_text(text)
        return path

    def test_missing_default_file_gives_empty_config(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": self.tmp.name}):
            self.assertEqual(ollama_client.load_config(), {})

    def test_default_path_follows_xdg(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/x"}):
            self.assertEqual(ollama_client.default_config_path(),
                             Path("/x/python_ollama/config.toml"))

    def test_loads_valid_file(self):
        path = self.write('host = "http://srv:11434"\nretries = 4\ntemperature = 1\n')
        config = ollama_client.load_config(path)
        self.assertEqual(config, {"host": "http://srv:11434", "retries": 4, "temperature": 1})

    def test_explicit_missing_file_is_an_error(self):
        with self.assertRaisesRegex(ollama_client.ConfigError, "not found"):
            ollama_client.load_config(Path(self.tmp.name) / "nope.toml")

    def test_unknown_key_is_rejected(self):
        path = self.write("timout = 30\n")
        with self.assertRaisesRegex(ollama_client.ConfigError, "unknown setting 'timout'"):
            ollama_client.load_config(path)

    def test_wrong_type_is_rejected(self):
        path = self.write('retries = "two"\n')
        with self.assertRaisesRegex(ollama_client.ConfigError, "'retries' has the wrong type"):
            ollama_client.load_config(path)

    def test_boolean_is_not_a_number(self):
        path = self.write("retries = true\n")
        with self.assertRaises(ollama_client.ConfigError):
            ollama_client.load_config(path)

    def test_invalid_toml_is_an_error(self):
        path = self.write("host = \n")
        with self.assertRaisesRegex(ollama_client.ConfigError, "cannot read"):
            ollama_client.load_config(path)

    def test_config_path_from_argv(self):
        self.assertEqual(ollama_client.config_path_from_argv(["--config", "a.toml", "list"]),
                         Path("a.toml"))
        self.assertIsNone(ollama_client.config_path_from_argv(["list"]))

    def parse(self, config, *argv, **env):
        with mock.patch.dict(os.environ, env):
            return ollama_client.build_parser(config).parse_args(list(argv))

    def test_built_in_defaults_without_config(self):
        args = self.parse({}, "ask", "hi")
        self.assertEqual((args.host, args.model, args.timeout, args.retries, args.temperature),
                         (ollama_client.DEFAULT_HOST, None, ollama_client.DEFAULT_TIMEOUT,
                          ollama_client.DEFAULT_RETRIES, 0.7))

    def test_config_replaces_defaults(self):
        config = {"host": "http://srv:1", "model": "m:1", "timeout": 30, "retries": 5,
                  "temperature": 0.2, "system": "be brief"}
        args = self.parse(config, "ask", "hi")
        self.assertEqual((args.host, args.model, args.timeout, args.retries),
                         ("http://srv:1", "m:1", 30.0, 5))
        self.assertEqual((args.temperature, args.system), (0.2, "be brief"))

    def test_environment_beats_config(self):
        args = self.parse({"host": "http://cfg:1", "model": "cfg"}, "list",
                          OLLAMA_HOST="http://env:1", OLLAMA_MODEL="env")
        self.assertEqual((args.host, args.model), ("http://env:1", "env"))

    def test_command_line_beats_everything(self):
        args = self.parse({"host": "http://cfg:1", "temperature": 0.2}, "--host", "http://cli:1",
                          "ask", "hi", "-t", "1.5", OLLAMA_HOST="http://env:1")
        self.assertEqual((args.host, args.temperature), ("http://cli:1", 1.5))

    def test_main_reports_bad_config(self):
        path = self.write("bogus = 1\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = ollama_client.main(["--config", str(path), "list"])
        self.assertEqual(rc, 2)
        self.assertIn("Config error", err.getvalue())

    def test_main_uses_config_file(self):
        server = FakeOllama()
        server.start()
        self.addCleanup(server.stop)
        path = self.write(f'host = "{server.url}"\n')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ollama_client.main(["--config", str(path), "list"])
        self.assertEqual(rc, 0)
        self.assertIn("fake:1", out.getvalue())


if __name__ == "__main__":
    unittest.main()
