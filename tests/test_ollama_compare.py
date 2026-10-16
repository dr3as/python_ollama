"""Tests for ollama_compare.py (the model comparison) and the client methods it relies on.

No user interface is involved here: everything runs against the fake server in
tests/fake_ollama.py, which records which models were loaded and unloaded and when.
"""
import json
import threading
import unittest
from unittest import mock

import ollama_compare
from ollama_client import OllamaClient
from ollama_compare import (
    GpuSampler, ModelRun, build_judge_prompt, compare_models, comparison_markdown, judge_runs,
    metric_rows, parse_judge_reply, run_model, unload_all, wait_until_unloaded,
)
from tests.fake_ollama import FakeOllama


class CompareTestCase(unittest.TestCase):
    """Base class: a fake server with three models, of which a:1 starts out loaded."""

    def setUp(self):
        self.server = FakeOllama(models=("a:1", "b:1", "c:1")).start()
        self.addCleanup(self.server.stop)
        self.client = OllamaClient(self.server.url, retry_delay=0.01)
        # The real wait for Ollama to confirm an unload is 20 s; tests that expect it to fail
        # would be painfully slow, so shorten it for every test in this file.
        patcher = mock.patch.object(ollama_compare, "UNLOAD_TIMEOUT", 0.3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def request_bodies(self, path):
        return [body for p, body in self.server.requests if p == path]


class TestClientAdditions(CompareTestCase):
    def test_unload_model_sends_keep_alive_zero_and_frees_it(self):
        self.client.unload_model("a:1")
        self.assertEqual(self.request_bodies("/api/generate")[0]["keep_alive"], 0)
        self.assertEqual(self.server.running, [])

    def test_stream_chat_returns_raw_chunks_with_timings_in_the_last(self):
        chunks = list(self.client.stream_chat([{"role": "user", "content": "hi"}], model="a:1",
                                              temperature=0.2))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].eval_count, 2)
        self.assertEqual(self.request_bodies("/api/chat")[0]["options"]["temperature"], 0.2)

    def test_chat_once_asks_for_json_only_when_requested(self):
        self.server.judge_reply = '{"A": 1}'
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(self.client.chat_once(messages, model="a:1", json_format=True), '{"A": 1}')
        self.assertEqual(self.request_bodies("/api/chat")[-1]["format"], "json")
        self.assertEqual(self.client.chat_once(messages, model="a:1"), "msgs=1 last=hi")
        self.assertNotIn("format", self.request_bodies("/api/chat")[-1])


class TestUnloading(CompareTestCase):
    def test_wait_until_unloaded(self):
        self.server.running.clear()
        self.assertTrue(wait_until_unloaded(self.client, timeout=0.3, poll=0.05))

    def test_wait_gives_up_while_a_model_is_still_loaded(self):
        self.assertFalse(wait_until_unloaded(self.client, timeout=0.2, poll=0.05))

    def test_unload_all(self):
        self.server.running[:] = ["a:1", "b:1"]
        self.assertTrue(unload_all(self.client, timeout=0.3))
        self.assertEqual(self.server.running, [])

    def test_unload_all_reports_a_model_that_will_not_go(self):
        self.server.stuck.add("a:1")
        self.assertFalse(unload_all(self.client, timeout=0.2))


class TestGpuSampler(unittest.TestCase):
    def test_tracks_baseline_and_peak(self):
        readings = iter([1_000, 3_000, 2_000, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500])

        def reader():
            return [{"vram_used": next(readings), "util_percent": 50.0}]

        sampler = GpuSampler(reader, interval=0.01)
        sampler.start()
        for _ in range(100):  # wait until the background thread has read a few values
            if sampler.peak_vram == 3_000:
                break
            threading.Event().wait(0.01)
        sampler.stop()
        self.assertEqual(sampler.baseline_vram, 1_000)
        self.assertEqual(sampler.peak_vram, 3_000)
        self.assertEqual(sampler.extra_vram, 2_000)

    def test_adds_up_several_gpus_and_takes_the_busiest_load(self):
        sampler = GpuSampler(lambda: [{"vram_used": 100, "util_percent": 10.0},
                                      {"vram_used": 200, "util_percent": 80.0}])
        sampler._sample()
        self.assertEqual((sampler.peak_vram, sampler.peak_util), (300, 80.0))

    def test_no_gpu_means_no_numbers(self):
        sampler = GpuSampler(lambda: None)
        sampler._sample()
        self.assertIsNone(sampler.extra_vram)

    def test_missing_values_count_as_zero(self):
        sampler = GpuSampler(lambda: [{"vram_used": None, "util_percent": None}])
        sampler._sample()
        self.assertEqual((sampler.peak_vram, sampler.peak_util), (0, 0))


class TestRunModel(CompareTestCase):
    def test_successful_run_collects_answer_timings_and_memory(self):
        run = run_model(self.client, "b:1", "hi")
        self.assertEqual((run.status, run.answer), ("ok", "msgs=1 last=hi"))
        self.assertEqual((run.prompt_tokens, run.reply_tokens), (10, 2))
        self.assertEqual(run.tokens_per_second, 4.0)
        self.assertEqual(run.load_seconds, 0.1)
        self.assertEqual((run.size_bytes, run.vram_bytes, run.context_length),
                         (4_000_000_000, 3_000_000_000, 4096))
        self.assertGreater(run.first_token_seconds, 0)
        self.assertGreaterEqual(run.total_seconds, run.first_token_seconds)
        self.assertEqual(run.warning, "")

    def test_starts_clean_and_leaves_nothing_loaded(self):
        run_model(self.client, "b:1", "hi")
        self.assertEqual(self.server.events, [("unload", "a:1"), ("load", "b:1"), ("unload", "b:1")])
        self.assertEqual(self.server.running, [])

    def test_system_prompt_and_temperature_are_sent(self):
        run_model(self.client, "b:1", "hi", system="be brief", temperature=0.1)
        body = self.request_bodies("/api/chat")[0]
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["options"]["temperature"], 0.1)

    def test_error_is_reported_not_raised(self):
        self.server.bad_models.add("b:1")
        run = run_model(self.client, "b:1", "hi")
        self.assertEqual(run.status, "error")
        self.assertIn("model failed to load", run.error)
        self.assertIsNone(run.first_token_seconds)

    def test_warns_when_the_previous_model_cannot_be_unloaded(self):
        self.server.stuck.add("a:1")
        run = run_model(self.client, "b:1", "hi")
        self.assertEqual(run.status, "ok")
        self.assertIn("another model was still loaded", run.warning)

    def test_cancel_stops_the_stream_and_keeps_the_partial_answer(self):
        cancel = threading.Event()
        run = run_model(self.client, "b:1", "hi", cancel=cancel,
                        on_text=lambda model, text, chunks, seconds: cancel.set())
        self.assertEqual(run.status, "stopped")
        self.assertEqual(run.answer, "msgs=1 ")  # only the first streamed piece arrived
        self.assertEqual(self.server.running, [])

    def test_callbacks_report_phases_and_growing_text(self):
        statuses, texts = [], []
        run_model(self.client, "b:1", "hi", on_status=lambda m, t: statuses.append((m, t)),
                  on_text=lambda m, text, chunks, seconds: texts.append((text, chunks)))
        self.assertEqual([t for _, t in statuses][-1], "generating...")
        self.assertEqual(texts, [("msgs=1 ", 1), ("msgs=1 last=hi", 2)])

    def test_gpu_numbers_when_a_reader_is_given(self):
        calls = []

        def reader():
            calls.append(1)
            return [{"vram_used": 1_000 * len(calls), "util_percent": 40.0 + len(calls)}]

        run = run_model(self.client, "b:1", "hi", gpu_reader=reader)
        self.assertGreater(run.gpu_extra_vram, 0)
        self.assertGreater(run.gpu_peak_util, 40.0)

    def test_no_gpu_numbers_without_a_reader(self):
        run = run_model(self.client, "b:1", "hi")
        self.assertIsNone(run.gpu_extra_vram)


class TestCompareModels(CompareTestCase):
    def test_models_run_one_at_a_time_in_order(self):
        runs, problem = compare_models(self.client, "hi", ["a:1", "b:1", "c:1"])
        self.assertIsNone(problem)
        self.assertEqual([(r.model, r.status) for r in runs],
                         [("a:1", "ok"), ("b:1", "ok"), ("c:1", "ok")])
        self.assertEqual(self.server.events, [
            ("unload", "a:1"), ("load", "a:1"), ("unload", "a:1"),
            ("load", "b:1"), ("unload", "b:1"),
            ("load", "c:1"), ("unload", "c:1"),
        ])
        self.assertEqual(self.server.max_loaded, 1)  # never two models in memory at once

    def test_one_failing_model_does_not_stop_the_others(self):
        self.server.bad_models.add("b:1")
        runs, _ = compare_models(self.client, "hi", ["a:1", "b:1", "c:1"])
        self.assertEqual([r.status for r in runs], ["ok", "error", "ok"])

    def test_cancel_skips_the_remaining_models_and_the_judge(self):
        cancel = threading.Event()
        self.server.judge_reply = '{"A": 5, "B": 5}'
        runs, problem = compare_models(self.client, "hi", ["a:1", "b:1", "c:1"], cancel=cancel,
                                       judge_model="c:1", on_run=lambda run: cancel.set())
        self.assertEqual([r.status for r in runs], ["ok", "skipped", "skipped"])
        self.assertIsNone(problem)
        self.assertFalse([b for b in self.request_bodies("/api/chat") if b.get("format") == "json"])

    def test_on_run_is_called_for_every_model_in_order(self):
        seen = []
        compare_models(self.client, "hi", ["a:1", "b:1"], on_run=lambda run: seen.append(run.model))
        self.assertEqual(seen, ["a:1", "b:1"])

    def test_judge_scores_are_attached(self):
        self.server.judge_reply = ('{"A": {"score": 8, "reason": "clear"},'
                                   ' "B": {"score": 4, "reason": "vague"}}')
        judging = []
        runs, problem = compare_models(self.client, "hi", ["a:1", "b:1"], judge_model="c:1",
                                       on_judging=judging.append)
        self.assertIsNone(problem)
        self.assertEqual(judging, ["c:1"])
        self.assertEqual(sorted(r.judge_score for r in runs), [4.0, 8.0])  # which is which is random
        self.assertEqual(self.server.running, [])  # the judge was unloaded too
        self.assertEqual(self.server.max_loaded, 1)


class TestJudge(CompareTestCase):
    def ok_run(self, model, answer):
        return ModelRun(model=model, status="ok", answer=answer)

    def test_scores_follow_the_order_the_judge_saw(self):
        runs = [self.ok_run("a:1", "first answer"), self.ok_run("b:1", "second answer")]
        self.server.judge_reply = ('{"A": {"score": 9, "reason": "great"},'
                                   ' "B": {"score": 3, "reason": "weak"}}')
        self.assertIsNone(judge_runs(self.client, "c:1", "q", runs, order=[1, 0]))
        # Order [1, 0] means the judge saw b:1 as "A" and a:1 as "B".
        self.assertEqual((runs[1].judge_score, runs[1].judge_reason), (9.0, "great"))
        self.assertEqual((runs[0].judge_score, runs[0].judge_reason), (3.0, "weak"))

    def test_the_judge_prompt_hides_model_names(self):
        runs = [self.ok_run("secret-a:1", "first answer"), self.ok_run("secret-b:1", "second answer")]
        self.server.judge_reply = '{"A": 5, "B": 5}'
        judge_runs(self.client, "c:1", "the question", runs, order=[0, 1])
        text = self.request_bodies("/api/chat")[-1]["messages"][0]["content"]
        self.assertIn("ANSWER A:\nfirst answer", text)
        self.assertIn("the question", text)
        self.assertNotIn("secret", text)

    def test_failed_runs_are_not_judged(self):
        runs = [self.ok_run("a:1", "answer"), ModelRun(model="b:1", status="error", error="x"),
                self.ok_run("c:1", "answer two")]
        self.server.judge_reply = '{"A": 6, "B": 7}'
        judge_runs(self.client, "c:1", "q", runs, order=[0, 2])
        self.assertIsNone(runs[1].judge_score)
        self.assertEqual((runs[0].judge_score, runs[2].judge_score), (6.0, 7.0))

    def test_needs_two_answers(self):
        problem = judge_runs(self.client, "c:1", "q", [self.ok_run("a:1", "only one")])
        self.assertIn("at least two", problem)

    def test_unreadable_reply_is_reported(self):
        runs = [self.ok_run("a:1", "x"), self.ok_run("b:1", "y")]
        self.server.judge_reply = "I think A is better."
        self.assertIn("could not be read", judge_runs(self.client, "c:1", "q", runs))
        self.assertIsNone(runs[0].judge_score)

    def test_failing_judge_model_is_reported_not_raised(self):
        runs = [self.ok_run("a:1", "x"), self.ok_run("b:1", "y")]
        self.server.bad_models.add("c:1")
        self.assertIn("judge failed", judge_runs(self.client, "c:1", "q", runs))


class TestParseJudgeReply(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_judge_reply('{"A": {"score": 7, "reason": "ok"}}', 1), {0: (7.0, "ok")})

    def test_json_wrapped_in_prose_and_code_fences(self):
        text = 'Sure! Here you go:\n```json\n{"A": {"score": 7, "reason": "ok"}}\n```\nHope it helps.'
        self.assertEqual(parse_judge_reply(text, 1), {0: (7.0, "ok")})

    def test_bare_numbers_are_accepted(self):
        self.assertEqual(parse_judge_reply('{"A": 7, "B": 4.5}', 2), {0: (7.0, ""), 1: (4.5, "")})

    def test_scores_are_clamped_to_1_to_10(self):
        self.assertEqual(parse_judge_reply('{"A": 15, "B": 0}', 2), {0: (10.0, ""), 1: (1.0, "")})

    def test_unusable_entries_are_skipped(self):
        text = '{"A": "good", "B": true, "C": null, "D": {"score": 6}}'
        self.assertEqual(parse_judge_reply(text, 4), {3: (6.0, "")})

    def test_more_entries_than_answers_are_ignored(self):
        self.assertEqual(parse_judge_reply('{"A": 5, "B": 6}', 1), {0: (5.0, "")})

    def test_garbage(self):
        for text in ("", "no json here", "{not json}", "[1, 2]"):
            self.assertEqual(parse_judge_reply(text, 2), {}, text)

    def test_build_prompt_labels_answers_with_letters(self):
        prompt = build_judge_prompt("why?", ["one", "two", "three"])
        for expected in ("REQUEST:\nwhy?", "ANSWER A:\none", "ANSWER C:\nthree", '"C": {"score"'):
            self.assertIn(expected, prompt)


def make_run(model, **kwargs):
    defaults = dict(status="ok", answer="an answer", first_token_seconds=1.0, total_seconds=5.0,
                    tokens_per_second=40.0, prompt_tokens=10, reply_tokens=200, load_seconds=2.0,
                    size_bytes=4_000_000_000, vram_bytes=4_000_000_000, context_length=4096)
    defaults.update(kwargs)
    return ModelRun(model=model, **defaults)


class TestMetricRows(unittest.TestCase):
    def rows(self, runs):
        return dict(metric_rows(runs))

    def test_best_finished_run_gets_a_star(self):
        rows = self.rows([make_run("a", total_seconds=5.0, tokens_per_second=40.0),
                          make_run("b", total_seconds=3.0, tokens_per_second=25.0)])
        self.assertEqual(rows["Total time"], ["5.0 s", "3.0 s ★"])
        self.assertEqual(rows["Tokens/s"], ["40.0 ★", "25.0"])

    def test_ties_all_get_the_star(self):
        rows = self.rows([make_run("a"), make_run("b")])
        self.assertEqual(rows["Total time"], ["5.0 s ★", "5.0 s ★"])

    def test_no_star_when_only_one_run_finished(self):
        rows = self.rows([make_run("a"), ModelRun(model="b", status="error", error="boom")])
        self.assertNotIn("★", rows["Total time"][0])

    def test_failed_and_skipped_runs_show_dashes(self):
        rows = self.rows([make_run("a"), ModelRun(model="b", status="error", error="boom"),
                          ModelRun(model="c", status="skipped")])
        self.assertEqual(rows["Status"], ["ok", "error: boom", "skipped"])
        self.assertEqual(rows["Total time"][1:], ["-", "-"])
        self.assertEqual(rows["Load time"][1:], ["-", "-"])
        self.assertEqual(rows["Model memory"][1:], ["-", "-"])

    def test_model_memory_shows_gpu_share_and_flags_cpu_spill(self):
        rows = self.rows([make_run("a"), make_run("b", size_bytes=10_000_000_000, vram_bytes=6_000_000_000)])
        self.assertEqual(rows["Model memory"], ["4.0 GB · 100% GPU", "10.0 GB · 60% GPU ⚠"])

    def test_optional_rows_appear_only_with_data(self):
        plain = self.rows([make_run("a"), make_run("b")])
        for label in ("GPU memory used", "GPU peak load", "Judge score", "Warning"):
            self.assertNotIn(label, plain)
        rich = self.rows([make_run("a", gpu_extra_vram=5e9, gpu_peak_util=91.0, judge_score=8.0,
                                   warning="careful"),
                          make_run("b", gpu_extra_vram=3e9, gpu_peak_util=55.0, judge_score=5.0)])
        self.assertEqual(rich["GPU memory used"], ["5.0 GB", "3.0 GB"])
        self.assertEqual(rich["GPU peak load"], ["91%", "55%"])
        self.assertEqual(rich["Judge score"], ["8/10 ★", "5/10"])
        self.assertEqual(rich["Warning"], ["careful", "-"])

    def test_long_errors_are_shortened(self):
        row = self.rows([ModelRun(model="a", status="error", error="x" * 100)])["Status"][0]
        self.assertLess(len(row), 60)


class TestMarkdownAndRuns(unittest.TestCase):
    def comparison(self, **extra):
        data = {
            "created_at": "2026-10-16 12:00:00", "temperature": 0.7, "system": None,
            "judge_model": None, "winner": None, "prompt": "Why is the sky blue?\nBe brief.",
            "runs": [make_run("a:1", answer="Rayleigh scattering.").to_dict(),
                     make_run("b:1", answer="", status="error", error="boom").to_dict()],
        }
        data.update(extra)
        return data

    def test_markdown_has_prompt_table_and_answers(self):
        text = comparison_markdown(self.comparison())
        for expected in ("# Model comparison", "> Why is the sky blue?", "> Be brief.",
                         "| | a:1 | b:1 |", "| Total time |", "### a:1", "Rayleigh scattering.",
                         "*(no answer: boom)*"):
            self.assertIn(expected, text)

    def test_markdown_shows_winner_judge_and_system_prompt(self):
        runs = [make_run("a:1", judge_score=8.0, judge_reason="clear").to_dict(), make_run("b:1").to_dict()]
        text = comparison_markdown(self.comparison(winner="a:1", judge_model="j:1", system="be nice",
                                                   runs=runs))
        for expected in ("Your pick: **a:1**", "### a:1 (your pick)", "Judge: j:1",
                         "System prompt: be nice", "*Judge: 8/10 - clear*"):
            self.assertIn(expected, text)

    def test_pipes_in_cells_do_not_break_the_table(self):
        runs = [make_run("a:1", warning="x | y").to_dict()]
        self.assertIn("x \\| y", comparison_markdown(self.comparison(runs=runs)))

    def test_model_run_survives_a_round_trip_and_unknown_keys(self):
        run = make_run("a:1", judge_score=7.0)
        data = json.loads(json.dumps(run.to_dict()))
        data["field_from_the_future"] = 1
        self.assertEqual(ModelRun.from_dict(data), run)


if __name__ == "__main__":
    unittest.main()
