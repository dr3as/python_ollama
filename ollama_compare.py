"""Compare several models on the same prompt, one after another, with timings and memory use.

This module has no user interface. ollama_tui.py shows the results (the Compare view), but the
logic lives here so it can be tested on its own and reused, for example from a command line tool.

How a comparison works, step by step (see compare_models):

1. For EACH model, in turn:
   a. Unload every model that is loaded, and wait until Ollama confirms the memory is free.
      Without this the second model would be loaded next to the first one. On a small GPU that
      pushes part of it onto the CPU, and the speed numbers would be wrong.
   b. Send the prompt, streaming the reply, and time it (first token, total, tokens/s).
   c. While the model is still loaded, ask Ollama how it is split between GPU and CPU memory.
      This has to happen BEFORE unloading it, because afterwards there is nothing to ask about.
   d. Unload the model again.
2. Optionally, a "judge" model reads all answers (in random order, without model names) and
   gives each a score. That runs last, on its own, for the same memory reason.

Everything runs sequentially on purpose: speed and memory numbers are only comparable when each
model had the whole machine to itself.
"""
import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ollama_client import OllamaClient, chat_stats

UNLOAD_TIMEOUT = 20.0  # seconds to wait for Ollama to confirm a model is gone
GPU_SAMPLE_INTERVAL = 0.5  # seconds between nvidia-smi readings during a run


def format_bytes(size: float) -> str:
    """5_200_000_000 -> "5.2 GB"; small sizes are shown in MB."""
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{size / 1e6:.0f} MB"


# ---- one model's result ---------------------------------------------------------------------------


@dataclass
class ModelRun:
    """Everything we learned about one model answering the prompt.

    A dataclass is a class whose __init__, __repr__ and comparison are generated from the field
    list below. Every field has a default, so a run can start empty and be filled in as it goes.
    """

    model: str
    # waiting -> ok | error | stopped (user pressed Esc during this model) | skipped (never started)
    status: str = "waiting"
    answer: str = ""
    error: str = ""
    warning: str = ""  # e.g. another model was still loaded: the numbers may be skewed
    first_token_seconds: Optional[float] = None  # request sent -> first text; includes loading
    total_seconds: Optional[float] = None
    tokens_per_second: float = 0.0  # speed of the generation itself (excludes loading)
    prompt_tokens: int = 0
    reply_tokens: int = 0
    load_seconds: float = 0.0  # time Ollama needed to load the model (a cold start, see above)
    size_bytes: int = 0  # memory the loaded model takes in total
    vram_bytes: int = 0  # ... of which in GPU memory (the rest runs on the CPU, which is slow)
    context_length: Optional[int] = None
    gpu_extra_vram: Optional[float] = None  # peak GPU memory during the run minus before it
    gpu_peak_util: Optional[float] = None  # highest GPU load in percent
    judge_score: Optional[float] = None
    judge_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRun":
        # Ignore unknown keys, so data saved by a newer version of the app still loads.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---- unloading and verifying that memory is free -----------------------------------------------


def wait_until_unloaded(client: OllamaClient, timeout: float = UNLOAD_TIMEOUT,
                        poll: float = 0.25) -> bool:
    """Poll Ollama until no model is loaded. False if some are still loaded after `timeout`.

    The unload request returns before the memory is actually freed, so asking /api/ps is the only
    reliable way to know the next model really starts from an empty GPU.
    """
    deadline = time.monotonic() + timeout
    while True:
        if not client.running_models():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


def unload_all(client: OllamaClient, timeout: float = UNLOAD_TIMEOUT) -> bool:
    """Unload every loaded model and wait for confirmation. True if the server is now empty."""
    for loaded in client.running_models():
        client.unload_model(loaded["name"])
    return wait_until_unloaded(client, timeout)


class GpuSampler:
    """Reads the GPU every half second in a background thread and remembers the peak.

    `reader` is a function returning a list of GPU dicts (see ollama_tui.read_gpus) or None. It
    is passed in rather than imported so this module stays independent of the UI and tests can
    supply fake readings.
    """

    def __init__(self, reader: Callable[[], Optional[List[Dict[str, Any]]]],
                 interval: float = GPU_SAMPLE_INTERVAL):
        self.reader = reader
        self.interval = interval
        self.baseline_vram: Optional[float] = None  # GPU memory in use before the model loads
        self.peak_vram: Optional[float] = None
        self.peak_util: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        gpus = self.reader()
        if not gpus:
            return
        # Ollama may spread a model over several GPUs, so add their memory up. For the load
        # the busiest GPU counts.
        used = sum(g["vram_used"] or 0 for g in gpus)
        util = max((g["util_percent"] or 0 for g in gpus), default=0)
        if self.baseline_vram is None:
            self.baseline_vram = used
        self.peak_vram = used if self.peak_vram is None else max(self.peak_vram, used)
        self.peak_util = util if self.peak_util is None else max(self.peak_util, util)

    def _loop(self) -> None:
        # Event.wait(timeout) sleeps like time.sleep, but returns at once when stop() is called.
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._sample()  # the baseline: taken before the model is loaded
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()  # one last reading, so a short run still has a peak

    @property
    def extra_vram(self) -> Optional[float]:
        """Peak GPU memory minus the baseline: roughly what the model itself needed."""
        if self.baseline_vram is None or self.peak_vram is None:
            return None
        return max(0.0, self.peak_vram - self.baseline_vram)


# ---- running the comparison ---------------------------------------------------------------------


def run_model(client: OllamaClient, model: str, prompt: str, system: Optional[str] = None,
              temperature: float = 0.7, cancel: Optional[threading.Event] = None,
              gpu_reader: Optional[Callable] = None,
              on_status: Optional[Callable[[str, str], None]] = None,
              on_text: Optional[Callable[[str, str, int, float], None]] = None) -> ModelRun:
    """Run the prompt on one model from a clean start and return its ModelRun.

    on_status(model, text) reports phases; on_text(model, text_so_far, chunks, seconds) is called
    for every streamed chunk. Both run in the calling thread, so a UI must hand the data over to
    its own thread (ollama_tui does this with call_from_thread).
    """
    cancel = cancel or threading.Event()
    status = on_status or (lambda model, text: None)
    run = ModelRun(model=model)

    status(model, "unloading other models...")
    try:
        if not unload_all(client, UNLOAD_TIMEOUT):  # passed explicitly so tests can shorten it
            # Do not abort: tell the user their numbers may be off and carry on.
            run.warning = "another model was still loaded when this one started"
    except Exception as e:
        run.warning = f"could not check that memory was free: {e}"

    sampler = GpuSampler(gpu_reader) if gpu_reader else None
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    parts: List[str] = []
    final_chunk = None
    status(model, "loading and answering...")
    try:
        if sampler:
            sampler.start()
        started = time.monotonic()  # monotonic: a clock that never jumps backwards
        stream = client.stream_chat(messages, model=model, temperature=temperature)
        for chunk in stream:
            if chunk.done:
                final_chunk = chunk
            content = chunk.message.content
            if content:
                if run.first_token_seconds is None:
                    run.first_token_seconds = time.monotonic() - started
                    status(model, "generating...")
                parts.append(content)
                if on_text:
                    on_text(model, "".join(parts), len(parts), time.monotonic() - started)
            if cancel.is_set():
                stream.close()
                run.status = "stopped"
                break
        else:
            run.status = "ok"  # the for loop ended without `break`: the stream completed
        run.total_seconds = time.monotonic() - started
    except Exception as e:
        run.status = "error"
        run.error = str(e) or type(e).__name__
    finally:
        if sampler:
            sampler.stop()
            run.gpu_extra_vram = sampler.extra_vram
            run.gpu_peak_util = sampler.peak_util
    run.answer = "".join(parts)

    if final_chunk is not None:
        stats = chat_stats(final_chunk)
        run.tokens_per_second = stats["tokens_per_second"]
        run.prompt_tokens = stats["prompt_tokens"]
        run.reply_tokens = stats["reply_tokens"]
        run.load_seconds = stats["load_seconds"]

    # Ask how the model is split between GPU and CPU while it is still loaded (see module docs).
    if run.status in ("ok", "stopped"):
        try:
            loaded = {m["name"]: m for m in client.running_models()}
            if model in loaded:
                run.size_bytes = loaded[model]["size"]
                run.vram_bytes = loaded[model]["size_vram"]
                run.context_length = loaded[model]["context_length"]
            others = sorted(set(loaded) - {model})
            if others and not run.warning:
                run.warning = f"other models were loaded during the run: {', '.join(others)}"
        except Exception:
            pass  # the memory split is a bonus; the speed numbers are still valid

    try:
        client.unload_model(model)
    except Exception:
        pass  # the next run (or unload_all at its start) deals with a model that stays loaded
    return run


def compare_models(client: OllamaClient, prompt: str, models: Sequence[str],
                   system: Optional[str] = None, temperature: float = 0.7,
                   cancel: Optional[threading.Event] = None,
                   gpu_reader: Optional[Callable] = None, judge_model: Optional[str] = None,
                   on_status: Optional[Callable[[str, str], None]] = None,
                   on_text: Optional[Callable[[str, str, int, float], None]] = None,
                   on_run: Optional[Callable[[ModelRun], None]] = None,
                   on_judging: Optional[Callable[[str], None]] = None,
                   ) -> Tuple[List[ModelRun], Optional[str]]:
    """Run the prompt on every model in turn. Returns (runs, judge_problem).

    A failing model does not stop the others. If `cancel` is set, the current model stops, the
    rest are marked "skipped" and no judging happens. judge_problem is None when judging worked
    or was not requested, otherwise a short explanation.
    """
    cancel = cancel or threading.Event()
    runs: List[ModelRun] = []
    for model in models:
        if cancel.is_set():
            run = ModelRun(model=model, status="skipped")
        else:
            run = run_model(client, model, prompt, system, temperature, cancel, gpu_reader,
                            on_status, on_text)
        runs.append(run)
        if on_run:
            on_run(run)

    judge_problem = None
    if judge_model and not cancel.is_set():
        if on_judging:
            on_judging(judge_model)
        judge_problem = judge_runs(client, judge_model, prompt, runs)
    return runs, judge_problem


# ---- the judge model ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_judge_prompt(prompt: str, answers: Sequence[str]) -> str:
    """The instructions sent to the judge. Answers are labelled A, B, C ... (no model names)."""
    blocks = "\n\n".join(f"ANSWER {LETTERS[i]}:\n{text}" for i, text in enumerate(answers))
    example = ", ".join(f'"{LETTERS[i]}": {{"score": 7, "reason": "one short sentence"}}'
                        for i in range(len(answers)))
    return (
        "You are judging answers from different AI assistants to the same request.\n\n"
        f"REQUEST:\n{prompt}\n\n{blocks}\n\n"
        "Rate each answer from 1 (poor) to 10 (excellent) for correctness, helpfulness and "
        "clarity. Reply with JSON only, in exactly this form:\n"
        f"{{{example}}}"
    )


def parse_judge_reply(text: str, count: int) -> Dict[int, Tuple[float, str]]:
    """Read the judge's JSON into {answer index: (score 1-10, reason)}.

    Small models often wrap JSON in prose or code fences, or skip an answer, so this is
    forgiving: it takes the first {...} block and ignores entries it cannot use.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)  # greedy: from the first { to the last }
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    scores: Dict[int, Tuple[float, str]] = {}
    for index in range(count):
        entry = data.get(LETTERS[index]) if isinstance(data, dict) else None
        if isinstance(entry, dict):
            score, reason = entry.get("score"), entry.get("reason", "")
        else:
            score, reason = entry, ""  # the judge may answer {"A": 7} without a reason
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        scores[index] = (min(10.0, max(1.0, float(score))), str(reason))
    return scores


def judge_runs(client: OllamaClient, judge_model: str, prompt: str, runs: List[ModelRun],
               order: Optional[Sequence[int]] = None) -> Optional[str]:
    """Let `judge_model` score the answers and store the result in each run. Returns a problem or None.

    Only runs that produced an answer are judged. They are shown in a random order (or the one
    given in `order`, a list of indexes into `runs`) so the judge cannot favour whichever answer
    happens to come first. Note: a judge that is also one of the compared models tends to prefer
    its own style, so a separate model is the fairer choice.
    """
    candidates = [i for i, r in enumerate(runs) if r.answer.strip() and r.status in ("ok", "stopped")]
    if len(candidates) < 2:
        return "need at least two answers to judge"
    shown = list(order) if order is not None else random.sample(candidates, len(candidates))
    try:
        unload_all(client, UNLOAD_TIMEOUT)  # the judge gets the whole GPU too
        reply = client.chat_once(
            [{"role": "user", "content": build_judge_prompt(prompt, [runs[i].answer for i in shown])}],
            model=judge_model, temperature=0.0, json_format=True,
        )
    except Exception as e:
        return f"judge failed: {e}"
    finally:
        try:
            client.unload_model(judge_model)
        except Exception:
            pass
    scores = parse_judge_reply(reply, len(shown))
    if not scores:
        return "the judge's reply could not be read"
    for position, (score, reason) in scores.items():
        runs[shown[position]].judge_score = score
        runs[shown[position]].judge_reason = reason
    return None


# ---- presenting results ------------------------------------------------------------------------


def _seconds(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f} s"


def _memory_cell(run: ModelRun) -> str:
    if not run.size_bytes:
        return "-"
    if run.vram_bytes >= run.size_bytes:
        return f"{format_bytes(run.size_bytes)} · 100% GPU"
    gpu = run.vram_bytes * 100 // run.size_bytes
    return f"{format_bytes(run.size_bytes)} · {gpu}% GPU ⚠"  # ⚠: part runs on the slow CPU


def _best(runs: Sequence[ModelRun], value: Callable[[ModelRun], Optional[float]],
          lowest: bool) -> set:
    """Indexes of the runs with the best value among finished runs (ties all count)."""
    scored = [(i, value(r)) for i, r in enumerate(runs) if r.status == "ok" and value(r)]
    if len(scored) < 2:  # a "best" of one is meaningless
        return set()
    target = (min if lowest else max)(v for _, v in scored)
    return {i for i, v in scored if v == target}


def metric_rows(runs: Sequence[ModelRun]) -> List[Tuple[str, List[str]]]:
    """The comparison table as (row label, one cell per run). ★ marks the best finished run."""
    fastest_first = _best(runs, lambda r: r.first_token_seconds, lowest=True)
    fastest_total = _best(runs, lambda r: r.total_seconds, lowest=True)
    fastest_speed = _best(runs, lambda r: r.tokens_per_second, lowest=False)
    top_score = _best(runs, lambda r: r.judge_score, lowest=False)

    def row(label: str, cell: Callable[[ModelRun], str], best: set = frozenset()):
        return label, [cell(r) + (" ★" if i in best else "") for i, r in enumerate(runs)]

    rows = [
        row("Status", lambda r: r.status if not r.error else f"error: {r.error[:40]}"),
        row("First token", lambda r: _seconds(r.first_token_seconds), fastest_first),
        row("Total time", lambda r: _seconds(r.total_seconds), fastest_total),
        row("Tokens/s", lambda r: f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "-",
            fastest_speed),
        row("Load time", lambda r: _seconds(r.load_seconds) if r.status in ("ok", "stopped") else "-"),
        row("Prompt tokens", lambda r: f"{r.prompt_tokens:,}" if r.prompt_tokens else "-"),
        row("Reply tokens", lambda r: f"{r.reply_tokens:,}" if r.reply_tokens else "-"),
        row("Model memory", _memory_cell),
        row("Context", lambda r: f"{r.context_length:,}" if r.context_length else "-"),
    ]
    if any(r.gpu_extra_vram is not None for r in runs):
        rows.append(row("GPU memory used", lambda r: "-" if r.gpu_extra_vram is None
                        else format_bytes(r.gpu_extra_vram)))
        rows.append(row("GPU peak load", lambda r: "-" if r.gpu_peak_util is None
                        else f"{r.gpu_peak_util:.0f}%"))
    if any(r.judge_score is not None for r in runs):
        rows.append(row("Judge score", lambda r: "-" if r.judge_score is None
                        else f"{r.judge_score:.0f}/10", top_score))
    if any(r.warning for r in runs):
        rows.append(row("Warning", lambda r: r.warning or "-"))
    return rows


def comparison_markdown(comparison: Dict[str, Any]) -> str:
    """Render a saved comparison (ChatStore.get_comparison) as Markdown for notes or docs."""
    runs = [ModelRun.from_dict(r) for r in comparison["runs"]]
    lines = [
        "# Model comparison",
        "",
        f"- Date: {comparison['created_at']} UTC",
        f"- Temperature: {comparison['temperature']:g}",
    ]
    if comparison.get("system"):
        lines.append(f"- System prompt: {comparison['system']}")
    if comparison.get("judge_model"):
        lines.append(f"- Judge: {comparison['judge_model']}")
    if comparison.get("winner"):
        lines.append(f"- Your pick: **{comparison['winner']}**")
    lines += ["", "## Prompt", ""] + [f"> {line}" for line in comparison["prompt"].split("\n")]
    lines += ["", "## Metrics", "", "| | " + " | ".join(r.model for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for label, cells in metric_rows(runs):
        lines.append(f"| {label} | " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines += ["", "(★ = best of the finished runs; ⚠ = part of the model ran on the CPU)", "",
              "## Answers"]
    for run in runs:
        pick = " (your pick)" if run.model == comparison.get("winner") else ""
        lines += ["", f"### {run.model}{pick}", ""]
        if run.judge_score is not None:
            lines += [f"*Judge: {run.judge_score:.0f}/10 - {run.judge_reason}*", ""]
        lines.append(run.answer or f"*(no answer: {run.error or run.status})*")
    return "\n".join(lines) + "\n"
