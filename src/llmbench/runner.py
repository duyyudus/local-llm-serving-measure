"""Run concurrency levels and turn raw requests into summaries."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from statistics import median, median_low
from typing import Any, Callable, Dict, List, Optional

from .client import ChatClient, RequestResult
from .prompts import PromptBuilder
from .stats import median_or_none, percentile

# A sustained slot that sees this many failures in a row stops launching
# requests instead of looping errors until the window deadline.
SLOT_MAX_CONSECUTIVE_FAILURES = 3
# Seconds of backoff before the retry after a failure; scales with the
# consecutive-failure count (0.25s, then 0.5s before the final attempt).
SLOT_FAILURE_BACKOFF_S = 0.25


@dataclass
class Wave:
    """One measured batch at a single concurrency level.

    In burst mode a wave is N requests fired together; in sustained mode it is
    a time window that keeps N requests in flight.
    """

    concurrency: int
    index: int
    warmup: bool
    requests: List[RequestResult]
    wave_start: float
    wave_end: float
    wall_s: float
    ok: int
    failed: int
    prompt_tokens: int
    completion_tokens: int
    output_tps: Optional[float]
    req_per_s: Optional[float]
    prefill_tps_aggregate: Optional[float]
    slots_stopped_early: int = 0
    prompt_tokens_in_window: Optional[float] = None
    completion_tokens_in_window: Optional[float] = None
    window_start: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "warmup": self.warmup,
            "wall_s": self.wall_s,
            "ok": self.ok,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "output_tps": self.output_tps,
            "req_per_s": self.req_per_s,
            "prefill_tps_aggregate": self.prefill_tps_aggregate,
            "slots_stopped_early": self.slots_stopped_early,
            "prompt_tokens_in_window": self.prompt_tokens_in_window,
            "completion_tokens_in_window": self.completion_tokens_in_window,
            # Relative to wave_start, matching the per-request
            # start_offset_s time base; the window opens at the last slot's
            # first launch, so this is the ramp-up length.
            "window_start_offset_s": (
                self.window_start - self.wave_start
                if self.window_start is not None
                else None
            ),
            "requests": [r.to_dict(self.wave_start) for r in self.requests],
        }


@dataclass
class LevelSummary:
    """Aggregate statistics for one concurrency level."""

    concurrency: int
    waves: int
    requests: int
    ok: int
    failed: int
    ttft_median_s: Optional[float]
    ttft_p90_s: Optional[float]
    prefill_tps_median: Optional[float]
    decode_tps_median: Optional[float]
    decode_tps_p10: Optional[float]
    prefill_tps_aggregate: Optional[float]
    output_tps_aggregate: Optional[float]
    requests_per_s: Optional[float]
    wall_s_median: Optional[float]
    prompt_tokens_total: int
    completion_tokens_total: int
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waves": self.waves,
            "requests": self.requests,
            "ok": self.ok,
            "failed": self.failed,
            "ttft_median_s": self.ttft_median_s,
            "ttft_p90_s": self.ttft_p90_s,
            "prefill_tps_median": self.prefill_tps_median,
            "decode_tps_median": self.decode_tps_median,
            "decode_tps_p10": self.decode_tps_p10,
            "prefill_tps_aggregate": self.prefill_tps_aggregate,
            "output_tps_aggregate": self.output_tps_aggregate,
            "requests_per_s": self.requests_per_s,
            "wall_s_median": self.wall_s_median,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "errors": self.errors,
        }


@dataclass
class LevelResult:
    """Everything measured at one concurrency level."""

    concurrency: int
    summary: LevelSummary
    waves: List[Wave]
    warmup_waves: List[Wave]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "summary": self.summary.to_dict(),
            "warmup_waves": [wave.to_dict() for wave in self.warmup_waves],
            "waves": [wave.to_dict() for wave in self.waves],
        }


@dataclass
class RunState:
    """Mutable run state so partial results survive failures and interrupts."""

    results: List[LevelResult] = field(default_factory=list)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    interrupted: bool = False


def _clip_ratio(
    start: float, end: float, window_start: float, window_end: float
) -> float:
    """Fraction of the interval [start, end] inside the measured window.

    Used to prorate the work a request produced outside a sustained window:
    tokens after the deadline (the drain tail) and tokens before the window
    opens (the staggered ramp-up) must not skew the window's rates.
    """
    if end <= start:
        return 1.0 if window_start <= start <= window_end else 0.0
    overlap = min(end, window_end) - max(start, window_start)
    return min(1.0, max(0.0, overlap / (end - start)))


def _build_wave(
    concurrency: int,
    index: int,
    warmup: bool,
    requests: List[RequestResult],
    *,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
) -> Wave:
    ok_requests = [request for request in requests if request.ok]
    prompt_tokens = sum(request.prompt_tokens or 0 for request in ok_requests)
    completion_tokens = sum(request.completion_tokens or 0 for request in ok_requests)

    wave_start = min((request.started_at for request in requests), default=time.perf_counter())
    wave_end = max((request.ended_at for request in requests), default=wave_start)

    prompt_in_window: Optional[float] = None
    completion_in_window: Optional[float] = None

    if window_end is not None:
        # Sustained window: measure rates inside the window and prorate work
        # outside it — the drain tail past the deadline and the staggered
        # ramp-up before the window opens — so neither skews the aggregates.
        if window_start is None:
            window_start = wave_start
        wall_s = max(window_end - window_start, 1e-9)
        prompt_in_window = 0.0
        completion_in_window = 0.0
        completed_in_window = 0
        for request in ok_requests:
            prompt_in_window += (
                _clip_ratio(
                    request.started_at,
                    request.first_token_at or request.ended_at,
                    window_start,
                    window_end,
                )
                * (request.prompt_tokens or 0)
            )
            completion_in_window += (
                _clip_ratio(
                    request.first_token_at or request.started_at,
                    request.last_token_at or request.ended_at,
                    window_start,
                    window_end,
                )
                * (request.completion_tokens or 0)
            )
            if window_start <= request.ended_at <= window_end:
                completed_in_window += 1

        output_tps = (
            completion_in_window / wall_s if completion_in_window > 0 else None
        )
        req_per_s = completed_in_window / wall_s if completed_in_window else None
        prefill_tps_aggregate = (
            prompt_in_window / wall_s if prompt_in_window > 0 else None
        )
    else:
        wall_s = max(wave_end - wave_start, 1e-9)

        output_tps = completion_tokens / wall_s if completion_tokens else None
        req_per_s = len(ok_requests) / wall_s if ok_requests else None

        first_tokens = [
            request.first_token_at
            for request in ok_requests
            if request.first_token_at is not None
        ]
        prefill_tps_aggregate = None
        if first_tokens and prompt_tokens:
            first_start = min(request.started_at for request in ok_requests)
            prefill_wall = max(first_tokens) - first_start
            if prefill_wall > 0:
                prefill_tps_aggregate = prompt_tokens / prefill_wall

    return Wave(
        concurrency=concurrency,
        index=index,
        warmup=warmup,
        requests=requests,
        wave_start=wave_start,
        wave_end=wave_end,
        wall_s=wall_s,
        ok=len(ok_requests),
        failed=len(requests) - len(ok_requests),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        output_tps=output_tps,
        req_per_s=req_per_s,
        prefill_tps_aggregate=prefill_tps_aggregate,
        prompt_tokens_in_window=prompt_in_window,
        completion_tokens_in_window=completion_in_window,
        window_start=window_start if window_end is not None else None,
    )


async def run_wave(
    client: ChatClient,
    builder: PromptBuilder,
    concurrency: int,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    index: int,
    warmup: bool,
) -> Wave:
    """Fire ``concurrency`` requests at once and collect their results."""
    prompts = [builder.build() for _ in range(concurrency)]
    tasks = [
        asyncio.create_task(client.stream(prompt, max_tokens, temperature, seed))
        for prompt in prompts
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    now = time.perf_counter()
    results: List[RequestResult] = []
    for item in raw_results:
        if isinstance(item, BaseException):
            results.append(
                RequestResult.error_result(
                    now, now, f"{type(item).__name__}: {item}"
                )
            )
        else:
            results.append(item)

    return _build_wave(concurrency, index, warmup, results)


async def run_sustained_window(
    client: ChatClient,
    builder: PromptBuilder,
    concurrency: int,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    index: int,
    duration: float,
    stagger: float = 0.0,
    on_request: Optional[Callable[[int, int, int, float], None]] = None,
) -> Wave:
    """Keep ``concurrency`` requests in flight for ``duration`` seconds.

    Every slot launches a new request as soon as its previous one finishes,
    until the deadline passes. Requests already in flight at the deadline are
    allowed to complete, but the wave's rates only count work done inside the
    window: tokens produced after the deadline (the drain tail) and before
    the window opens (the staggered ramp-up) are prorated by each request's
    token rate, so neither skews the aggregate columns. The window itself
    opens when the last slot launches and lasts ``duration`` seconds, so a
    stretched ramp costs wall-clock time but not measurement quality.

    ``stagger`` spreads the slots' first launches: slot ``i`` waits
    ``i * stagger`` seconds. With identical request shapes a batching server
    would otherwise re-launch all slots in lockstep, turning sustained mode
    into back-to-back bursts.

    A slot that sees ``SLOT_MAX_CONSECUTIVE_FAILURES`` failures in a row stops
    launching requests, with a short backoff between failed attempts, instead
    of looping errors until the deadline. The returned wave's
    ``slots_stopped_early`` reports how many slots stopped this way.

    ``on_request`` is called after each finished request with
    ``(concurrency, done, failed, elapsed_s)`` so long windows can report
    progress.
    """
    results: List[RequestResult] = []
    started = time.perf_counter()
    # The measured window opens when the last slot launches, so the staggered
    # ramp-up — which a cold first warmup can stretch — is not counted in the
    # rates. The cap on ``stagger`` keeps the ramp shorter than the window.
    ramp = (concurrency - 1) * stagger if stagger > 0 else 0.0
    deadline = started + ramp + duration
    first_launches: List[Optional[float]] = [None] * concurrency
    failed_count = 0
    stopped_early = 0

    async def slot(slot_index: int) -> None:
        nonlocal failed_count, stopped_early
        if stagger > 0 and slot_index > 0:
            await asyncio.sleep(slot_index * stagger)
        consecutive_failures = 0
        while time.perf_counter() < deadline:
            prompt = builder.build()
            if first_launches[slot_index] is None:
                first_launches[slot_index] = time.perf_counter()
            result = await client.stream(prompt, max_tokens, temperature, seed)
            results.append(result)
            if result.ok:
                consecutive_failures = 0
            else:
                failed_count += 1
                consecutive_failures += 1
            if on_request is not None:
                on_request(
                    concurrency,
                    len(results),
                    failed_count,
                    time.perf_counter() - started,
                )
            if not result.ok:
                if consecutive_failures >= SLOT_MAX_CONSECUTIVE_FAILURES:
                    stopped_early += 1
                    break
                await asyncio.sleep(SLOT_FAILURE_BACKOFF_S * consecutive_failures)

    # return_exceptions keeps one broken slot from killing the others; a slot
    # that raises simply stops launching new requests.
    tasks = [
        asyncio.create_task(slot(slot_index)) for slot_index in range(concurrency)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    now = time.perf_counter()
    for item in raw_results:
        if isinstance(item, BaseException):
            results.append(
                RequestResult.error_result(now, now, f"{type(item).__name__}: {item}")
            )

    # The window opens when the last slot actually launches its first request.
    window_start = max(
        (launch for launch in first_launches if launch is not None),
        default=started,
    )
    wave = _build_wave(
        concurrency,
        index,
        False,
        results,
        window_start=window_start,
        window_end=deadline,
    )
    wave.slots_stopped_early = stopped_early
    return wave


def summarize_level(concurrency: int, waves: List[Wave]) -> LevelSummary:
    """Combine the measured waves of one concurrency level."""
    requests = [request for wave in waves for request in wave.requests]
    ok_requests = [request for request in requests if request.ok]

    # Per-request columns reflect the measured window only: requests that
    # started during a sustained window's ramp-up ran with fewer than N
    # streams in flight, so they decode faster and queue less than the
    # steady-state samples the table describes.
    windowed = [
        request
        for wave in waves
        for request in wave.requests
        if request.ok
        and (wave.window_start is None or request.started_at >= wave.window_start)
    ]
    if not windowed:
        windowed = ok_requests

    ttfts = [
        request.ttft_s
        for request in windowed
        if request.ttft_s is not None
    ]
    prefills = [
        request.prefill_tps
        for request in windowed
        if request.prefill_tps is not None
    ]
    decodes = [
        request.decode_tps
        for request in windowed
        if request.decode_tps is not None
    ]

    prefill_aggregate = [
        wave.prefill_tps_aggregate
        for wave in waves
        if wave.prefill_tps_aggregate is not None
    ]
    output_aggregate = [
        wave.output_tps for wave in waves if wave.output_tps is not None
    ]
    request_rates = [
        wave.req_per_s for wave in waves if wave.req_per_s is not None
    ]

    errors: List[str] = []
    for request in requests:
        if not request.ok and request.error and request.error not in errors:
            errors.append(request.error)

    return LevelSummary(
        concurrency=concurrency,
        waves=len(waves),
        requests=len(requests),
        ok=len(ok_requests),
        failed=len(requests) - len(ok_requests),
        ttft_median_s=median_or_none(ttfts),
        ttft_p90_s=percentile(ttfts, 90),
        prefill_tps_median=median_or_none(prefills),
        decode_tps_median=median_or_none(decodes),
        decode_tps_p10=percentile(decodes, 10),
        prefill_tps_aggregate=median_or_none(prefill_aggregate),
        output_tps_aggregate=median_or_none(output_aggregate),
        requests_per_s=median_or_none(request_rates),
        wall_s_median=median_or_none([wave.wall_s for wave in waves]),
        prompt_tokens_total=sum(wave.prompt_tokens for wave in waves),
        completion_tokens_total=sum(wave.completion_tokens for wave in waves),
        errors=errors[:5],
    )


def _first_error(waves: List[Wave]) -> str:
    for wave in waves:
        for request in wave.requests:
            if request.error:
                return request.error
    return "unknown error"


def ok_per_wave(waves: List[Wave]) -> int:
    """Median-low of per-wave ok counts; the conservative ok/wave figure."""
    return int(median_low([wave.ok for wave in waves])) if waves else 0


def failure_warning(
    concurrency: int,
    summary: LevelSummary,
    waves: List[Wave],
    pattern: str = "burst",
) -> str:
    """Single source for the failure warning shown and exported."""
    if summary.ok == 0:
        return (
            f"users={concurrency}: all {summary.requests} requests failed; "
            "no metrics for this level."
        )
    if pattern == "sustained":
        # ok per window counts completions, not streams, so it must not be
        # presented as the number of concurrent streams.
        return (
            f"users={concurrency}: only {summary.ok}/{summary.requests} "
            "requests succeeded; failed requests are excluded from the "
            "statistics."
        )
    ok_wave = ok_per_wave(waves)
    streams = "stream" if ok_wave == 1 else "streams"
    return (
        f"users={concurrency}: only {summary.ok}/{summary.requests} requests "
        f"succeeded; median {ok_wave} ok/wave, so metrics reflect about "
        f"{ok_wave} concurrent {streams}."
    )


async def run_benchmark(
    client: ChatClient,
    builder: PromptBuilder,
    levels: List[int],
    *,
    state: RunState,
    waves: int,
    warmup_waves: int,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    continue_on_error: bool = False,
    pattern: str = "burst",
    duration: float = 30.0,
    on_wave: Optional[Callable[[Wave], None]] = None,
    on_request: Optional[Callable[[int, int, int, float], None]] = None,
) -> RunState:
    """Run every concurrency level, appending results to ``state``.

    ``pattern`` selects the load pattern. ``burst`` fires each wave as N
    simultaneous requests and waits for the slowest. ``sustained`` measures
    each wave as a window of ``duration`` seconds that keeps N requests in
    flight, which is closer to real serving. Warmup waves always run as
    bursts.
    """
    if pattern not in ("burst", "sustained"):
        raise ValueError(f"unknown load pattern: {pattern!r}")

    for level in levels:
        warmups: List[Wave] = []
        for index in range(warmup_waves):
            wave = await run_wave(
                client,
                builder,
                level,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                index=index,
                warmup=True,
            )
            warmups.append(wave)
            if on_wave is not None:
                on_wave(wave)
            if wave.ok == 0 and not continue_on_error:
                state.error = (
                    f"all {level} warmup requests failed at users={level}: "
                    f"{_first_error([wave])}"
                )
                return state

        measured: List[Wave] = []
        # Spread the sustained slots' first launches by one request time over
        # the level, estimated from the burst warmup waves. Without an
        # estimate (no warmup) slots launch together.
        stagger = 0.0
        if pattern == "sustained" and warmups:
            warm_wall = median([wave.wall_s for wave in warmups])
            stagger = min(warm_wall, duration) / level
        for index in range(waves):
            if pattern == "sustained":
                wave = await run_sustained_window(
                    client,
                    builder,
                    level,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    index=index,
                    duration=duration,
                    stagger=stagger,
                    on_request=on_request,
                )
            else:
                wave = await run_wave(
                    client,
                    builder,
                    level,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    index=index,
                    warmup=False,
                )
            measured.append(wave)
            if on_wave is not None:
                on_wave(wave)

        summary = summarize_level(level, measured)
        state.results.append(
            LevelResult(
                concurrency=level,
                summary=summary,
                waves=measured,
                warmup_waves=warmups,
            )
        )

        if summary.ok == 0 and not continue_on_error:
            state.error = (
                f"all {level} measured requests failed at users={level}: "
                f"{_first_error(measured)}"
            )
            return state

        if summary.failed:
            state.warnings.append(
                failure_warning(level, summary, measured, pattern=pattern)
            )

        stopped = max(
            (wave.slots_stopped_early for wave in measured), default=0
        )
        if stopped:
            streams = "stream" if level == 1 else "streams"
            state.warnings.append(
                f"users={level}: {stopped} of {level} {streams} stopped "
                "early after repeated failures; fewer concurrent streams "
                "than requested."
            )

    return state
