"""Run concurrency levels and turn raw requests into summaries."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from statistics import median_low
from typing import Any, Callable, Dict, List, Optional

from .client import ChatClient, RequestResult
from .prompts import PromptBuilder
from .stats import median_or_none, percentile


@dataclass
class Wave:
    """One batch of concurrent requests at a single concurrency level."""

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


def _build_wave(
    concurrency: int,
    index: int,
    warmup: bool,
    requests: List[RequestResult],
) -> Wave:
    ok_requests = [request for request in requests if request.ok]
    prompt_tokens = sum(request.prompt_tokens or 0 for request in ok_requests)
    completion_tokens = sum(request.completion_tokens or 0 for request in ok_requests)

    wave_start = min((request.started_at for request in requests), default=time.perf_counter())
    wave_end = max((request.ended_at for request in requests), default=wave_start)
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


def summarize_level(concurrency: int, waves: List[Wave]) -> LevelSummary:
    """Combine the measured waves of one concurrency level."""
    requests = [request for wave in waves for request in wave.requests]
    ok_requests = [request for request in requests if request.ok]

    ttfts = [
        request.ttft_s
        for request in ok_requests
        if request.ttft_s is not None
    ]
    prefills = [
        request.prefill_tps
        for request in ok_requests
        if request.prefill_tps is not None
    ]
    decodes = [
        request.decode_tps
        for request in ok_requests
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
    on_wave: Optional[Callable[[Wave], None]] = None,
) -> RunState:
    """Run every concurrency level, appending results to ``state``."""
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
        for index in range(waves):
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
            ok_per_wave = int(median_low([wave.ok for wave in measured]))
            streams = "stream" if ok_per_wave == 1 else "streams"
            state.warnings.append(
                f"users={level}: only {summary.ok}/{summary.requests} requests "
                f"succeeded; median {ok_per_wave} ok/wave, so metrics reflect "
                f"about {ok_per_wave} concurrent {streams}."
            )

    return state
