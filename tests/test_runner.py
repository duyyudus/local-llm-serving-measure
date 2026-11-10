import asyncio
import time

import pytest

from llmbench.client import RequestResult
from llmbench.prompts import Prompt
from llmbench.runner import RunState, run_benchmark


class StubClient:
    def __init__(self, fail_after=None, fail_every=None):
        self.fail_after = fail_after
        self.fail_every = fail_every
        self.calls = 0

    async def stream(self, prompt, max_tokens, temperature=0.0, seed=None):
        self.calls += 1
        now = time.perf_counter()
        should_fail = False
        if self.fail_after is not None and self.calls > self.fail_after:
            should_fail = True
        if self.fail_every is not None and self.calls % self.fail_every == 0:
            should_fail = True
        if should_fail:
            return RequestResult.error_result(now, now + 0.001, "boom")
        return RequestResult(
            ok=True,
            error=None,
            started_at=now,
            first_token_at=now + 0.01,
            last_token_at=now + 0.02,
            ended_at=now + 0.03,
            ttft_s=0.01,
            decode_s=0.01,
            total_s=0.03,
            prompt_tokens=10,
            completion_tokens=2,
            prefill_tps=1000.0,
            decode_tps=100.0,
            usage_reported=True,
            streamed=True,
            text_chars=10,
        )


class StubBuilder:
    def build(self):
        return Prompt("hello", 10)


def test_run_benchmark_keeps_partial_results_when_a_level_fails():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(fail_after=1),
            StubBuilder(),
            [1, 2],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
        )
        return state

    state = asyncio.run(run())

    assert len(state.results) == 2
    assert state.results[0].concurrency == 1
    assert state.results[0].summary.ok == 1
    assert state.results[0].summary.prefill_tps_aggregate == pytest.approx(1000.0)
    assert state.results[1].concurrency == 2
    assert state.results[1].summary.failed == 2
    assert state.error is not None
    assert "users=2" in state.error


def test_run_benchmark_records_failed_level_in_results():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(fail_after=0),
            StubBuilder(),
            [1, 2],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
        )
        return state

    state = asyncio.run(run())

    assert len(state.results) == 1
    assert state.results[0].concurrency == 1
    assert state.results[0].summary.failed == 1
    assert state.error is not None


def test_run_benchmark_warns_when_a_level_partially_fails():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(fail_after=2),
            StubBuilder(),
            [1, 2],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
        )
        return state

    state = asyncio.run(run())

    assert len(state.results) == 2
    assert state.results[1].concurrency == 2
    assert state.results[1].summary.ok == 1
    assert state.results[1].summary.failed == 1
    assert state.error is None
    assert any("users=2" in warning and "1/2" in warning for warning in state.warnings)


def test_partial_failure_warning_uses_per_wave_successes():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(fail_every=2),
            StubBuilder(),
            [2],
            state=state,
            waves=3,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
        )
        return state

    state = asyncio.run(run())

    summary = state.results[0].summary
    assert summary.ok == 3
    assert summary.requests == 6
    assert summary.failed == 3
    assert len(state.warnings) == 1
    assert "3/6 requests succeeded" in state.warnings[0]
    assert "median 1 ok/wave" in state.warnings[0]
    assert "about 1 concurrent stream" in state.warnings[0]


def test_all_failed_level_warns_without_zero_stream_wording():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(fail_after=0),
            StubBuilder(),
            [2],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            continue_on_error=True,
        )
        return state

    state = asyncio.run(run())

    assert len(state.warnings) == 1
    assert "users=2: all 2 requests failed" in state.warnings[0]
    assert "0 concurrent" not in state.warnings[0]


def test_warmup_failure_stops_without_adding_level():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(fail_after=0),
            StubBuilder(),
            [1, 2],
            state=state,
            waves=1,
            warmup_waves=1,
            max_tokens=2,
            temperature=0.0,
            seed=None,
        )
        return state

    state = asyncio.run(run())

    assert state.results == []
    assert state.error is not None
    assert "warmup" in state.error
