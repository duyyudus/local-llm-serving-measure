import asyncio
import time

import pytest

from llmbench import runner
from llmbench.client import RequestResult
from llmbench.prompts import Prompt
from llmbench.runner import (
    SLOT_FAILURE_BACKOFF_S,
    SLOT_MAX_CONSECUTIVE_FAILURES,
    RunState,
    _build_wave,
    run_benchmark,
    run_sustained_window,
    summarize_level,
)


def _ok_result(now):
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
        return _ok_result(now)


class SlowStubClient(StubClient):
    """Stub client whose requests take real time and track overlap."""

    def __init__(self, request_s=0.02, fail_every=None, fail_first=None):
        super().__init__(fail_every=fail_every)
        self.request_s = request_s
        self.fail_first = fail_first
        self.in_flight = 0
        self.max_in_flight = 0
        self.launch_times = []

    async def stream(self, prompt, max_tokens, temperature=0.0, seed=None):
        self.calls += 1
        call_index = self.calls
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.launch_times.append(time.perf_counter())
        start = self.launch_times[-1]
        try:
            await asyncio.sleep(self.request_s)
            now = time.perf_counter()
            # Decide from the index captured at entry: under concurrency the
            # shared counter has usually moved on by the time we wake up.
            fail_every = (
                self.fail_every is not None and call_index % self.fail_every == 0
            )
            fail_first = (
                self.fail_first is not None and call_index <= self.fail_first
            )
            if fail_every or fail_first:
                return RequestResult.error_result(start, now, "boom")
            return RequestResult(
                ok=True,
                error=None,
                started_at=start,
                first_token_at=now,
                last_token_at=now + 0.001,
                ended_at=now + 0.001,
                ttft_s=now - start,
                decode_s=0.001,
                total_s=now + 0.001 - start,
                prompt_tokens=10,
                completion_tokens=2,
                prefill_tps=1000.0,
                decode_tps=1000.0,
                usage_reported=True,
                streamed=True,
                text_chars=10,
            )
        finally:
            self.in_flight -= 1


class FastFailStubClient:
    """Stub whose requests fail immediately, with no delay."""

    def __init__(self):
        self.calls = 0

    async def stream(self, prompt, max_tokens, temperature=0.0, seed=None):
        self.calls += 1
        now = time.perf_counter()
        return RequestResult.error_result(now, now, "boom")


class PhasedStubClient:
    """Stub with a distinct prefill phase followed by a spread decode phase."""

    def __init__(
        self,
        prefill_s=0.02,
        decode_s=0.08,
        tokens=10,
        cold_s=0.0,
        cold_calls=0,
    ):
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.tokens = tokens
        self.cold_s = cold_s
        self.cold_calls = cold_calls
        self.calls = 0

    async def stream(self, prompt, max_tokens, temperature=0.0, seed=None):
        self.calls += 1
        call_index = self.calls
        start = time.perf_counter()
        if call_index <= self.cold_calls:
            await asyncio.sleep(self.cold_s)
        await asyncio.sleep(self.prefill_s)
        first = time.perf_counter()
        await asyncio.sleep(self.decode_s)
        last = time.perf_counter()
        return RequestResult(
            ok=True,
            error=None,
            started_at=start,
            first_token_at=first,
            last_token_at=last,
            ended_at=last,
            ttft_s=first - start,
            decode_s=last - first,
            total_s=last - start,
            prompt_tokens=self.tokens,
            completion_tokens=self.tokens,
            prefill_tps=self.tokens / (first - start),
            decode_tps=(self.tokens - 1) / (last - first),
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


def test_sustained_window_keeps_concurrency_in_flight():
    async def run():
        client = SlowStubClient(request_s=0.02)
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            3,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.15,
        )
        return client, wave

    client, wave = asyncio.run(run())

    assert client.max_in_flight == 3
    assert client.calls > 3
    assert wave.concurrency == 3
    assert wave.ok == client.calls
    assert len(wave.requests) == client.calls
    assert wave.wall_s == pytest.approx(0.15, abs=0.02)
    assert wave.output_tps is not None
    assert wave.req_per_s is not None


def test_sustained_window_stops_launching_after_deadline():
    async def run():
        client = SlowStubClient(request_s=0.02)
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            2,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.1,
        )
        return client, wave

    client, wave = asyncio.run(run())

    # Every request launched before the deadline; in-flight ones finished.
    starts = [request.started_at for request in wave.requests]
    ends = [request.ended_at for request in wave.requests]
    deadline_relative = min(starts) + 0.1
    assert max(starts) < deadline_relative + 0.05
    assert max(ends) >= deadline_relative
    assert client.calls == len(wave.requests)


def test_sustained_window_records_failures():
    async def run():
        client = SlowStubClient(request_s=0.01, fail_every=2)
        return client, await run_sustained_window(
            client,
            StubBuilder(),
            2,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.1,
        )

    client, wave = asyncio.run(run())

    assert wave.failed == sum(1 for request in wave.requests if not request.ok)
    assert wave.failed >= 1
    assert wave.ok == client.calls - wave.failed


def test_sustained_window_reports_progress_per_request():
    async def run():
        client = SlowStubClient(request_s=0.01)
        events = []

        wave = await run_sustained_window(
            client,
            StubBuilder(),
            2,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.08,
            on_request=lambda concurrency, done, failed, elapsed: events.append(
                (concurrency, done, failed)
            ),
        )
        return wave, events

    wave, events = asyncio.run(run())

    assert events
    assert events[-1] == (2, len(wave.requests), wave.failed)
    assert [done for _, done, _ in events] == list(range(1, len(events) + 1))


def test_run_benchmark_sustained_mode_runs_windows_with_burst_warmup():
    async def run():
        client = SlowStubClient(request_s=0.01)
        state = RunState()
        await run_benchmark(
            client,
            StubBuilder(),
            [2],
            state=state,
            waves=2,
            warmup_waves=1,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="sustained",
            duration=0.08,
        )
        return client, state

    client, state = asyncio.run(run())

    assert len(state.results) == 1
    level = state.results[0]
    assert level.summary.waves == 2
    # Two windows, each keeping 2 slots busy for 0.08s: more than 2 per window.
    assert level.summary.requests >= 4
    warmup_requests = len(level.warmup_waves[0].requests)
    assert level.summary.requests == client.calls - warmup_requests
    assert len(level.warmup_waves) == 1
    assert warmup_requests == 2
    assert level.summary.decode_tps_median is not None
    assert level.summary.output_tps_aggregate is not None


def test_run_benchmark_burst_mode_is_the_default():
    async def run():
        client = StubClient()
        state = RunState()
        await run_benchmark(
            client,
            StubBuilder(),
            [2],
            state=state,
            waves=2,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
        )
        return client, state

    client, state = asyncio.run(run())

    # Burst fires exactly waves * concurrency requests, no rolling refill.
    assert client.calls == 4
    assert state.results[0].summary.waves == 2
    assert state.results[0].summary.requests == 4


def test_run_benchmark_rejects_unknown_pattern():
    async def run():
        state = RunState()
        await run_benchmark(
            StubClient(),
            StubBuilder(),
            [1],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="rolling",
        )

    with pytest.raises(ValueError):
        asyncio.run(run())


def test_sustained_window_stops_hammering_on_fast_failures():
    async def run():
        client = FastFailStubClient()
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            4,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=1.0,
        )
        return client, wave

    client, wave = asyncio.run(run())

    # Without the breaker each slot would loop errors until the deadline.
    assert client.calls == 4 * SLOT_MAX_CONSECUTIVE_FAILURES
    assert wave.failed == client.calls
    assert wave.ok == 0
    assert wave.slots_stopped_early == 4
    # Backoff paces the retries instead of spinning on the server.
    min_wall = SLOT_FAILURE_BACKOFF_S * (SLOT_MAX_CONSECUTIVE_FAILURES - 1)
    assert wave.wall_s >= min_wall - 0.01


def test_sustained_window_resumes_after_transient_failures(monkeypatch):
    monkeypatch.setattr(runner, "SLOT_FAILURE_BACKOFF_S", 0.01)

    async def run():
        client = SlowStubClient(request_s=0.01, fail_first=2)
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            1,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.15,
        )
        return client, wave

    client, wave = asyncio.run(run())

    # Two early failures do not stop the slot: the streak resets on success.
    assert client.calls >= 3
    assert wave.ok >= 1
    assert wave.failed == 2
    assert wave.slots_stopped_early == 0


def test_sustained_progress_counts_failures_incrementally(monkeypatch):
    monkeypatch.setattr(runner, "SLOT_FAILURE_BACKOFF_S", 0.01)

    async def run():
        client = SlowStubClient(request_s=0.01, fail_every=3)
        events = []
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            2,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.12,
            on_request=lambda concurrency, done, failed, elapsed: events.append(
                (done, failed)
            ),
        )
        return wave, events

    wave, events = asyncio.run(run())

    assert events
    assert events[-1] == (len(wave.requests), wave.failed)
    failed_counts = [failed for _, failed in events]
    assert failed_counts == sorted(failed_counts)


def test_sustained_failure_warning_does_not_claim_concurrency():
    async def run():
        state = RunState()
        await run_benchmark(
            SlowStubClient(request_s=0.01, fail_first=8),
            StubBuilder(),
            [4],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="sustained",
            duration=1.0,
        )
        return state

    state = asyncio.run(run())

    level = state.results[0]
    assert level.summary.failed == 8
    assert level.summary.ok >= 4
    assert any("excluded from the statistics" in w for w in state.warnings)
    assert all("concurrent streams" not in w for w in state.warnings)


def test_sustained_warns_when_streams_stop_early(monkeypatch):
    monkeypatch.setattr(runner, "SLOT_MAX_CONSECUTIVE_FAILURES", 2)
    monkeypatch.setattr(runner, "SLOT_FAILURE_BACKOFF_S", 0.01)

    async def run():
        state = RunState()
        await run_benchmark(
            SlowStubClient(request_s=0.01, fail_first=3),
            StubBuilder(),
            [2],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="sustained",
            duration=0.3,
        )
        return state

    state = asyncio.run(run())

    level = state.results[0]
    assert level.summary.failed == 3
    assert level.summary.ok >= 1
    assert level.waves[0].slots_stopped_early == 1
    assert any(
        "users=2: 1 of 2 streams stopped early" in w for w in state.warnings
    )


def _max_launch_cluster(times, tolerance_s=0.005):
    """Largest group of launch timestamps closer together than tolerance_s."""
    ordered = sorted(times)
    best = 1
    run = 1
    for previous, current in zip(ordered, ordered[1:]):
        if current - previous < tolerance_s:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def test_sustained_window_staggers_launches():
    request_s = 0.06

    async def run(stagger):
        client = SlowStubClient(request_s=request_s)
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            4,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.3,
            stagger=stagger,
        )
        return client, wave

    staggered_client, staggered = asyncio.run(run(request_s / 4))
    clustered_client, clustered = asyncio.run(run(0.0))

    # Identical request shapes make lockstep slots relaunch together; the
    # stagger keeps the launches spread across the window instead.
    assert len(staggered.requests) > 4
    assert _max_launch_cluster(staggered_client.launch_times) < 4
    assert _max_launch_cluster(clustered_client.launch_times) == 4
    first_four = sorted(staggered_client.launch_times)[:4]
    assert first_four[3] - first_four[0] >= 3 * (request_s / 4) - 0.005


def test_run_benchmark_sustained_staggers_from_warmup():
    async def run():
        client = SlowStubClient(request_s=0.06)
        state = RunState()
        await run_benchmark(
            client,
            StubBuilder(),
            [4],
            state=state,
            waves=1,
            warmup_waves=1,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="sustained",
            duration=0.3,
        )
        return state

    state = asyncio.run(run())

    measured = state.results[0].waves[0]
    assert len(measured.requests) > 4
    assert _max_launch_cluster([r.started_at for r in measured.requests]) < 4


def test_sustained_window_excludes_the_ramp_from_the_rates():
    async def run():
        client = PhasedStubClient(prefill_s=0.02, decode_s=0.08, tokens=10)
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            3,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.10,
            stagger=0.04,
        )
        return wave

    wave = asyncio.run(run())

    # The slots launch 0.04 apart, so slot 2 starts ~0.08 in: the measured
    # window is the 0.10s from then, not the ~0.18s since slot 0 launched.
    assert wave.wall_s == pytest.approx(0.10, abs=0.02)
    # Work done during the ramp (e.g. slot 0's first prefill) is not counted.
    assert wave.prompt_tokens_in_window < wave.prompt_tokens
    assert wave.completion_tokens_in_window < wave.completion_tokens


def test_sustained_cold_warmup_does_not_lower_the_rates():
    async def run(cold):
        client = PhasedStubClient(
            prefill_s=0.02,
            decode_s=0.08,
            tokens=10,
            cold_s=0.30 if cold else 0.0,
            cold_calls=2 if cold else 0,
        )
        state = RunState()
        await run_benchmark(
            client,
            StubBuilder(),
            [2],
            state=state,
            waves=1,
            warmup_waves=1,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="sustained",
            duration=0.15,
        )
        return state

    warm = asyncio.run(run(cold=False))
    cold = asyncio.run(run(cold=True))

    # The cold warmup stretches the stagger estimate, but the ramp is not
    # measured, so the window's rates stay at the warm level.
    warm_tps = warm.results[0].summary.output_tps_aggregate
    cold_tps = cold.results[0].summary.output_tps_aggregate
    assert cold_tps == pytest.approx(warm_tps, rel=0.2)


def test_sustained_per_request_stats_exclude_ramp_samples():
    def sample(started_at, decode_tps):
        return RequestResult(
            ok=True,
            error=None,
            started_at=started_at,
            first_token_at=started_at + 0.01,
            last_token_at=started_at + 0.02,
            ended_at=started_at + 0.02,
            ttft_s=0.01,
            decode_s=0.01,
            total_s=0.02,
            prompt_tokens=10,
            completion_tokens=10,
            prefill_tps=1000.0,
            decode_tps=decode_tps,
            usage_reported=True,
            streamed=True,
            text_chars=10,
        )

    window_start = 100.0
    ramp = [sample(99.0, 300.0), sample(99.5, 300.0)]
    inside = [sample(100.0, 100.0), sample(100.5, 100.0)]

    ramp_wave = _build_wave(
        2, 0, False, ramp, window_start=window_start, window_end=101.0
    )
    inside_wave = _build_wave(
        2, 1, False, inside, window_start=window_start, window_end=101.0
    )

    summary = summarize_level(2, [ramp_wave, inside_wave])

    # Ramp-up requests ran with fewer than N streams in flight and decode
    # faster; they must not feed the per-request medians.
    assert ramp_wave.window_start == window_start
    assert inside_wave.window_start == window_start
    assert summary.decode_tps_median == pytest.approx(100.0)
    assert summary.decode_tps_p10 == pytest.approx(100.0)
    assert summary.ttft_median_s == pytest.approx(0.01)


def test_burst_per_request_stats_keep_every_sample():
    def sample(started_at, decode_tps):
        return RequestResult(
            ok=True,
            error=None,
            started_at=started_at,
            first_token_at=started_at + 0.01,
            last_token_at=started_at + 0.02,
            ended_at=started_at + 0.02,
            ttft_s=0.01,
            decode_s=0.01,
            total_s=0.02,
            prompt_tokens=10,
            completion_tokens=10,
            prefill_tps=1000.0,
            decode_tps=decode_tps,
            usage_reported=True,
            streamed=True,
            text_chars=10,
        )

    wave = _build_wave(2, 0, False, [sample(0.0, 300.0), sample(1.0, 100.0)])

    assert wave.window_start is None
    summary = summarize_level(2, [wave])
    assert summary.decode_tps_median == pytest.approx(200.0)


def test_run_benchmark_sustained_without_warmup_launches_together():
    async def run():
        client = SlowStubClient(request_s=0.06)
        state = RunState()
        await run_benchmark(
            client,
            StubBuilder(),
            [4],
            state=state,
            waves=1,
            warmup_waves=0,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            pattern="sustained",
            duration=0.3,
        )
        return state

    state = asyncio.run(run())

    measured = state.results[0].waves[0]
    assert _max_launch_cluster([r.started_at for r in measured.requests]) == 4


def test_sustained_window_prorates_the_drain_tail():
    async def run():
        client = PhasedStubClient(prefill_s=0.02, decode_s=0.08, tokens=10)
        wave = await run_sustained_window(
            client,
            StubBuilder(),
            1,
            max_tokens=2,
            temperature=0.0,
            seed=None,
            index=0,
            duration=0.06,
        )
        return wave

    wave = asyncio.run(run())

    # The request runs [0, 0.10] in a window ending at 0.06: all prompt
    # tokens land inside, half the decode tokens do.
    assert wave.wall_s == pytest.approx(0.06, abs=0.01)
    assert wave.prompt_tokens == 10
    assert wave.completion_tokens == 10
    assert wave.prompt_tokens_in_window == pytest.approx(10.0, rel=0.1)
    assert wave.completion_tokens_in_window == pytest.approx(5.0, rel=0.1)
    assert wave.prefill_tps_aggregate == pytest.approx(10.0 / 0.06, rel=0.1)
    assert wave.output_tps == pytest.approx(5.0 / 0.06, rel=0.1)
    # The request finished after the deadline, so it is not counted as an
    # in-window completion.
    assert wave.req_per_s is None
    assert wave.ok == 1


def test_clip_ratio_prorates_intervals_against_the_window_end():
    from llmbench.runner import _clip_ratio

    window_start, window_end = 0.0, 0.15

    # Fully inside the window.
    assert _clip_ratio(0.00, 0.10, window_start, window_end) == 1.0
    # Ending exactly at the deadline still counts fully.
    assert _clip_ratio(0.05, 0.15, window_start, window_end) == 1.0
    # Straddling the deadline: only the part before it counts.
    assert _clip_ratio(0.05, 0.20, window_start, window_end) == pytest.approx(
        10.0 / 15.0
    )
    # Entirely after the deadline counts for nothing.
    assert _clip_ratio(0.16, 0.20, window_start, window_end) == 0.0
    # Degenerate (empty) intervals.
    assert _clip_ratio(0.10, 0.10, window_start, window_end) == 1.0
    assert _clip_ratio(0.20, 0.20, window_start, window_end) == 0.0

    # The window also excludes the ramp-up before it opens.
    assert _clip_ratio(0.00, 0.10, 0.05, 0.15) == pytest.approx(0.5)
    assert _clip_ratio(0.00, 0.04, 0.05, 0.15) == 0.0
    assert _clip_ratio(0.02, 0.20, 0.05, 0.15) == pytest.approx(0.10 / 0.18)
