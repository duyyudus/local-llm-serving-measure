import io
from datetime import datetime, timezone

from rich.console import Console

from llmbench.client import RequestResult
from llmbench.report import build_report, render_table
from llmbench.runner import LevelResult, LevelSummary, Wave, failure_warning


def _request(completion_tokens, *, ok=True, usage_reported=True):
    return RequestResult(
        ok=ok,
        error=None if ok else "boom",
        started_at=0.0,
        first_token_at=0.01 if ok else None,
        last_token_at=0.02 if ok else None,
        ended_at=0.03,
        ttft_s=0.01 if ok else None,
        decode_s=0.01 if ok else None,
        total_s=0.03,
        prompt_tokens=10 if ok else None,
        completion_tokens=completion_tokens,
        prefill_tps=1000.0 if ok else None,
        decode_tps=100.0 if ok else None,
        usage_reported=usage_reported,
        streamed=ok,
        text_chars=10 if ok else 0,
    )


def _wave(requests, index=0):
    ok = sum(request.ok for request in requests)
    failed = len(requests) - ok
    ok_requests = [request for request in requests if request.ok]
    prompt_total = sum(request.prompt_tokens or 0 for request in ok_requests)
    completion_total = sum(request.completion_tokens or 0 for request in ok_requests)

    return Wave(
        concurrency=len(requests),
        index=index,
        warmup=False,
        requests=requests,
        wave_start=0.0,
        wave_end=0.03,
        wall_s=0.03,
        ok=ok,
        failed=failed,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        output_tps=completion_total / 0.03 if completion_total else None,
        req_per_s=ok / 0.03 if ok else None,
        prefill_tps_aggregate=1000.0 if ok else None,
    )


def _level(waves_requests):
    waves = [
        _wave(requests, index=index)
        for index, requests in enumerate(waves_requests)
    ]
    requests = sum(len(wave.requests) for wave in waves)
    ok = sum(wave.ok for wave in waves)
    failed = sum(wave.failed for wave in waves)
    prompt_total = sum(wave.prompt_tokens for wave in waves)
    completion_total = sum(wave.completion_tokens for wave in waves)

    summary = LevelSummary(
        concurrency=waves[0].concurrency,
        waves=len(waves),
        requests=requests,
        ok=ok,
        failed=failed,
        ttft_median_s=0.01 if ok else None,
        ttft_p90_s=0.01 if ok else None,
        prefill_tps_median=1000.0 if ok else None,
        decode_tps_median=100.0 if ok else None,
        decode_tps_p10=100.0 if ok else None,
        prefill_tps_aggregate=1000.0 if ok else None,
        output_tps_aggregate=completion_total / 0.03 if completion_total else None,
        requests_per_s=ok / 0.03 if ok else None,
        wall_s_median=0.03,
        prompt_tokens_total=prompt_total,
        completion_tokens_total=completion_total,
        errors=["boom"] if failed else [],
    )
    return LevelResult(
        concurrency=waves[0].concurrency,
        summary=summary,
        waves=waves,
        warmup_waves=[],
    )


def _render(levels, warnings=None):
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, width=200)
    render_table(
        console,
        {"model": "model", "endpoint": "endpoint"},
        levels,
        warnings=warnings or [],
    )
    return buffer.getvalue()


def test_output_length_warning_ignores_unreported_usage():
    level = _level(
        [
            [
                _request(128, usage_reported=False),
                _request(64, usage_reported=False),
            ]
        ]
    )

    assert "output lengths varied" not in _render([level])


def test_output_length_warning_uses_reported_usage():
    level = _level([[_request(128), _request(64)]])

    assert "output lengths varied" in _render([level])


def test_partial_failure_row_is_marked():
    level = _level([[_request(128), _request(None, ok=False, usage_reported=False)]])
    warning = failure_warning(level.concurrency, level.summary, level.waves)

    output = _render([level], warnings=[warning])

    assert "Warning:" in output
    assert "2 (1 ok/wave)" in output
    assert "only 1/2 requests succeeded" in output
    assert "median 1 ok/wave" in output
    assert "about 1 concurrent stream" in output


def test_partial_failure_label_uses_per_wave_successes():
    level = _level(
        [
            [
                _request(128),
                _request(64),
                _request(None, ok=False, usage_reported=False),
            ],
            [
                _request(128),
                _request(64),
                _request(None, ok=False, usage_reported=False),
            ],
        ]
    )

    warning = failure_warning(level.concurrency, level.summary, level.waves)

    output = _render([level], warnings=[warning])

    assert "3 (2 ok/wave)" in output
    assert "3 (4 ok)" not in output
    assert "only 4/6 requests succeeded" in output
    assert "median 2 ok/wave" in output
    assert "about 2 concurrent streams" in output


def test_build_report_keeps_error_null_for_warnings():
    now = datetime.now(timezone.utc)
    report = build_report(
        {},
        [],
        now,
        now,
        1.0,
        error=None,
        warnings=["users=3: only 2/3 requests succeeded"],
    )

    assert report["error"] is None
    assert report["warnings"] == ["users=3: only 2/3 requests succeeded"]
