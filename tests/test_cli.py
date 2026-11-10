import asyncio
import io

import pytest
from rich.console import Console

import llmbench.cli as cli
from llmbench.cli import build_parser, main
from llmbench.runner import RunState


def test_api_key_is_not_printed_in_help(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test")

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "sk-secret-test" not in captured.out
    assert "OPENAI_API_KEY" in captured.out


def test_prompt_tokens_and_deprecated_alias():
    parser = build_parser()

    args = parser.parse_args(["--model", "model", "--prompt-tokens", "123"])
    assert args.prompt_tokens == 123

    args = parser.parse_args(["--model", "model", "--max-context", "456"])
    assert args.prompt_tokens == 456

    args = parser.parse_args(["--model", "model"])
    assert args.prompt_tokens == 2048


def test_pattern_defaults_to_burst_and_parses_sustained():
    parser = build_parser()

    args = parser.parse_args(["--model", "model"])
    assert args.pattern == "burst"
    assert args.duration == 30.0

    args = parser.parse_args(
        ["--model", "model", "--pattern", "sustained", "--duration", "5"]
    )
    assert args.pattern == "sustained"
    assert args.duration == 5.0


def test_duration_must_be_positive(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--model", "model", "--duration", "0"])

    assert excinfo.value.code == 2
    assert "--duration must be > 0" in capsys.readouterr().err


def test_unknown_pattern_is_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--model", "model", "--pattern", "rolling"])

    assert excinfo.value.code == 2


def test_disabled_param_warning_prints_on_early_stop(monkeypatch):
    args = build_parser().parse_args(["--model", "model", "--quiet"])

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.disabled_params = ["ignore_eos"]

        async def aclose(self):
            pass

    async def fake_run_benchmark(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "ChatClient", FakeClient)
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    buffer = io.StringIO()
    status = Console(file=buffer, no_color=True)
    state = RunState()

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(cli._run(args, [1], {}, status, state))

    assert "disabled: ignore_eos" in buffer.getvalue()


def test_sustained_flags_reach_run_benchmark_and_progress_throttles(monkeypatch):
    args = build_parser().parse_args(
        ["--model", "model", "--pattern", "sustained", "--duration", "2"]
    )

    captured = {}

    async def fake_run_benchmark(client, builder, levels, **kwargs):
        captured.update(kwargs)
        on_request = kwargs["on_request"]
        for done in range(1, 6):
            on_request(2, done, 0, 0.1 * done)
        return kwargs["state"]

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    buffer = io.StringIO()
    status = Console(file=buffer, no_color=True)
    state = RunState()

    asyncio.run(cli._run(args, [2], {}, status, state))

    assert captured["pattern"] == "sustained"
    assert captured["duration"] == 2.0
    output = buffer.getvalue()
    # Five completions arrive within a second of each other; only the first
    # prints, so long sustained windows do not flood the terminal.
    assert "1 done, 0 failed" in output
    assert "2 done" not in output
