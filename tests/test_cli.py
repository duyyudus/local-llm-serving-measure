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
