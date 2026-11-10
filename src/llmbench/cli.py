"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console

from . import __version__
from .client import ChatClient, normalize_endpoint
from .prompts import PromptBuilder, TokenCounter
from .report import build_report, render_table, write_json
from .runner import RunState, Wave, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description=(
            "Benchmark an OpenAI-compatible chat server under increasing "
            "parallel load. Each concurrency level measures time to first "
            "token (prefill) and steady-state decode speed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: llm-bench --endpoint http://localhost:8000/v1 "
            "--model Qwen/Qwen2.5-7B-Instruct --max-parallel 8 "
            "--prompt-tokens 4096 --max-tokens 128"
        ),
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default="http://localhost:8000/v1",
        help="Base URL or full chat completions URL.",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name exactly as the server expects it.",
    )
    parser.add_argument(
        "--max-parallel",
        "-p",
        type=int,
        default=4,
        help="Highest concurrency level; levels default to 1..max-parallel.",
    )
    parser.add_argument(
        "--prompt-tokens",
        "-c",
        dest="prompt_tokens",
        type=int,
        default=2048,
        help="Target prompt size in tokens, including the chat template.",
    )
    parser.add_argument(
        "--max-context",
        dest="prompt_tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="Deprecated alias for --prompt-tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=128,
        help="Maximum output tokens per request.",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated concurrency levels, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=3,
        help="Measured waves per concurrency level.",
    )
    parser.add_argument(
        "--warmup-waves",
        type=int,
        default=1,
        help="Warmup waves per concurrency level.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature sent with every request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request read timeout in seconds.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token; defaults to OPENAI_API_KEY when omitted.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding used to size prompts.",
    )
    parser.add_argument(
        "--chat-overhead",
        type=int,
        default=16,
        help="Tokens reserved for chat template and role markers.",
    )
    parser.add_argument(
        "--unique-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a unique nonce per prompt to defeat prefix caching.",
    )
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask vLLM-style servers to ignore EOS and emit max_tokens.",
    )
    parser.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request token usage in the final streaming chunk.",
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON object merged into every request body.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write the full result set to this JSON file.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable TLS certificate verification.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running levels even when every request in a wave fails.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-wave progress lines.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _parse_levels(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[int]:
    if args.levels:
        try:
            levels = sorted({int(item.strip()) for item in args.levels.split(",") if item.strip()})
        except ValueError:
            parser.error("--levels must be a comma-separated list of integers")
        if not levels or any(level < 1 for level in levels):
            parser.error("--levels values must be >= 1")
        return levels
    return list(range(1, args.max_parallel + 1))


def _parse_extra_body(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    if not args.extra_body:
        return {}
    try:
        extra = json.loads(args.extra_body)
    except json.JSONDecodeError as exc:
        parser.error(f"--extra-body is not valid JSON: {exc}")
    if not isinstance(extra, dict):
        parser.error("--extra-body must be a JSON object")
    return extra


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    if args.prompt_tokens < 1:
        parser.error("--prompt-tokens must be >= 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.waves < 1:
        parser.error("--waves must be >= 1")
    if args.warmup_waves < 0:
        parser.error("--warmup-waves must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.chat_overhead < 0:
        parser.error("--chat-overhead must be >= 0")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")


async def _run(
    args: argparse.Namespace,
    levels: List[int],
    extra_body: Dict[str, Any],
    status: Console,
    state: RunState,
) -> RunState:
    counter = TokenCounter(args.encoding)
    if not counter.available:
        status.print(
            f"[yellow]Warning:[/] tiktoken encoding '{args.encoding}' is "
            f"unavailable ({counter.error}); prompts use ~4 chars/token."
        )

    target_text_tokens = max(1, args.prompt_tokens - args.chat_overhead)
    if args.prompt_tokens <= args.chat_overhead:
        status.print(
            "[yellow]Warning:[/] --prompt-tokens is not larger than "
            "--chat-overhead; using one text token."
        )

    builder = PromptBuilder(
        counter,
        target_text_tokens,
        unique=args.unique_prompt,
    )

    api_key = (
        args.api_key
        if args.api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )

    max_connections = max(max(levels), args.max_parallel)
    client = ChatClient(
        endpoint=args.endpoint,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        verify=not args.no_verify,
        extra_body=extra_body,
        ignore_eos=args.ignore_eos,
        include_usage=args.include_usage,
        seed=args.seed,
        counter=counter,
        max_connections=max_connections,
    )

    def on_wave(wave: Wave) -> None:
        if args.quiet:
            return
        kind = "warmup" if wave.warmup else f"wave {wave.index + 1}/{args.waves}"
        status.print(
            f"[dim]users={wave.concurrency:<3} {kind:<12} "
            f"{wave.ok}/{len(wave.requests)} ok  wall={wave.wall_s:.2f}s[/]"
        )

    try:
        await run_benchmark(
            client,
            builder,
            levels,
            state=state,
            waves=args.waves,
            warmup_waves=args.warmup_waves,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            continue_on_error=args.continue_on_error,
            on_wave=on_wave,
        )
    finally:
        if client.disabled_params:
            status.print(
                "[yellow]Warning:[/] server rejected optional parameters; "
                f"disabled: {', '.join(sorted(set(client.disabled_params)))}"
            )
        await client.aclose()

    return state


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console(no_color=args.no_color)
    status = Console(stderr=True, no_color=args.no_color)

    _validate(args, parser)
    levels = _parse_levels(args, parser)
    extra_body = _parse_extra_body(args, parser)

    try:
        endpoint = normalize_endpoint(args.endpoint)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.quiet:
        status.print(f"[bold]Benchmarking[/] {args.model} @ {endpoint}")
        status.print(
            f"[dim]levels={levels} waves={args.waves} "
            f"warmup={args.warmup_waves} prompt_tokens={args.prompt_tokens} "
            f"max_tokens={args.max_tokens}[/]"
        )
        if max(levels) > args.max_parallel:
            status.print(
                f"[yellow]Note:[/] levels include {max(levels)} > "
                f"--max-parallel {args.max_parallel}; using {max(levels)} "
                "connections."
            )

    state = RunState()
    started_at = datetime.now(timezone.utc)
    clock_start = time.perf_counter()

    try:
        asyncio.run(_run(args, levels, extra_body, status, state))
    except KeyboardInterrupt:
        state.interrupted = True
        if state.error is None:
            state.error = "interrupted by user"
        status.print("[yellow]Interrupted; showing partial results.[/]")
    except ValueError as exc:
        if state.error is None:
            state.error = f"configuration error: {exc}"
        status.print(f"[red]Configuration error:[/] {exc}")
    except Exception as exc:
        if state.error is None:
            state.error = f"{type(exc).__name__}: {exc}"
        status.print(f"[red]Benchmark failed:[/] {state.error}")

    finished_at = datetime.now(timezone.utc)
    duration_s = time.perf_counter() - clock_start

    config: Dict[str, Any] = {
        "endpoint": endpoint,
        "model": args.model,
        "levels": levels,
        "max_parallel": args.max_parallel,
        "prompt_tokens": args.prompt_tokens,
        "chat_overhead_tokens": args.chat_overhead,
        "max_tokens": args.max_tokens,
        "waves": args.waves,
        "warmup_waves": args.warmup_waves,
        "temperature": args.temperature,
        "seed": args.seed,
        "encoding": args.encoding,
        "unique_prompt": args.unique_prompt,
        "ignore_eos": args.ignore_eos,
        "include_usage": args.include_usage,
        "timeout_s": args.timeout,
        "extra_body": extra_body,
    }

    if state.error:
        status.print(f"[red]Stopped:[/] {state.error}")

    render_table(console, config, state.results)

    if state.warnings and not state.error:
        status.print(
            "[yellow]Completed with partial failures; see table warnings.[/]"
        )

    if args.json_path:
        report = build_report(
            config,
            state.results,
            started_at,
            finished_at,
            duration_s,
            error=state.error,
            warnings=state.warnings,
            interrupted=state.interrupted,
        )
        write_json(args.json_path, report)
        if not args.quiet:
            status.print(f"[green]Wrote[/] {args.json_path}")

    if state.interrupted:
        return 130
    if state.error or state.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
