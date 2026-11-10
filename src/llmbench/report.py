"""Terminal table rendering and JSON export."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.table import Table

from . import __version__
from .runner import LevelResult, ok_per_wave


def _fmt_compact(value: Optional[float], decimals: int = 1) -> str:
    """Format a number compactly so wide rates do not break the table."""
    if value is None:
        return "—"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.{decimals}f}"


def render_table(
    console: Console,
    config: Dict[str, Any],
    levels: List[LevelResult],
    *,
    warnings: List[str],
) -> None:
    if not levels:
        console.print("[yellow]No levels completed.[/]")

    table = Table(
        title=f"{config['model']} @ {config['endpoint']}",
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        show_edge=False,
        pad_edge=False,
        header_style="bold",
    )
    table.add_column("Users", justify="right", style="bold", min_width=8)
    table.add_column("TTFT\nmed", justify="right")
    table.add_column("TTFT\np90", justify="right")
    table.add_column("Prefill\nmed", justify="right")
    table.add_column("Decode\nmed", justify="right")
    table.add_column("Prefill\nagg", justify="right")
    table.add_column("Output\nagg", justify="right")
    table.add_column("Req/s", justify="right")

    has_errors = any(level.summary.failed for level in levels)
    if has_errors:
        table.add_column("Err", justify="right")

    for level in levels:
        summary = level.summary
        users_label = str(level.concurrency)
        if summary.failed:
            users_label = f"{level.concurrency} ({ok_per_wave(level.waves)} ok/wave)"
        row = [
            users_label,
            _fmt_compact(
                summary.ttft_median_s * 1000.0 if summary.ttft_median_s is not None else None,
                decimals=0,
            ),
            _fmt_compact(
                summary.ttft_p90_s * 1000.0 if summary.ttft_p90_s is not None else None,
                decimals=0,
            ),
            _fmt_compact(summary.prefill_tps_median, decimals=1),
            _fmt_compact(summary.decode_tps_median, decimals=1),
            _fmt_compact(summary.prefill_tps_aggregate, decimals=1),
            _fmt_compact(summary.output_tps_aggregate, decimals=1),
            _fmt_compact(summary.requests_per_s, decimals=2),
        ]
        if has_errors:
            row.append(str(summary.failed))
        table.add_row(*row)

    console.print(table)
    console.print(
        "[dim]TTFT in ms; token speeds in tok/s. TTFT includes connection, "
        "queueing, prefill and the first token. Prefill med = prompt_tokens / "
        "TTFT (includes queueing). Prefill agg = total prompt tokens / time "
        "until the last request gets its first token. Decode = "
        "(completion_tokens - 1) / (last_token - first_token). "
        "Agg columns are per-wave medians. Users shows '(N ok/wave)' (median "
        "per wave) when some requests failed.[/]"
    )

    for warning in warnings:
        console.print(f"[yellow]Warning:[/] {warning}")

    if has_errors:
        for level in levels:
            if level.summary.errors:
                first_error = level.summary.errors[0].replace("\n", " ")[:240]
                console.print(
                    f"[yellow]users={level.concurrency} error:[/] {first_error}"
                )

    for level in levels:
        lengths = [
            request.completion_tokens
            for wave in level.waves
            for request in wave.requests
            if request.ok
            and request.usage_reported
            and request.completion_tokens is not None
        ]
        if len(lengths) < 2:
            continue
        low, high = min(lengths), max(lengths)
        if high - low >= max(2, int(high * 0.1)):
            console.print(
                f"[yellow]Warning:[/] users={level.concurrency} output lengths "
                f"varied from {low} to {high} tokens; decode throughput may not "
                f"reflect {level.concurrency} concurrent streams."
            )


def build_report(
    config: Dict[str, Any],
    levels: List[LevelResult],
    started_at: datetime,
    finished_at: datetime,
    duration_s: float,
    *,
    error: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    interrupted: bool = False,
) -> Dict[str, Any]:
    return {
        "schema_version": 3,
        "tool": "llm-bench",
        "version": __version__,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": duration_s,
        "error": error,
        "warnings": warnings or [],
        "interrupted": interrupted,
        "config": config,
        "levels": [level.to_dict() for level in levels],
    }


def write_json(path: str, report: Dict[str, Any]) -> None:
    target = Path(path)
    if target.parent and str(target.parent) != ".":
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
