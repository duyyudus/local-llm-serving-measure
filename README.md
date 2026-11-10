# local-llm-bench

A small Python CLI that measures how fast a local OpenAI-compatible LLM server
handles **many parallel requests**. For each concurrency level it reports
effective prefill speed, steady-state decode speed, and aggregate throughput.

Works with vLLM, SGLang, llama.cpp server, LM Studio, TGI, Ollama's OpenAI
shim, and anything else that implements `POST /v1/chat/completions` with
streaming.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

That creates `.venv`, installs the project in editable mode, and installs the
dev dependency group. For a runtime-only environment:

```bash
uv sync --no-dev
```

Run tests with:

```bash
uv run pytest
```

## Quick start

```bash
uv run llm-bench \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-parallel 8 \
  --prompt-tokens 4096 \
  --max-tokens 128 \
  --json results.json
```

This runs concurrency levels `1,2,3,4,5,6,7,8`. Each level gets one warmup wave
and three measured waves, so the total request count is
`(1+2+...+8) * (1+3) = 144`.

Pick explicit levels instead:

```bash
uv run llm-bench -m my-model --levels 1,2,4,8 --waves 5 --warmup-waves 1
```

## Load patterns

`--pattern` chooses how each concurrency level generates load:

- **burst** (default): each wave fires N requests at the same moment and ends
  when the slowest request finishes. This is the classic benchmark shape and
  the aggregate columns describe one synchronized batch. Because every wave
  starts with a scheduling stampede and ends with a shrinking batch, decode
  numbers can jitter between waves.
- **sustained**: each measured wave is a time window of `--duration` seconds
  (default 30) that keeps exactly N requests in flight. As soon as a request
  finishes, a new one starts, so the server sees a steady stream instead of a
  synchronized stampede. This is closer to real serving, and per-request
  metrics are pooled across many requests, which gives steadier decode
  numbers.

Details that matter in sustained mode:

- `--waves` counts windows. Three windows of 30 seconds per level is 90
  seconds per level; use `--waves 1` or `--waves 2` for long sessions.
- Slots start out of step: slot `i` waits `i × (estimated request time / N)`
  before its first request. The estimate comes from the burst warmup waves
  and is capped at `--duration / N`, which limits the ramp-up to under one
  `--duration` (and so the wall-clock cost of a window to under two) even
  when the warmup estimate is bad. With identical request shapes, a batching
  server would otherwise relaunch every slot in lockstep — back-to-back
  bursts. `--warmup-waves 0` leaves no estimate, so slots launch together.
- The window clock starts when the last slot launches, so the staggered
  ramp-up is not measured: a slow or cold first warmup can stretch the ramp
  (and the wall-clock run time of the window) but does not dilute the rates.
  Per-request medians likewise use only requests that started inside the
  window, since ramp-up requests ran with fewer than N streams in flight.
- Requests already running when the window ends are allowed to finish, but
  they do not dilute the rates: `Prefill agg` and `Output agg` count only the
  tokens produced inside the window (tokens outside it are prorated by each
  request's token rate) and divide by the window length. `Req/s` counts
  requests that completed inside the window.
- Warmup waves always run as bursts, whatever the pattern; they exist to warm
  the server, not to measure it.
- Requests that fail do not refill the slot in a tight loop: after 3
  consecutive failures a slot stops launching requests (with a short backoff
  between failed attempts), and a warning reports how many streams stopped
  early, since effective concurrency is then lower than requested.
- The columns keep their meanings, with one shift: `Prefill agg` becomes
  prompt tokens prefilled inside the window per window second (prompts are
  prefilled continuously rather than in one synchronized batch), and
  `Output agg` is tokens generated inside the window per window second. With
  the drain tail prorated away, these are steady-state rates rather than
  burst averages.

```bash
uv run llm-bench -m my-model --levels 4,8 --pattern sustained \
  --duration 30 --waves 2 --json sustained.json
```

## What the table means

| Column | Meaning |
| --- | --- |
| `Users` | Number of requests fired simultaneously (burst) or kept in flight (sustained). Shows `(N ok/wave)` when some requests in that level failed; `N` is the median successful requests per wave, so the metrics reflect about that many concurrent streams. In sustained mode it shows `(N failed)` instead, because completions per window say nothing about how many streams ran. |
| `TTFT med` / `p90` | Time to first streamed token, in milliseconds. Includes connection, server queueing, prefill, and the first decode step. |
| `Prefill med` | `prompt_tokens / TTFT` for each request, in tokens/sec. This is effective prefill speed, not pure GPU prefill rate. |
| `Decode med` | `(completion_tokens - 1) / (last_token - first_token)` per request, in tokens/sec. The p10 slow-tail value is kept in the JSON export. |
| `Prefill agg` | Total prompt tokens across the wave divided by the time until the last request receives its first token. This measures aggregate prefill throughput under load. In sustained mode: prompt tokens prefilled inside the window divided by the window length. |
| `Output agg` | Total completion tokens across the wave divided by wave wall time. In sustained mode: tokens generated inside the window (the drain tail is prorated) divided by the window length. |
| `Req/s` | Successful requests divided by wave wall time. In sustained mode: requests that completed inside the window divided by the window length. |
| `Err` | Failed requests at that level; shown only when at least one request failed. Failed requests are excluded from the other statistics. |

Aggregate columns are the median of the measured waves, which is more stable
than a single wave. Large rates are abbreviated with `k`/`M`; the JSON export
contains the exact values.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--endpoint`, `-e` | `http://localhost:8000/v1` | Base URL or full `/chat/completions` URL. |
| `--model`, `-m` | required | Model name as the server expects it. |
| `--max-parallel`, `-p` | `4` | Highest level; default levels are `1..max-parallel`. |
| `--prompt-tokens`, `-c` | `2048` | Target prompt tokens, including the chat template. `--max-context` remains as a deprecated alias. |
| `--max-tokens`, `-n` | `128` | Output tokens requested per request. |
| `--levels` | `1..max-parallel` | Explicit comma-separated levels. |
| `--waves` | `3` | Measured waves per level. With `--pattern sustained`, this is the number of sustained windows per level. |
| `--warmup-waves` | `1` | Warmup waves per level. Set `0` to disable. Warmups always run as bursts. |
| `--pattern` | `burst` | Load pattern per measured wave: `burst` fires N requests together and waits for the slowest; `sustained` keeps N requests in flight. |
| `--duration` | `30` | Target length of each sustained window, in seconds. Only used with `--pattern sustained`. |
| `--temperature` | `0.0` | Sampling temperature. |
| `--seed` | none | Optional sampling seed. |
| `--timeout` | `600` | Per-request read timeout, seconds. |
| `--api-key` | unset; falls back to `$OPENAI_API_KEY` | Bearer token. The environment value is never printed in `--help`. |
| `--encoding` | `cl100k_base` | tiktoken encoding used to size prompts. |
| `--chat-overhead` | `16` | Tokens subtracted from `--prompt-tokens` for chat template overhead. |
| `--unique-prompt` / `--no-unique-prompt` | unique | Add a nonce and corpus offset per prompt to defeat prefix caching. |
| `--ignore-eos` / `--no-ignore-eos` | on | Send vLLM-style `ignore_eos` so outputs reach `--max-tokens`. |
| `--include-usage` / `--no-include-usage` | on | Send `stream_options.include_usage` to get server token counts. |
| `--extra-body` | none | JSON object merged into every request body. |
| `--json PATH` | none | Write full per-request results and summaries. |
| `--no-verify` | off | Disable TLS verification. |
| `--continue-on-error` | off | Keep going when every request in a wave fails. |
| `--quiet`, `-q` | off | Hide per-wave progress lines. |

## Metric details and caveats

- **TTFT is effective latency.** Under concurrency it includes time spent in
  the server's queue. That is the number a user actually experiences, but it is
  not a pure prefill throughput measurement. `Prefill agg` is the better
  under-load number: total prompt tokens divided by the time until the last
  request gets its first token.
- **Decode excludes the first token.** The first token is part of TTFT, so
  decode speed uses `completion_tokens - 1`. Requests with fewer than two
  completion tokens show `—`.
- **Token counts come from the server when available.** If the server does not
  return `usage`, the tool falls back to local tiktoken counting. Prompt sizes
  are always approximate because tiktoken is not the model's own tokenizer.
  tiktoken may need network access the first time it loads an encoding; if that
  fails, the tool warns and falls back to roughly four characters per token.
- **Unique prompts defeat prefix caching by default.** Reusing the same prompt
  can make later prefill artificially fast. Use `--no-unique-prompt` only when
  you deliberately want to measure cache behavior.
- **Optional parameters may be rejected.** `ignore_eos` and `stream_options`
  are not part of the original OpenAI API. If a server returns HTTP 400 and
  explicitly names one of those parameters, the client disables only that
  parameter and retries. Context-length errors and generic errors are not
  treated as unsupported parameters. A warning is printed when something was
  disabled.
- **Output lengths can vary.** If requests stop at different token counts, the
  effective batch shrinks and decode throughput may not reflect the advertised
  concurrency. The table prints a warning when output lengths vary
  significantly within a level.
- **Non-streaming servers cannot be split.** If the server ignores `stream` and
  returns a single JSON response, TTFT is reported as the whole request and
  decode speed is `—`.
- **Context limits are the server's job.** `--prompt-tokens + --max-tokens`
  must fit the model's context window. The tool does not know that window; the
  server will reject oversized requests.
- **Partial results are kept.** If a level fails or the run is interrupted, the
  table and JSON still contain every level that completed, plus an `error`
  field explaining where the run stopped. If only some requests in a level
  fail, the row is marked `(N ok/wave)` (or `(N failed)` in sustained mode),
  the JSON records a warning, and the exit code is `1`. A completed run keeps
  `error: null`; partial failures live in `warnings`. An interrupt discards
  the entire level that was in progress — in burst mode too — so only fully
  completed levels are kept.
- **Run the same way twice.** Warm caches, GPU clocks, background load, and
  network noise all move the numbers. Treat small differences as noise.

## Exit codes

- `0` — the run completed with no failed requests.
- `1` — the run completed but some requests failed, or it stopped because a
  level failed. The table and JSON are still written.
- `130` — interrupted with Ctrl+C. The partial table and JSON are still written.
- `2` — invalid CLI arguments or an invalid endpoint.

Hitting the server's concurrency limit is a normal benchmark result, so `1`
does not mean the output is unusable; check `warnings` and the `Err` column.

## JSON output

`--json results.json` writes a self-contained record:

```json
{
  "schema_version": 4,
  "tool": "llm-bench",
  "version": "0.1.0",
  "started_at": "2026-11-10T12:00:00+00:00",
  "finished_at": "2026-11-10T12:04:12+00:00",
  "duration_s": 252.4,
  "error": null,
  "warnings": [],
  "interrupted": false,
  "config": {
    "endpoint": "...",
    "model": "...",
    "levels": [1, 2, 4],
    "pattern": "burst",
    "duration_s": 30.0,
    "prompt_tokens": 2048
  },
  "levels": [
    {
      "concurrency": 1,
      "summary": {
        "ttft_median_s": 0.12,
        "prefill_tps_median": 17000.0,
        "decode_tps_median": 92.0,
        "prefill_tps_aggregate": 16800.0,
        "output_tps_aggregate": 90.0
      },
      "warmup_waves": [],
      "waves": [
        {
          "index": 0,
          "wall_s": 1.42,
          "requests": [
            {
              "ok": true,
              "ttft_s": 0.12,
              "decode_s": 1.30,
              "prompt_tokens": 2048,
              "completion_tokens": 128,
              "prefill_tps": 17066.7,
              "decode_tps": 97.7,
              "usage_reported": true
            }
          ]
        }
      ]
    }
  ]
}
```

Every measured request is included, so the file can be plotted or post-processed
without re-running the benchmark.

## Tips for useful numbers

1. Warm up the server. Keep `--warmup-waves 1` unless startup is instant.
2. Use enough output tokens for decode to dominate: `--max-tokens 128` or more.
3. Use a long prompt to make prefill visible: `--prompt-tokens 4096` or more.
4. Run levels up to the concurrency you actually expect, not just the maximum
   the GPU can hold.
5. Run the benchmark twice and compare. The second run is usually cleaner.
6. Record the exact model, quantization, server version, GPU, and command line
   alongside the JSON output.
7. If burst decode numbers jitter between waves, switch to
   `--pattern sustained --duration 30 --waves 1` and compare the `Output agg`
   columns instead.
