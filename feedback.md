# Code review feedback: `llm-bench`

Reviewed against the original requirements: configurable endpoint and model,
max parallel requests, max context length, and a table of prefill and decode
speed for each concurrency level.

**Status:** All findings from three rounds of review have been fixed, and the 35
tests pass. The fixes were checked with the test suite and by running the CLI
against a fake streaming server, including partial failures across several
waves, a server that is down, and Ctrl+C. No bugs are open.

## Optional cleanup

- **The partial-failure warning text is built in two places:**
  `src/llmbench/runner.py:342-347` and `src/llmbench/report.py:104-110`. If one
  copy changes, the terminal output and the JSON `warnings` can drift apart.
  Build the message once in the runner and have the report print
  `state.warnings`.

## Open questions

These need a decision before any more work:

1. **"Initial config":** CLI flags only, as built now? A saved config file (for
   example `llm-bench init` writes `~/.config/llm-bench.toml`, and later runs
   reuse the endpoint and model)? Or an interactive prompt the first time it
   runs?
2. **Context length sweep:** should it also sweep prompt sizes (for example
   1k/4k/16k × users 1..N)? Prefill behaviour changes a lot with prompt length.
3. **Load pattern:** every level is currently a burst: N requests start
   together and the level waits for the slowest. Add a "sustained" mode that
   keeps N requests running for a set time? It's closer to real serving and
   gives steadier decode numbers.
4. **Auto-discovery:** make the model name optional, picked automatically from
   `GET /v1/models` when the server serves only one model?
