"""Prompt generation with a target token length.

The token counts are only an estimate of what the server will see: every model
has its own tokenizer. The server-reported ``usage.prompt_tokens`` is used for
the final metrics whenever the server provides it.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import List, Optional

CORPUS = """
Local language model serving turns one large matrix multiplication into a
stream of small, latency-sensitive operations. A request arrives, the prompt
is tokenized, and the server performs a prefill pass over every prompt token
before it can emit the first response token. Prefill is usually compute bound
and benefits from batching, because many prompt tokens can be processed in
parallel. Decode is different: each new token depends on the previous one, so
the work is sequential and memory bandwidth often becomes the limiting factor.
A serving system that looks fast with one user can slow down sharply when
several users share the same GPU, because the scheduler must interleave decode
steps and fit larger batches into limited memory.

Throughput and latency are not the same measurement. Throughput counts how
many tokens the system produces per second across all active requests, while
latency describes how long one user waits. Increasing the batch size usually
raises throughput and lowers per-request decode speed at the same time. The
useful operating point depends on the workload: an interactive assistant cares
about time to first token and smooth token delivery, whereas an offline batch
job cares mostly about total tokens per second. A good benchmark reports both
views rather than collapsing them into a single number.

Parallel requests also expose scheduling behavior. When ten requests arrive at
once, the server may admit all of them, queue some of them, or split them into
smaller batches. Time to first token then includes queueing delay as well as
the actual prefill computation, so it is an effective latency rather than a
pure hardware prefill rate. Decode speed measured after the first token is
closer to the model's steady-state generation rate, but it still includes any
time the request spends waiting between scheduled decode steps.

Prompt length changes the balance. Long prompts make prefill dominate and put
pressure on the attention mechanism, while short prompts make decode and
scheduling overhead dominate. Output length has a similar effect: a short
generation is mostly a prefill test, and a long generation is mostly a decode
test. For repeatable measurements, keep the prompt and output lengths fixed
across concurrency levels, warm up the server, and run several waves instead
of trusting a single request.

Caching is another source of misleading results. Many serving stacks keep a
prefix cache so that a repeated prompt can skip part of prefill. If every
request in a benchmark uses the same text, later requests may look much faster
than the first one. This tool therefore puts a unique nonce at the start of
each prompt by default. Use the opposite setting only when prefix-cache
behavior is the thing you intend to measure.

Finally, remember that the client is part of the measurement. Connection
setup, TLS handshakes, HTTP parsing, and Python scheduling all add overhead.
Reusing a connection pool and sending requests concurrently reduces that
overhead, but it does not remove it. Compare runs produced by the same tool on
the same machine, and treat small differences as noise rather than as a
reliable ranking of serving systems.
"""


class TokenCounter:
    """Token counter backed by tiktoken, with a character-based fallback."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.encoding_name = encoding
        self._encoder = None
        self._tried = False
        self.error: Optional[str] = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._encoder is not None

    @property
    def encoder(self):
        self._ensure()
        return self._encoder

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(self.encoding_name)
        except Exception as exc:  # pragma: no cover - depends on environment
            self.error = f"{type(exc).__name__}: {exc}"
            self._encoder = None

    def encode(self, text: str) -> List[int]:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("tiktoken encoding is unavailable")
        return encoder.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("tiktoken encoding is unavailable")
        return encoder.decode(token_ids)

    def count(self, text: str) -> int:
        encoder = self.encoder
        if encoder is not None:
            return len(encoder.encode(text))
        return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True)
class Prompt:
    """A generated prompt and its locally estimated token count."""

    text: str
    estimated_tokens: int


class PromptBuilder:
    """Build prompts of roughly ``target_tokens`` tokens.

    When ``unique`` is true, each prompt starts with a random nonce and starts
    at a different offset in the corpus. That prevents server-side prefix
    caching from making later requests unrealistically cheap.
    """

    def __init__(
        self,
        counter: TokenCounter,
        target_tokens: int,
        unique: bool = True,
    ) -> None:
        self.counter = counter
        self.target_tokens = max(1, int(target_tokens))
        self.unique = unique
        self._corpus_ids: Optional[List[int]] = None

    def _corpus_token_ids(self) -> List[int]:
        if self._corpus_ids is None:
            self._corpus_ids = self.counter.encode(CORPUS)
        return self._corpus_ids

    def build(self) -> Prompt:
        nonce = secrets.token_hex(4)
        prefix = f"[request {nonce}] " if self.unique else ""
        encoder = self.counter.encoder

        if encoder is None:
            prefix_tokens = self.counter.count(prefix)
            body_tokens = max(1, self.target_tokens - prefix_tokens)
            chars = body_tokens * 4
            body = (CORPUS * (chars // len(CORPUS) + 1))[:chars]
            return Prompt(text=prefix + body, estimated_tokens=self.target_tokens)

        corpus = self._corpus_token_ids()
        prefix_ids = encoder.encode(prefix)
        body_tokens = max(1, self.target_tokens - len(prefix_ids))
        offset = (int(nonce, 16) % len(corpus)) if self.unique else 0

        repeats = math.ceil((offset + body_tokens) / len(corpus))
        body_ids = (corpus * repeats)[offset : offset + body_tokens]
        token_ids = prefix_ids + body_ids

        return Prompt(text=encoder.decode(token_ids), estimated_tokens=len(token_ids))
