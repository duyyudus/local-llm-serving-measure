"""Async streaming client for OpenAI-compatible chat completions."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx

from .prompts import Prompt, TokenCounter


def normalize_endpoint(endpoint: str) -> str:
    """Turn a base URL into an OpenAI-compatible chat completions URL."""
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("endpoint must not be empty")

    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("endpoint must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("endpoint must include a host")

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        new_path = path
    else:
        if path in ("", "/"):
            path = "/v1"
        new_path = path + "/chat/completions"

    return urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))


class ApiError(Exception):
    """An HTTP error returned by the server."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        message = f"HTTP {status_code}: {body[:300]}" if status_code >= 400 else body[:300]
        super().__init__(message)


@dataclass
class RequestResult:
    """Timing and token accounting for one request."""

    ok: bool
    error: Optional[str]
    started_at: float
    first_token_at: Optional[float]
    last_token_at: Optional[float]
    ended_at: float
    ttft_s: Optional[float]
    decode_s: Optional[float]
    total_s: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    prefill_tps: Optional[float]
    decode_tps: Optional[float]
    usage_reported: bool
    streamed: bool
    text_chars: int

    @classmethod
    def error_result(cls, started_at: float, ended_at: float, error: str) -> "RequestResult":
        return cls(
            ok=False,
            error=error,
            started_at=started_at,
            first_token_at=None,
            last_token_at=None,
            ended_at=ended_at,
            ttft_s=None,
            decode_s=None,
            total_s=max(0.0, ended_at - started_at),
            prompt_tokens=None,
            completion_tokens=None,
            prefill_tps=None,
            decode_tps=None,
            usage_reported=False,
            streamed=False,
            text_chars=0,
        )

    def to_dict(self, wave_start: Optional[float] = None) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "start_offset_s": (
                self.started_at - wave_start if wave_start is not None else None
            ),
            "ttft_s": self.ttft_s,
            "decode_s": self.decode_s,
            "total_s": self.total_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "usage_reported": self.usage_reported,
            "streamed": self.streamed,
            "text_chars": self.text_chars,
        }


class ChatClient:
    """A pooled async client that measures streaming chat completions."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        verify: bool = True,
        extra_body: Optional[Dict[str, Any]] = None,
        ignore_eos: bool = True,
        include_usage: bool = True,
        seed: Optional[int] = None,
        counter: Optional[TokenCounter] = None,
        max_connections: int = 4,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.url = normalize_endpoint(endpoint)
        self.model = model
        self.api_key = api_key
        self.counter = counter or TokenCounter()
        self.extra_body = dict(extra_body or {})
        self.use_ignore_eos = ignore_eos
        self.use_stream_options = include_usage
        self.use_seed = seed is not None
        self.disabled_params: List[str] = []
        self._lock = asyncio.Lock()

        pool = max(1, int(max_connections))
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(30.0, timeout)),
            limits=httpx.Limits(
                max_connections=pool,
                max_keepalive_connections=pool,
            ),
            verify=verify,
            transport=transport,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    def _body(
        self,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if self.use_stream_options:
            body["stream_options"] = {"include_usage": True}
        if self.use_ignore_eos:
            body["ignore_eos"] = True
        if seed is not None and self.use_seed:
            body["seed"] = seed
        body.update(self.extra_body)
        return body

    async def stream(
        self,
        prompt: Prompt,
        max_tokens: int,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> RequestResult:
        """Send one streaming request and return its measurements.

        A 400 response caused by an unsupported optional parameter triggers one
        retry with that parameter disabled. The original start timestamp is
        kept so the retry cost is reflected in the measured latency.
        """
        start = time.perf_counter()
        attempts = 0

        while True:
            body = self._body(prompt, max_tokens, temperature, seed)
            try:
                return await self._stream_once(prompt, body, start)
            except ApiError as exc:
                if exc.status_code == 400 and attempts < 3:
                    field = await self._disable_optional_for(body, exc.body)
                    if field is not None:
                        attempts += 1
                        continue
                ended = time.perf_counter()
                return RequestResult.error_result(start, ended, str(exc))
            except httpx.HTTPError as exc:
                ended = time.perf_counter()
                return RequestResult.error_result(
                    start, ended, f"{type(exc).__name__}: {exc}"
                )
            except Exception as exc:  # defensive: never let one request kill a wave
                ended = time.perf_counter()
                return RequestResult.error_result(
                    start, ended, f"{type(exc).__name__}: {exc}"
                )

    async def _disable_optional_for(
        self,
        body: Dict[str, Any],
        error_text: str,
    ) -> Optional[str]:
        """Disable one optional parameter named by the server's error.

        A parameter is only disabled when it was present in the failed request
        body and the error text explicitly names it. Context-length errors are
        never treated as unsupported parameters.
        """
        lower = error_text.lower()
        context_markers = (
            "context length",
            "maximum context",
            "context window",
            "context size",
            "available context",
            "exceeds the available context",
            "too many tokens",
            "prompt is too long",
            "input is too long",
            "reduce the length",
        )
        if any(marker in lower for marker in context_markers):
            return None

        candidates = []
        if "ignore_eos" in body:
            candidates.append(("ignore_eos", "use_ignore_eos"))
        if "stream_options" in body:
            candidates.append(("stream_options", "use_stream_options"))
        if "seed" in body:
            candidates.append(("seed", "use_seed"))

        async with self._lock:
            for name, attribute in candidates:
                if name in lower:
                    changed = False
                    if getattr(self, attribute):
                        setattr(self, attribute, False)
                        changed = True
                    if name in self.extra_body:
                        self.extra_body.pop(name, None)
                        changed = True
                    if changed and name not in self.disabled_params:
                        self.disabled_params.append(name)
                    return name
        return None

    async def _stream_once(
        self,
        prompt: Prompt,
        body: Dict[str, Any],
        start: float,
    ) -> RequestResult:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self.client.stream(
            "POST", self.url, json=body, headers=headers
        ) as response:
            if response.status_code >= 400:
                raw = await response.aread()
                raise ApiError(
                    response.status_code,
                    raw.decode("utf-8", "replace")[:2000],
                )

            content_type = response.headers.get("content-type", "").lower()
            if "application/json" in content_type and "text/event-stream" not in content_type:
                raw = await response.aread()
                ended = time.perf_counter()
                return self._non_stream_result(prompt, raw, start, ended)

            return await self._stream_result(prompt, response, start)

    async def _stream_result(
        self,
        prompt: Prompt,
        response: httpx.Response,
        start: float,
    ) -> RequestResult:
        first: Optional[float] = None
        last: Optional[float] = None
        parts: List[str] = []
        usage: Optional[Dict[str, Any]] = None

        async for line in response.aiter_lines():
            if not line:
                continue

            payload: Optional[str] = None
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
            elif line.startswith("{"):
                payload = line.strip()

            if not payload:
                continue

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue

            if chunk.get("error"):
                raise ApiError(200, json.dumps(chunk["error"]))

            if chunk.get("usage"):
                usage = chunk["usage"]

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                if isinstance(piece, str) and piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    parts.append(piece)

        ended = time.perf_counter()
        if first is None:
            first = ended
        if last is None:
            last = ended

        return self._result(
            prompt=prompt,
            start=start,
            first=first,
            last=last,
            ended=ended,
            parts=parts,
            usage=usage,
            streamed=True,
        )

    def _non_stream_result(
        self,
        prompt: Prompt,
        raw: bytes,
        start: float,
        ended: float,
    ) -> RequestResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return RequestResult.error_result(
                start, ended, "server returned a non-JSON response"
            )
        if not isinstance(data, dict):
            return RequestResult.error_result(
                start, ended, "server returned an unexpected JSON payload"
            )
        if data.get("error"):
            return RequestResult.error_result(
                start, ended, json.dumps(data["error"])
            )

        parts: List[str] = []
        for choice in data.get("choices") or []:
            message = choice.get("message") or {}
            piece = (
                message.get("content")
                or message.get("reasoning_content")
                or message.get("reasoning")
                or choice.get("text")
                or ""
            )
            if isinstance(piece, str) and piece:
                parts.append(piece)

        return self._result(
            prompt=prompt,
            start=start,
            first=ended,
            last=ended,
            ended=ended,
            parts=parts,
            usage=data.get("usage"),
            streamed=False,
        )

    def _result(
        self,
        *,
        prompt: Prompt,
        start: float,
        first: float,
        last: float,
        ended: float,
        parts: List[str],
        usage: Optional[Dict[str, Any]],
        streamed: bool,
    ) -> RequestResult:
        text = "".join(parts)
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        usage_reported = False

        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
                usage_reported = True
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
                usage_reported = True

        if prompt_tokens is None:
            prompt_tokens = prompt.estimated_tokens
        if completion_tokens is None:
            completion_tokens = self.counter.count(text) if text else 0

        ttft = max(0.0, first - start)
        total = max(0.0, ended - start)
        decode_s = (last - first) if last > first else None
        decode_tps = None
        if completion_tokens >= 2 and decode_s is not None and decode_s > 0:
            decode_tps = (completion_tokens - 1) / decode_s
        prefill_tps = prompt_tokens / ttft if ttft > 0 else None

        if completion_tokens == 0 and not text:
            return RequestResult.error_result(start, ended, "server returned no tokens")

        return RequestResult(
            ok=True,
            error=None,
            started_at=start,
            first_token_at=first,
            last_token_at=last,
            ended_at=ended,
            ttft_s=ttft,
            decode_s=decode_s,
            total_s=total,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            usage_reported=usage_reported,
            streamed=streamed,
            text_chars=len(text),
        )
