import asyncio
import json

import httpx
import pytest

from llmbench.client import ChatClient
from llmbench.prompts import Prompt


class ChunkStream(httpx.AsyncByteStream):
    """Yield SSE lines with a small delay so decode timing is measurable."""

    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(0.001)
            yield chunk


def _stream_response(chunks):
    lines = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
    lines.append(b"data: [DONE]\n\n")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=ChunkStream(lines),
    )


def _success_chunks():
    return [
        {"choices": [{"index": 0, "delta": {"content": "hello "}}]},
        {"choices": [{"index": 0, "delta": {"content": "world"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 42, "completion_tokens": 2, "total_tokens": 44},
        },
    ]


def test_stream_parses_usage_and_timings():
    def handler(request):
        return _stream_response(_success_chunks())

    async def run():
        client = ChatClient(
            "http://test/v1", "model", transport=httpx.MockTransport(handler)
        )
        try:
            return await client.stream(Prompt("hi", 10), max_tokens=2)
        finally:
            await client.aclose()

    result = asyncio.run(run())

    assert result.ok
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 2
    assert result.text_chars == len("hello world")
    assert result.ttft_s is not None and result.ttft_s > 0
    assert result.decode_s is not None and result.decode_s > 0
    assert result.decode_tps is not None and result.decode_tps > 0
    assert result.usage_reported is True
    assert result.streamed is True


def test_only_named_optional_param_is_disabled():
    def handler(request):
        body = json.loads(request.content)
        if "ignore_eos" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unrecognized request argument supplied: ignore_eos"
                    }
                },
            )
        return _stream_response(_success_chunks())

    async def run():
        client = ChatClient(
            "http://test/v1",
            "model",
            transport=httpx.MockTransport(handler),
            seed=123,
        )
        try:
            results = await asyncio.gather(
                *(
                    client.stream(Prompt(f"hi {index}", 10), max_tokens=2, seed=123)
                    for index in range(3)
                )
            )
        finally:
            await client.aclose()
        return client, results

    client, results = asyncio.run(run())

    assert all(result.ok for result in results)
    assert set(client.disabled_params) == {"ignore_eos"}
    assert client.use_stream_options is True
    assert client.use_seed is True


def test_context_error_does_not_disable_params():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "the request exceeds the available context size"
                }
            },
        )

    async def run():
        client = ChatClient(
            "http://test/v1",
            "model",
            transport=httpx.MockTransport(handler),
            seed=123,
        )
        try:
            result = await client.stream(Prompt("hi", 10), max_tokens=2, seed=123)
        finally:
            await client.aclose()
        return client, result

    client, result = asyncio.run(run())

    assert not result.ok
    assert client.disabled_params == []
    assert client.use_ignore_eos is True
    assert client.use_stream_options is True
    assert client.use_seed is True


def test_generic_400_does_not_disable_params():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async def run():
        client = ChatClient(
            "http://test/v1",
            "model",
            transport=httpx.MockTransport(handler),
            seed=123,
        )
        try:
            result = await client.stream(Prompt("hi", 10), max_tokens=2, seed=123)
        finally:
            await client.aclose()
        return client, result

    client, result = asyncio.run(run())

    assert not result.ok
    assert client.disabled_params == []


def test_extra_body_parameter_removal_is_reported():
    def handler(request):
        body = json.loads(request.content)
        if "seed" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unrecognized request argument supplied: seed"
                    }
                },
            )
        return _stream_response(_success_chunks())

    async def run():
        client = ChatClient(
            "http://test/v1",
            "model",
            transport=httpx.MockTransport(handler),
            extra_body={"seed": 1},
        )
        try:
            result = await client.stream(Prompt("hi", 10), max_tokens=2)
        finally:
            await client.aclose()
        return client, result

    client, result = asyncio.run(run())

    assert result.ok
    assert client.disabled_params == ["seed"]
    assert "seed" not in client.extra_body


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_delta_counts_as_token(field):
    chunks = [
        {"choices": [{"delta": {field: "think "}}]},
        {"choices": [{"delta": {"content": "answer"}}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
    ]

    def handler(request):
        return _stream_response(chunks)

    async def run():
        client = ChatClient(
            "http://test/v1", "model", transport=httpx.MockTransport(handler)
        )
        try:
            return await client.stream(Prompt("hi", 10), max_tokens=2)
        finally:
            await client.aclose()

    result = asyncio.run(run())

    assert result.ok
    assert result.text_chars == len("think answer")
    assert result.decode_tps is not None
