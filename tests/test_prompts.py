import pytest

from llmbench.prompts import PromptBuilder, TokenCounter


def test_prompt_hits_target_token_count():
    counter = TokenCounter("cl100k_base")
    if not counter.available:
        pytest.skip("tiktoken encoding is unavailable in this environment")

    builder = PromptBuilder(counter, 128, unique=True)
    prompt = builder.build()

    assert prompt.estimated_tokens == 128
    assert abs(counter.count(prompt.text) - 128) <= 8


def test_unique_prompts_differ():
    counter = TokenCounter("cl100k_base")
    if not counter.available:
        pytest.skip("tiktoken encoding is unavailable in this environment")

    builder = PromptBuilder(counter, 64, unique=True)
    assert builder.build().text != builder.build().text


def test_non_unique_prompts_are_identical():
    counter = TokenCounter("cl100k_base")
    if not counter.available:
        pytest.skip("tiktoken encoding is unavailable in this environment")

    builder = PromptBuilder(counter, 64, unique=False)
    assert builder.build().text == builder.build().text
