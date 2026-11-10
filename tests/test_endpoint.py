import pytest

from llmbench.client import normalize_endpoint


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
        (
            "http://localhost:8000/v1/chat/completions",
            "http://localhost:8000/v1/chat/completions",
        ),
        ("https://host/api/v1", "https://host/api/v1/chat/completions"),
    ],
)
def test_normalize_endpoint(raw, expected):
    assert normalize_endpoint(raw) == expected


@pytest.mark.parametrize("raw", ["", "localhost:8000", "ftp://host/v1"])
def test_normalize_endpoint_rejects_invalid(raw):
    with pytest.raises(ValueError):
        normalize_endpoint(raw)
