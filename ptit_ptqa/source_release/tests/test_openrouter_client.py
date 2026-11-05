from __future__ import annotations

import json
import sqlite3

from src.llm.openrouter_client import OpenRouterClient, OpenRouterConfig


class FakeResponse:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def successful_payload(query_id=1):
    value = {"query_id": query_id, "value": "ok"}
    return {
        "id": "gen-1",
        "model": "qwen/qwen3-14b",
        "provider": "DeepInfra",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_client(tmp_path, session, **overrides):
    values = {
        "cache_path": tmp_path / "cache.sqlite",
        "max_retries": 2,
        "backoff_base_seconds": 0,
    }
    values.update(overrides)
    config = OpenRouterConfig(**values)
    return OpenRouterClient(
        config,
        api_key="test-key",
        session=session,
        sleep_fn=lambda _: None,
        random_fn=lambda: 0,
    )


def test_cache_resumes_without_second_api_call(tmp_path):
    session = FakeSession([FakeResponse(200, successful_payload())])
    client = make_client(tmp_path, session)
    kwargs = {
        "query_id": 1,
        "messages": [{"role": "user", "content": "test"}],
        "json_schema": {
            "type": "object",
            "properties": {"query_id": {"type": "integer"}, "value": {"type": "string"}},
            "required": ["query_id", "value"],
        },
        "validator": lambda value: [] if value["query_id"] == 1 else ["wrong id"],
    }
    first = client.complete_json(**kwargs)
    second = client.complete_json(**kwargs)

    assert first.status == "ok"
    assert not first.cached
    assert second.cached
    assert len(session.calls) == 1


def test_cached_payload_is_revalidated_with_current_semantic_guard(tmp_path):
    session = FakeSession([FakeResponse(200, successful_payload())])
    client = make_client(tmp_path, session)
    kwargs = {
        "query_id": 1,
        "messages": [{"role": "user", "content": "test"}],
        "json_schema": {"type": "object"},
    }
    assert client.complete_json(**kwargs).status == "ok"
    cached = client.complete_json(**kwargs, validator=lambda _: ["new semantic error"])
    assert cached.cached
    assert cached.status == "invalid"
    assert cached.validation_errors == ["new semantic error"]
    assert len(session.calls) == 1


def test_retries_429_and_persists_safe_provenance(tmp_path):
    session = FakeSession(
        [
            FakeResponse(429, text="rate limited", headers={"Retry-After": "0"}),
            FakeResponse(200, successful_payload()),
        ]
    )
    client = make_client(tmp_path, session)
    result = client.complete_json(
        query_id=1,
        messages=[{"role": "user", "content": "test"}],
        json_schema={"type": "object"},
    )

    assert result.status == "ok"
    assert result.retry_count == 1
    body = session.calls[0][1]["json"]
    assert body["model"] == "qwen/qwen3-14b"
    assert body["repetition_penalty"] == 1.08
    assert body["provider"] == {
        "only": ["deepinfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "zdr": True,
        "data_collection": "deny",
    }
    with sqlite3.connect(tmp_path / "cache.sqlite") as connection:
        request_json = connection.execute("SELECT request_json FROM generations").fetchone()[0]
    assert "test-key" not in request_json
    assert "Authorization" not in request_json


def test_validation_error_is_cached_as_invalid(tmp_path):
    session = FakeSession([FakeResponse(200, successful_payload(query_id=99))])
    client = make_client(tmp_path, session)
    result = client.complete_json(
        query_id=1,
        messages=[{"role": "user", "content": "test"}],
        json_schema={"type": "object"},
        validator=lambda value: ["wrong query_id"] if value["query_id"] != 1 else [],
    )
    assert result.status == "invalid"
    assert result.validation_errors == ["wrong query_id"]


def test_http_200_error_envelope_is_not_treated_as_model_output(tmp_path):
    payload = {
        "id": "gen-error",
        "error": {"code": 400, "message": "Upstream schema is unsupported"},
    }
    session = FakeSession([FakeResponse(200, payload)])
    client = make_client(tmp_path, session)
    result = client.complete_json(
        query_id=1,
        messages=[{"role": "user", "content": "test"}],
        json_schema={"type": "object"},
    )
    assert result.status == "error"
    assert result.http_status == 400
    assert "unsupported" in result.error
