"""Resumable OpenRouter JSON client with durable provenance.

The API key is read only from the environment. Request bodies persisted to the
SQLite cache never contain authorization headers or other credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json_text(content: Any) -> tuple[dict[str, Any] | None, str, str | None]:
    if isinstance(content, dict):
        return content, _canonical_json(content), None
    if not isinstance(content, str):
        return None, "", f"assistant content has unsupported type {type(content).__name__}"

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, text, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, text, "structured response must be a JSON object"
    return parsed, text, None


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str = "qwen/qwen3-14b"
    providers: tuple[str, ...] = ("deepinfra",)
    endpoint: str = DEFAULT_ENDPOINT
    cache_path: Path = Path("data/cache/openrouter_queryspec_v1.sqlite")
    prompt_version: str = "queryspec_prompt_v1"
    schema_version: str = "queryspec_schema_v1"
    temperature: float = 0.1
    top_p: float = 0.8
    repetition_penalty: float = 1.08
    max_completion_tokens: int = 1800
    seed: int = 20261023
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 180.0
    require_zdr: bool = True
    deny_data_collection: bool = True
    allow_provider_fallbacks: bool = False
    require_parameters: bool = True
    app_title: str = "R2AI Stage 3 QuerySpec"
    app_url: str = "https://github.com/aigurutinix/r2ai-stage-2"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, root: Path | None = None) -> "OpenRouterConfig":
        data = dict(value)
        data["providers"] = tuple(data.get("providers") or ())
        cache_path = Path(data.get("cache_path", cls.cache_path))
        if root is not None and not cache_path.is_absolute():
            cache_path = root / cache_path
        data["cache_path"] = cache_path
        return cls(**data)


@dataclass
class OpenRouterResult:
    request_hash: str
    query_id: str
    status: str
    parsed: dict[str, Any] | None
    raw_content: str
    validation_errors: list[str] = field(default_factory=list)
    cached: bool = False
    model_requested: str = ""
    model_returned: str = ""
    provider_requested: str = ""
    provider_actual: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    input_hash: str = ""
    response_hash: str = ""
    generation_id: str = ""
    http_status: int | None = None
    retry_count: int = 0
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str = ""
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "query_id": self.query_id,
            "status": self.status,
            "parsed": self.parsed,
            "raw_content": self.raw_content,
            "validation_errors": self.validation_errors,
            "cached": self.cached,
            "provenance": {
                "model_requested": self.model_requested,
                "model_returned": self.model_returned,
                "provider_requested": self.provider_requested,
                "provider_actual": self.provider_actual,
                "prompt_version": self.prompt_version,
                "schema_version": self.schema_version,
                "input_hash": self.input_hash,
                "response_hash": self.response_hash,
                "generation_id": self.generation_id,
                "http_status": self.http_status,
                "retry_count": self.retry_count,
                "latency_ms": self.latency_ms,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "finish_reason": self.finish_reason,
                "error": self.error,
            },
        }


class SQLiteResponseCache:
    """Small process-safe cache; each operation owns its SQLite connection."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    request_hash TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model_requested TEXT NOT NULL,
                    model_returned TEXT,
                    provider_requested TEXT,
                    provider_actual TEXT,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    response_hash TEXT,
                    generation_id TEXT,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    parsed_json TEXT,
                    raw_content TEXT,
                    validation_errors_json TEXT,
                    http_status INTEGER,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    finish_reason TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generations_query_id ON generations(query_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generations_status ON generations(status)"
            )

    def get(self, request_hash: str) -> OpenRouterResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generations WHERE request_hash = ?", (request_hash,)
            ).fetchone()
        if row is None:
            return None
        parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else None
        errors = json.loads(row["validation_errors_json"] or "[]")
        return OpenRouterResult(
            request_hash=row["request_hash"],
            query_id=row["query_id"],
            status=row["status"],
            parsed=parsed,
            raw_content=row["raw_content"] or "",
            validation_errors=errors,
            cached=True,
            model_requested=row["model_requested"],
            model_returned=row["model_returned"] or "",
            provider_requested=row["provider_requested"] or "",
            provider_actual=row["provider_actual"] or "",
            prompt_version=row["prompt_version"],
            schema_version=row["schema_version"],
            input_hash=row["input_hash"],
            response_hash=row["response_hash"] or "",
            generation_id=row["generation_id"] or "",
            http_status=row["http_status"],
            retry_count=row["retry_count"],
            latency_ms=row["latency_ms"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            finish_reason=row["finish_reason"] or "",
            error=row["error"] or "",
        )

    def put(
        self,
        result: OpenRouterResult,
        *,
        request_body: Mapping[str, Any],
        response_body: Mapping[str, Any] | None,
    ) -> None:
        now = _utc_now()
        values = {
            "request_hash": result.request_hash,
            "query_id": result.query_id,
            "status": result.status,
            "created_at": now,
            "updated_at": now,
            "model_requested": result.model_requested,
            "model_returned": result.model_returned,
            "provider_requested": result.provider_requested,
            "provider_actual": result.provider_actual,
            "prompt_version": result.prompt_version,
            "schema_version": result.schema_version,
            "input_hash": result.input_hash,
            "response_hash": result.response_hash,
            "generation_id": result.generation_id,
            "request_json": _canonical_json(request_body),
            "response_json": _canonical_json(response_body) if response_body is not None else None,
            "parsed_json": _canonical_json(result.parsed) if result.parsed is not None else None,
            "raw_content": result.raw_content,
            "validation_errors_json": _canonical_json(result.validation_errors),
            "http_status": result.http_status,
            "retry_count": result.retry_count,
            "latency_ms": result.latency_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "finish_reason": result.finish_reason,
            "error": result.error,
        }
        columns = list(values)
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(
            f"{column}=excluded.{column}" for column in columns if column not in {"request_hash", "created_at"}
        )
        sql = (
            f"INSERT INTO generations ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(request_hash) DO UPDATE SET {updates}"
        )
        with self._connect() as connection:
            connection.execute(sql, [values[column] for column in columns])


Validator = Callable[[dict[str, Any]], Sequence[str]]


class OpenRouterClient:
    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        api_key: str | None = None,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ):
        self.config = config
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not config.model.startswith("qwen/"):
            raise ValueError("Competition safety guard: only an explicitly pinned Qwen model is allowed")
        if not config.providers:
            raise ValueError("At least one provider must be pinned")
        self._injected_session = session
        self._thread_local = threading.local()
        self.sleep_fn = sleep_fn
        self.random_fn = random_fn
        self.cache = SQLiteResponseCache(config.cache_path)

    def _provider_preferences(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "only": list(self.config.providers),
            "allow_fallbacks": self.config.allow_provider_fallbacks,
            "require_parameters": self.config.require_parameters,
        }
        if self.config.require_zdr:
            result["zdr"] = True
        if self.config.deny_data_collection:
            result["data_collection"] = "deny"
        return result

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
            "HTTP-Referer": self.config.app_url,
            "X-OpenRouter-Title": self.config.app_title,
        }

    def _session(self) -> requests.Session:
        if self._injected_session is not None:
            return self._injected_session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _request_body(
        self,
        *,
        query_id: str,
        messages: Sequence[Mapping[str, Any]],
        json_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "max_completion_tokens": self.config.max_completion_tokens,
            "seed": self.config.seed,
            "stream": False,
            "provider": self._provider_preferences(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "r2ai_query_spec",
                    "strict": True,
                    "schema": dict(json_schema),
                },
            },
            "metadata": {
                "query_id": str(query_id),
                "prompt_version": self.config.prompt_version,
                "schema_version": self.config.schema_version,
            },
        }

    def complete_json(
        self,
        *,
        query_id: str | int,
        messages: Sequence[Mapping[str, Any]],
        json_schema: Mapping[str, Any],
        validator: Validator | None = None,
        force: bool = False,
    ) -> OpenRouterResult:
        query_id_text = str(query_id)
        body = self._request_body(query_id=query_id_text, messages=messages, json_schema=json_schema)
        request_material = {
            "body": body,
            "prompt_version": self.config.prompt_version,
            "schema_version": self.config.schema_version,
        }
        request_hash = _sha256_text(_canonical_json(request_material))
        input_hash = _sha256_text(_canonical_json(messages))
        if not force:
            cached = self.cache.get(request_hash)
            if cached is not None and cached.status in {"ok", "invalid"}:
                if cached.parsed is not None and validator is not None:
                    cached.validation_errors = [str(error) for error in validator(cached.parsed)]
                    cached.status = "ok" if not cached.validation_errors else "invalid"
                    cached.error = "; ".join(cached.validation_errors)
                return cached

        started = time.perf_counter()
        response_body: dict[str, Any] | None = None
        last_error = ""
        last_status: int | None = None
        retries_used = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._session().post(
                    self.config.endpoint,
                    headers=self._headers(),
                    json=body,
                    timeout=(self.config.connect_timeout_seconds, self.config.read_timeout_seconds),
                )
                last_status = int(response.status_code)
                if response.status_code == 200:
                    response_body = response.json()
                    break
                error_text = response.text[:2000]
                last_error = f"HTTP {response.status_code}: {error_text}"
                if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= self.config.max_retries:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 0.0
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.config.max_retries:
                    break
                delay = 0.0
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break

            retries_used += 1
            exponential = self.config.backoff_base_seconds * (2**attempt)
            delay = max(delay, exponential + self.random_fn() * exponential * 0.25)
            self.sleep_fn(delay)

        latency_ms = int((time.perf_counter() - started) * 1000)
        provider_requested = ",".join(self.config.providers)
        if response_body is None:
            result = OpenRouterResult(
                request_hash=request_hash,
                query_id=query_id_text,
                status="error",
                parsed=None,
                raw_content="",
                model_requested=self.config.model,
                provider_requested=provider_requested,
                prompt_version=self.config.prompt_version,
                schema_version=self.config.schema_version,
                input_hash=input_hash,
                http_status=last_status,
                retry_count=retries_used,
                latency_ms=latency_ms,
                error=last_error or "request failed without a response",
            )
            self.cache.put(result, request_body=body, response_body=None)
            return result

        api_error = response_body.get("error")
        if api_error:
            if isinstance(api_error, Mapping):
                api_error_code = api_error.get("code")
                api_error_message = str(api_error.get("message") or api_error)
            else:
                api_error_code = None
                api_error_message = str(api_error)
            try:
                effective_status = int(api_error_code)
            except (TypeError, ValueError):
                effective_status = last_status
            result = OpenRouterResult(
                request_hash=request_hash,
                query_id=query_id_text,
                status="error",
                parsed=None,
                raw_content="",
                model_requested=self.config.model,
                provider_requested=provider_requested,
                prompt_version=self.config.prompt_version,
                schema_version=self.config.schema_version,
                input_hash=input_hash,
                response_hash=_sha256_text(_canonical_json(response_body)),
                generation_id=str(response_body.get("id") or ""),
                http_status=effective_status,
                retry_count=retries_used,
                latency_ms=latency_ms,
                error=f"OpenRouter API error: {api_error_message}",
            )
            self.cache.put(result, request_body=body, response_body=response_body)
            return result

        choices = response_body.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        parsed, raw_content, parse_error = _extract_json_text(message.get("content"))
        validation_errors: list[str] = []
        if parse_error:
            validation_errors.append(parse_error)
        if parsed is not None and validator is not None:
            validation_errors.extend(str(error) for error in validator(parsed))

        usage = response_body.get("usage") or {}
        metadata = response_body.get("openrouter_metadata") or {}
        provider_actual = str(
            response_body.get("provider")
            or metadata.get("provider_name")
            or metadata.get("provider")
            or ""
        )
        model_returned = str(response_body.get("model") or "")
        if model_returned and model_returned != self.config.model:
            validation_errors.append(
                f"model mismatch: requested {self.config.model}, received {model_returned}"
            )
        allowed_providers = {provider.casefold() for provider in self.config.providers}
        if provider_actual and provider_actual.casefold() not in allowed_providers:
            validation_errors.append(
                f"provider mismatch: expected one of {sorted(self.config.providers)}, received {provider_actual}"
            )
        response_hash = _sha256_text(_canonical_json(response_body))
        result = OpenRouterResult(
            request_hash=request_hash,
            query_id=query_id_text,
            status="ok" if parsed is not None and not validation_errors else "invalid",
            parsed=parsed,
            raw_content=raw_content,
            validation_errors=validation_errors,
            model_requested=self.config.model,
            model_returned=model_returned,
            provider_requested=provider_requested,
            provider_actual=provider_actual,
            prompt_version=self.config.prompt_version,
            schema_version=self.config.schema_version,
            input_hash=input_hash,
            response_hash=response_hash,
            generation_id=str(response_body.get("id") or ""),
            http_status=last_status,
            retry_count=retries_used,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=str(choice.get("finish_reason") or ""),
            error="; ".join(validation_errors),
        )
        self.cache.put(result, request_body=body, response_body=response_body)
        return result
