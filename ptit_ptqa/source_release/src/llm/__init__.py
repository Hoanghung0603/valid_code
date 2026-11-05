"""LLM clients and structured query-understanding primitives."""

from .openrouter_client import OpenRouterClient, OpenRouterConfig, OpenRouterResult
from .query_spec import QuerySpec, build_query_spec_messages, query_spec_json_schema

__all__ = [
    "OpenRouterClient",
    "OpenRouterConfig",
    "OpenRouterResult",
    "QuerySpec",
    "build_query_spec_messages",
    "query_spec_json_schema",
]
