"""Streaming client for llama-server's OpenAI-compatible endpoint.

The request ``model`` field is how router mode switches models per request —
the client never caches or rewrites it. Transport failures surface as
``LLMError``; a clean stream always terminates with ``StreamEnd``.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Literal

import httpx

from workbench.core.protocols import (
    ChatMessage,
    ChatRequest,
    ReasoningDelta,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    Usage,
    UsageReport,
)
from workbench.logging_setup import get_logger

logger = get_logger("llm")

_CONNECT_TIMEOUT_S = 5.0
_READ_TIMEOUT_S = 120.0

_FINISH_REASONS: dict[str, Literal["stop", "length", "tool_calls", "error", "aborted"]] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "aborted": "aborted",
}


class LLMError(Exception):
    """Transport or protocol failure talking to llama-server."""


def _payload(request: ChatRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": request.model,
        "messages": [_message_dict(message) for message in request.messages],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in request.tools
        ]
    return payload


def _message_dict(message: ChatMessage) -> dict[str, object]:
    data: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return data


class LLMClientHttp:
    """``LLMClient`` implementation over HTTP+SSE (one HTTP call per stream)."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = _payload(request)
        timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT_S,
            read=_READ_TIMEOUT_S,
            write=_CONNECT_TIMEOUT_S,
            pool=_CONNECT_TIMEOUT_S,
        )
        started = time.monotonic()
        finish_reason: str | None = None
        ttft: float | None = None
        saw_done = False
        call_ids: dict[int, str] = {}

        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout) as http:
                async with http.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            saw_done = True
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise LLMError(f"malformed stream chunk: {data!r}") from exc

                        for event in self._events_from_chunk(chunk, call_ids):
                            if (
                                isinstance(event, (TextDelta, ReasoningDelta))
                                and ttft is None
                            ):
                                ttft = time.monotonic() - started
                            yield event

                        for choice in chunk.get("choices") or []:
                            reason = choice.get("finish_reason")
                            if reason:
                                finish_reason = reason
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"llama-server returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"llama-server request failed: {exc}") from exc

        if not saw_done and finish_reason is None:
            raise LLMError("stream ended without a completion marker")
        stop: Literal["stop", "length", "tool_calls", "error", "aborted"]
        if finish_reason is None:
            stop = "stop"
        else:
            stop = _FINISH_REASONS.get(finish_reason, "error")
        yield StreamEnd(stop_reason=stop, ttft_s=ttft)

    def _events_from_chunk(
        self, chunk: dict[str, object], call_ids: dict[int, str]
    ) -> list[StreamEvent]:
        """Translate one OpenAI chunk into zero or more tagged stream events."""
        events: list[StreamEvent] = []

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            events.append(
                UsageReport(
                    Usage(
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                    )
                )
            )

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return events
        for choice in choices:
            delta = (choice or {}).get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if reasoning:
                events.append(ReasoningDelta(text=reasoning))
            content = delta.get("content")
            if content:
                events.append(TextDelta(text=content))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                function = call.get("function") or {}
                if call.get("id") or function.get("name"):
                    call_id = str(call.get("id") or f"call_{index}")
                    call_ids[index] = call_id
                    events.append(
                        ToolCallStart(call_id=call_id, name=str(function.get("name", "")))
                    )
                arguments = function.get("arguments")
                if arguments:
                    mapped = call_ids.get(index)
                    if mapped is not None:
                        events.append(ToolCallDelta(call_id=mapped, arguments_delta=arguments))
        return events
