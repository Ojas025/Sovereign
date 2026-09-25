"""Streaming LLM client: OpenAI-compatible SSE against the fake llama-server."""

import json

import pytest

from fixtures.fake_llama_server import FakeBehavior, FakeLlamaServer
from workbench.core.protocols import (
    ChatMessage,
    ChatRequest,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    UsageReport,
)
from workbench.llm.client import LLMClientHttp, LLMError


def make_request(
    *,
    model: str = "qwen-coder",
    messages: tuple[ChatMessage, ...] | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=messages or (ChatMessage(role="user", content="hi"),),
        model=model,
    )


async def collect(client: LLMClientHttp, request: ChatRequest) -> list[object]:
    return [event async for event in client.stream(request)]


async def test_streams_text_deltas_then_stream_end_with_usage() -> None:
    with FakeLlamaServer(FakeBehavior()) as server:
        events = await collect(LLMClientHttp(server.base_url), make_request())

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "Hello from fake"
    assert all(
        e.text for e in events if isinstance(e, TextDelta)
    ), "role-only chunks must not surface as empty text deltas"
    assert any(isinstance(e, UsageReport) for e in events)

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.stop_reason == "stop"
    assert end.ttft_s is not None and end.ttft_s >= 0


async def test_ttft_reflects_time_to_first_text() -> None:
    behavior = FakeBehavior(chunk_delay_s=0.1)
    with FakeLlamaServer(behavior) as server:
        events = await collect(LLMClientHttp(server.base_url), make_request())

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.ttft_s is not None and end.ttft_s >= 0.1


async def test_streams_tool_call_with_arguments_assembled_by_call_id() -> None:
    with FakeLlamaServer(FakeBehavior(responses=("tool_call",))) as server:
        events = await collect(LLMClientHttp(server.base_url), make_request())

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert starts[0].name == "bash"
    assert starts[0].call_id == "call_1"

    arguments = "".join(
        e.arguments_delta for e in events if isinstance(e, ToolCallDelta) and e.call_id == "call_1"
    )
    assert json.loads(arguments) == {"command": "ls"}

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.stop_reason == "tool_calls"


async def test_sends_model_messages_and_tools_in_openai_shape() -> None:
    """Router-mode model switching and history replay depend on this payload."""
    messages = (
        ChatMessage(role="user", content="run ls"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="call_9", name="bash", arguments='{"command": "ls"}'),),
        ),
        ChatMessage(role="tool", content="file.txt", tool_call_id="call_9"),
    )
    request = ChatRequest(
        messages=messages,
        model="qwen-coder",
    )

    with FakeLlamaServer(FakeBehavior()) as server:
        await collect(LLMClientHttp(server.base_url), request)

    body = server.requests[-1].body
    assert body is not None
    assert body["model"] == "qwen-coder"
    assert body["stream"] is True

    sent = body["messages"]
    assert sent[1]["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command": "ls"}',
    }
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "call_9"


async def test_tools_are_serialized_when_provided() -> None:
    from workbench.core.protocols import ToolSpec

    request = ChatRequest(
        messages=(ChatMessage(role="user", content="hi"),),
        model="qwen-coder",
        tools=(ToolSpec(name="bash", description="run a command", parameters={"type": "object"}),),
    )

    with FakeLlamaServer(FakeBehavior()) as server:
        await collect(LLMClientHttp(server.base_url), request)

    body = server.requests[-1].body
    assert body is not None
    assert body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run a command",
            "parameters": {"type": "object"},
        },
    }


async def test_http_error_raises_llm_error() -> None:
    with FakeLlamaServer(FakeBehavior(responses=("error500",))) as server:
        with pytest.raises(LLMError, match="500"):
            await collect(LLMClientHttp(server.base_url), make_request())


async def test_reasoning_deltas_surface_before_content_and_arm_ttft() -> None:
    """Reasoning models stream 'reasoning_content'; it must not be dropped or
    merged into the answer, and ttft must reflect the first generated token."""
    from workbench.core.protocols import ReasoningDelta

    behavior = FakeBehavior(reasoning_chunks=("think", "ing"), chunk_delay_s=0.1)
    with FakeLlamaServer(behavior) as server:
        events = await collect(LLMClientHttp(server.base_url), make_request())

    reasoning = [e.text for e in events if isinstance(e, ReasoningDelta)]
    assert reasoning == ["think", "ing"]

    content = [e.text for e in events if isinstance(e, TextDelta)]
    assert content and "thinking" not in "".join(content), "reasoning must stay separate"

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.ttft_s is not None and end.ttft_s >= 0.1, "first reasoning token arms ttft"


async def test_connection_refused_raises_llm_error() -> None:
    with pytest.raises(LLMError):
        await collect(LLMClientHttp("http://127.0.0.1:1"), make_request())


async def test_scripted_text_response_streams_its_own_content() -> None:
    behavior = FakeBehavior(responses=("text:custom content 123",))
    with FakeLlamaServer(behavior) as server:
        events = await collect(LLMClientHttp(server.base_url), make_request())

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "custom content 123"
