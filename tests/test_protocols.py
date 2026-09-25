"""Protocol contracts every pluggable component must satisfy."""

from collections.abc import AsyncIterator
from typing import Any

from workbench.core.protocols import (
    LLMClient,
    ModelProvider,
    Router,
    Sandbox,
    Tool,
)


class FakeRouter:
    async def classify(self, message: str) -> Any:
        return None


class FakeLLMClient:
    def stream(self, request: Any) -> AsyncIterator[Any]:
        async def empty() -> AsyncIterator[Any]:
            yield None

        return empty()


class FakeTool:
    name = "fake"
    description = "fake tool"
    parameters: dict[str, object] = {}

    async def execute(self, arguments: Any, context: Any) -> Any:
        return None


class FakeSandbox:
    def run(self, command: str, timeout_s: float) -> Any:
        return None


class FakeProvider:
    def download(self, spec: str, destination: Any) -> Any:
        return None


def test_protocol_satisfaction() -> None:
    assert isinstance(FakeRouter(), Router)
    assert isinstance(FakeLLMClient(), LLMClient)
    assert isinstance(FakeTool(), Tool)
    assert isinstance(FakeSandbox(), Sandbox)
    assert isinstance(FakeProvider(), ModelProvider)


def test_missing_methods_fail_protocol_check() -> None:
    class NotARouter:
        pass

    assert not isinstance(NotARouter(), Router)
    assert not isinstance(NotARouter(), Tool)


def test_sandbox_signature_contract() -> None:
    """Sandbox.run is synchronous and takes a command plus timeout in seconds."""

    import inspect

    params = inspect.signature(FakeSandbox.run).parameters
    assert list(params) == ["self", "command", "timeout_s"]
    assert inspect.iscoroutinefunction(FakeSandbox.run) is False


def test_tool_and_router_are_async() -> None:
    import inspect

    assert inspect.iscoroutinefunction(FakeTool.execute)
    assert inspect.iscoroutinefunction(FakeRouter.classify)


def test_llm_client_stream_returns_async_iterator() -> None:
    stream = FakeLLMClient().stream(object())
    assert isinstance(stream, AsyncIterator)
