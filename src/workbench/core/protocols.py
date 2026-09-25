"""Protocol interfaces for pluggable workbench components.

Every cross-module contract is declared once here; the plugin registry, the
agent loop, the TUI, and the metrics layer depend on these types only.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

# Async yes/no callback used by guarded tool operations (TUI prompt, headless policy).
ConfirmFn = Callable[[str], Awaitable[bool]]


# --- LLM chat -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model, replayed in message history."""

    id: str
    name: str
    arguments: str  # JSON string, accumulated from stream deltas


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    model: str
    tools: tuple[ToolSpec, ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


# --- streaming events (tagged union) ------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """One piece of the model's hidden thinking (kept separate from the answer)."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    call_id: str
    arguments_delta: str


@dataclass(frozen=True, slots=True)
class UsageReport:
    usage: Usage


@dataclass(frozen=True, slots=True)
class StreamEnd:
    stop_reason: Literal["stop", "length", "tool_calls", "error", "aborted"]
    ttft_s: float | None = None  # time to first text delta (display + metrics)


StreamEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | UsageReport
    | StreamEnd
)


# --- routing ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Classification:
    intent: str
    difficulty: float
    confidence: float
    needs_tools: bool
    backend: str


# --- tools & sandbox ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Execution context handed to every tool call."""

    workspace_root: Path
    confirm: ConfirmFn


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# --- protocols ----------------------------------------------------------


@runtime_checkable
class Router(Protocol):
    async def classify(self, message: str) -> Classification:
        """Classify one user message into intent/difficulty/needs_tools."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream one chat completion as tagged events."""
        ...


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: Mapping[str, object]

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        """Run the tool against validated arguments inside the sandbox."""
        ...


@runtime_checkable
class Sandbox(Protocol):
    def run(self, command: str, timeout_s: float) -> SandboxResult:
        """Execute a shell command under the sandbox backend."""
        ...


@runtime_checkable
class ModelProvider(Protocol):
    def download(self, spec: str, destination: Path) -> Path:
        """Fetch a model artifact into the local store and return its path."""
        ...
