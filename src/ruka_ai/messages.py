from typing import Annotated, Literal

from pydantic import Field

from .types import JSONValue, ModelPricing
from .utils import WireModel, current_timestamp_ms


# --- usage & cost ------------------------------------------------------------


class UsageCost(WireModel):
    """Response cost in USD, computed locally from ModelPricing."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    total: float = 0.0


class Usage(WireModel):
    """Token usage reported by the provider for one assistant response."""

    input: int = 0                # prompt tokens NOT served from cache
    output: int = 0               # completion tokens (includes reasoning tokens)
    cache_read: int = 0           # prompt tokens served from cache
    reasoning: int | None = None  # subset of output, when the provider reports it
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


def calculate_cost(usage: Usage, pricing: ModelPricing) -> UsageCost:
    """Price a Usage with per-million-token rates."""
    per_token = 1 / 1_000_000
    input_cost = usage.input * pricing.input * per_token
    output_cost = usage.output * pricing.output * per_token
    cache_read_rate = pricing.cache_read if pricing.cache_read is not None else pricing.input
    cache_read_cost = usage.cache_read * cache_read_rate * per_token
    return UsageCost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        total=input_cost + output_cost + cache_read_cost,
    )


class ResponseTiming(WireModel):
    time_to_first_output_ms: int | None = Field(default=None, ge=0)
    total_duration_ms: int = Field(ge=0)


# --- diagnostics -------------------------------------------------------------


class DiagnosticError(WireModel):
    name: str | None = None  # exception class, e.g. "HTTPStatusError"
    message: str
    code: str | int | None = None  # HTTP status or provider error code


class AssistantMessageDiagnostic(WireModel):
    """Something worth knowing about how a response was produced.

    Examples: a retry after a 429, tool-call arguments that weren't valid JSON,
    an unknown finish_reason. `type` is a free-form slug like "retry".
    """

    type: str
    timestamp: int = Field(default_factory=current_timestamp_ms)
    error: DiagnosticError | None = None
    details: dict[str, JSONValue] | None = None


# --- content blocks ----------------------------------------------------------


class TextContent(WireModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContent(WireModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolCall(WireModel):
    """A tool call content block requested by the assistant."""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)


type AssistantContent = Annotated[
    TextContent | ThinkingContent | ToolCall, Field(discriminator="type")
]
type UserContent = TextContent
type ToolResultContent = TextContent


# --- messages ----------------------------------------------------------------


StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


class UserMessage(WireModel):
    role: Literal["user"] = "user"
    content: list[UserContent]
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content)


class AssistantMessage(WireModel):
    """One model response: ordered content blocks plus response metadata."""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    model: str
    response_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    timing: ResponseTiming | None = None
    diagnostics: list[AssistantMessageDiagnostic] = Field(default_factory=list)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def thinking_text(self) -> str:
        return "".join(
            block.thinking for block in self.content if isinstance(block, ThinkingContent)
        )

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.content if isinstance(block, ToolCall))


class ToolResultMessage(WireModel):
    """The result of executing one ToolCall, sent back to the model."""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent] = Field(default_factory=list)
    is_error: bool = False
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content)


type Message = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]
