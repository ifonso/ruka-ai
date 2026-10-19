"""Events a provider yields while streaming one assistant response.

Every event carries `partial`, the AssistantMessage built so far, so a UI can
just re-render it. `content_index` points at the block inside partial.content.
"""

from typing import Annotated, Literal

from pydantic import Field

from .messages import AssistantMessage, ToolCall
from .utils import WireModel


class AssistantStartEvent(WireModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(WireModel):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class TextDeltaEvent(WireModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(WireModel):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(WireModel):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class ThinkingDeltaEvent(WireModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(WireModel):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(WireModel):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class ToolCallDeltaEvent(WireModel):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str  # raw JSON fragment of the arguments
    partial: AssistantMessage


class ToolCallEndEvent(WireModel):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall  # arguments fully parsed
    partial: AssistantMessage


class AssistantDoneEvent(WireModel):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class AssistantErrorEvent(WireModel):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage  # partial message with stop_reason/error_message set


type AssistantMessageEvent = Annotated[
    AssistantStartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | AssistantDoneEvent
    | AssistantErrorEvent,
    Field(discriminator="type"),
]
