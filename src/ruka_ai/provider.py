import asyncio

from collections.abc import AsyncIterator
from typing import Protocol

from .messages import Message
from .events import AssistantMessageEvent
from .tools import ToolDefinition


class AiProvider(Protocol):
    """Contract the agent loop depends on. Each provider (e.g. together.py) implements it."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream one model response.

        Never raises for provider/network failures: the stream ends with an
        AssistantErrorEvent instead, so the loop always gets a final message.
        When `cancel` is set, stop reading and end with reason="aborted".
        """
        ...
