"""Together AI provider, talking to the OpenAI-compatible HTTP API directly.

One call to `stream_response` = one POST /chat/completions with `stream: true`.
The response is Server-Sent Events: lines like `data: {...json chunk...}`,
ending with `data: [DONE]`. Each chunk carries a small `delta` (a piece of
text, reasoning, or tool-call arguments) that we append to an
AssistantMessage while yielding events for the UI.
"""

import asyncio
import json
import time

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from .events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .messages import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageDiagnostic,
    DiagnosticError,
    Message,
    ResponseTiming,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    calculate_cost,
)
from .tools import ToolDefinition
from .types import AiProviderConfig, JSONObject


type Chunk = dict[str, Any]

# Besides 5xx: timeout, conflict, too early, rate limited. Worth retrying.
TRANSIENT_STATUSES = {408, 409, 425, 429}


class ProviderError(Exception):
    def __init__(self, message: str, *, code: str | int | None = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class TogetherProvider:
    def __init__(self, config: AiProviderConfig, *, client: httpx.AsyncClient | None = None):
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        # `partial` is the live message being built. Events hold a reference
        # to it, not a copy: copy it yourself if you need a snapshot.
        partial = AssistantMessage(model=model)
        stream = _StreamState(partial)
        started_at = time.monotonic()

        yield AssistantStartEvent(partial=partial)

        try:
            payload = self._build_payload(model, system, messages, tools)
            async for chunk in self._stream_chunks(payload, partial, cancel):
                for event in stream.apply_chunk(chunk):
                    if partial.timing is None:
                        partial.timing = ResponseTiming(
                            time_to_first_output_ms=_elapsed_ms(started_at),
                            total_duration_ms=0,
                        )
                    yield event

            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError

            for event in stream.close_block():
                yield event
        except asyncio.CancelledError:
            # Raised by our own cancel check above, or by the caller cancelling
            # the task. Either way, finish with a message instead of vanishing.
            yield self._finish_with_error(partial, stream, started_at, "aborted", "Request aborted")
            return
        except Exception as error:
            partial.diagnostics.append(_diagnostic("request_failed", error))
            yield self._finish_with_error(partial, stream, started_at, "error", str(error))
            return

        self._apply_usage(partial, stream.usage)
        stop_reason = stream.stop_reason()
        if stop_reason == "stop" and stream.finish_reason not in (None, "stop", "eos", "tool_calls"):
            partial.diagnostics.append(
                AssistantMessageDiagnostic(
                    type="unknown_finish_reason",
                    details={"finish_reason": stream.finish_reason},
                )
            )
        partial.stop_reason = stop_reason
        partial.timing = _timing(partial, started_at)
        yield AssistantDoneEvent(reason=stop_reason, message=partial)

    # --- request -------------------------------------------------------------

    def _build_payload(
        self,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> JSONObject:
        payload: JSONObject = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *_to_wire_messages(messages)],
            "stream": True,
        }
        if tools:
            payload["tools"] = [_to_wire_tool(tool) for tool in tools]
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        return payload

    async def _stream_chunks(
        self,
        payload: JSONObject,
        partial: AssistantMessage,
        cancel: asyncio.Event | None,
    ) -> AsyncIterator[Chunk]:
        """POST the request (retrying transient failures) and yield parsed SSE chunks.

        Retries only happen before the first chunk arrives: once output has
        streamed to the UI, replaying it would duplicate text.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            **(self.config.headers or {}),
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        attempt = 0
        while True:
            if cancel is not None and cancel.is_set():
                return
            received_any = False
            try:
                async with self._client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors="replace")
                        raise ProviderError(
                            _error_message(body) or f"HTTP {response.status_code}",
                            code=response.status_code,
                            retryable=_is_transient(response.status_code),
                        )
                    async for line in response.aiter_lines():
                        if cancel is not None and cancel.is_set():
                            return
                        chunk = _parse_sse_line(line)
                        if chunk is None:
                            continue
                        if chunk == "done":
                            return
                        received_any = True
                        yield chunk
                    return
            except (ProviderError, httpx.TransportError) as error:
                retryable = isinstance(error, httpx.TransportError) or error.retryable
                if received_any or not retryable or attempt >= self.config.max_retries:
                    raise
                attempt += 1
                delay = min(0.5 * 2 ** (attempt - 1), self.config.max_retry_delay_seconds)
                partial.diagnostics.append(
                    _diagnostic("retry", error, details={"attempt": attempt, "delay_seconds": delay})
                )
                await asyncio.sleep(delay)

    # --- finishing -----------------------------------------------------------

    def _apply_usage(self, partial: AssistantMessage, usage: Usage | None) -> None:
        if usage is None:
            partial.diagnostics.append(AssistantMessageDiagnostic(type="missing_usage"))
            return
        pricing = self.config.pricing.get(partial.model)
        if pricing is None:
            partial.diagnostics.append(
                AssistantMessageDiagnostic(type="missing_pricing", details={"model": partial.model})
            )
        else:
            usage.cost = calculate_cost(usage, pricing)
        partial.usage = usage

    def _finish_with_error(
        self,
        partial: AssistantMessage,
        stream: "_StreamState",
        started_at: float,
        reason: str,
        message: str,
    ) -> AssistantErrorEvent:
        # Drop tool calls: their arguments may be cut off, and the loop must
        # never execute a call from a failed response.
        partial.content = [block for block in partial.content if not isinstance(block, ToolCall)]
        if stream.usage is not None:
            self._apply_usage(partial, stream.usage)
        partial.stop_reason = reason  # type: ignore[assignment]
        partial.error_message = message
        partial.timing = _timing(partial, started_at)
        return AssistantErrorEvent(reason=reason, error=partial)  # type: ignore[arg-type]


class _StreamState:
    """Turns OpenAI-style chunks into content blocks and events.

    Only one block is "open" at a time. When a delta of a different kind
    arrives (e.g. text after reasoning, or a new tool call), the open block is
    closed first, so blocks stay in the order the model produced them.
    """

    def __init__(self, partial: AssistantMessage):
        self.partial = partial
        self.open_index: int | None = None
        self.open_kind: str | None = None  # "text" | "thinking" | "tool"
        self.tool_call_index: int | None = None  # provider's index for the open tool call
        self.tool_args = ""
        self.finish_reason: str | None = None
        self.usage: Usage | None = None

    def apply_chunk(self, chunk: Chunk) -> list[AssistantMessageEvent]:
        if error := chunk.get("error"):
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ProviderError(message or "Provider error in stream")

        if self.partial.response_id is None and isinstance(chunk.get("id"), str):
            self.partial.response_id = chunk["id"]
        if isinstance(chunk.get("usage"), dict):
            self.usage = _parse_usage(chunk["usage"])

        events: list[AssistantMessageEvent] = []
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}

            # The field name varies by model on Together.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                events += self._append_thinking(reasoning)

            content = delta.get("content")
            if isinstance(content, str) and content:
                events += self._append_text(content)

            for tool_delta in delta.get("tool_calls") or []:
                events += self._append_tool_call(tool_delta)

            if finish_reason := choice.get("finish_reason"):
                self.finish_reason = finish_reason
        return events

    def stop_reason(self) -> StopReason:
        if self.finish_reason == "length":
            return "length"
        # Some models send finish_reason="stop" even after a tool call.
        if self.partial.tool_calls:
            return "toolUse"
        return "stop"

    def _append_thinking(self, delta: str) -> list[AssistantMessageEvent]:
        events = self._open("thinking", lambda: ThinkingContent(thinking=""), ThinkingStartEvent)
        block = self.partial.content[self.open_index]
        block.thinking += delta
        events.append(ThinkingDeltaEvent(content_index=self.open_index, delta=delta, partial=self.partial))
        return events

    def _append_text(self, delta: str) -> list[AssistantMessageEvent]:
        events = self._open("text", lambda: TextContent(text=""), TextStartEvent)
        block = self.partial.content[self.open_index]
        block.text += delta
        events.append(TextDeltaEvent(content_index=self.open_index, delta=delta, partial=self.partial))
        return events

    def _append_tool_call(self, tool_delta: dict[str, Any]) -> list[AssistantMessageEvent]:
        index = tool_delta.get("index", 0)
        function = tool_delta.get("function") or {}

        # The first delta of each tool call carries its id and name.
        new_call = self.open_kind != "tool" or index != self.tool_call_index
        events = self._open(
            "tool",
            lambda: ToolCall(id=tool_delta.get("id") or f"call_{index}", name=function.get("name") or ""),
            ToolCallStartEvent,
            force=new_call,
        )
        if new_call:
            self.tool_call_index = index
            self.tool_args = ""

        if args := function.get("arguments"):
            self.tool_args += args
            events.append(ToolCallDeltaEvent(content_index=self.open_index, delta=args, partial=self.partial))
        return events

    def _open(
        self,
        kind: str,
        new_block: Callable[[], AssistantContent],
        start_event: type[TextStartEvent | ThinkingStartEvent | ToolCallStartEvent],
        *,
        force: bool = False,
    ) -> list[AssistantMessageEvent]:
        """Make sure a block of `kind` is open, closing the previous one if needed."""
        if self.open_kind == kind and not force:
            return []
        events = self.close_block()
        self.partial.content.append(new_block())
        self.open_index = len(self.partial.content) - 1
        self.open_kind = kind
        events.append(start_event(content_index=self.open_index, partial=self.partial))
        return events

    def close_block(self) -> list[AssistantMessageEvent]:
        if self.open_index is None:
            return []
        index, kind = self.open_index, self.open_kind
        block = self.partial.content[index]
        self.open_index = self.open_kind = self.tool_call_index = None

        if kind == "text":
            return [TextEndEvent(content_index=index, content=block.text, partial=self.partial)]
        if kind == "thinking":
            return [ThinkingEndEvent(content_index=index, content=block.thinking, partial=self.partial)]

        try:
            arguments = json.loads(self.tool_args) if self.tool_args.strip() else {}
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
        except ValueError as error:
            # Keep going with empty args: the tool will fail validation and the
            # model sees that error in the tool result, so it can try again.
            arguments = {}
            self.partial.diagnostics.append(
                _diagnostic("invalid_tool_arguments", error, details={"tool": block.name, "raw": self.tool_args})
            )
        block.arguments = arguments
        return [ToolCallEndEvent(content_index=index, tool_call=block, partial=self.partial)]


# --- wire conversion ---------------------------------------------------------


def _to_wire_messages(messages: list[Message]) -> list[JSONObject]:
    wire: list[JSONObject] = []
    for message in messages:
        if isinstance(message, UserMessage):
            wire.append({"role": "user", "content": message.text})
        elif isinstance(message, AssistantMessage):
            # Failed/aborted responses stay in history for the UI but aren't
            # sent back: they may be cut off mid-sentence.
            if message.stop_reason in ("error", "aborted"):
                continue
            # Past reasoning isn't sent back; only the visible answer and calls.
            entry: JSONObject = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in message.tool_calls
                ]
            wire.append(entry)
        elif isinstance(message, ToolResultMessage):
            wire.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text})
    return wire


def _to_wire_tool(tool: ToolDefinition) -> JSONObject:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def _parse_sse_line(line: str) -> Chunk | str | None:
    """Return a chunk, "done" for the end marker, or None for lines to skip."""
    if not line.startswith("data:"):
        return None  # blank separators, ": keep-alive" comments, event: lines
    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return "done"
    if not data:
        return None
    return json.loads(data)


def _parse_usage(raw: dict[str, Any]) -> Usage:
    prompt = raw.get("prompt_tokens") or 0
    completion = raw.get("completion_tokens") or 0
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cached = prompt_details.get("cached_tokens") or raw.get("cached_tokens") or 0
    return Usage(
        input=prompt - cached,
        output=completion,
        cache_read=cached,
        reasoning=completion_details.get("reasoning_tokens"),
        total_tokens=raw.get("total_tokens") or prompt + completion,
    )


# --- helpers -----------------------------------------------------------------


def _is_transient(status: int) -> bool:
    return status in TRANSIENT_STATUSES or status >= 500


def _error_message(body: str) -> str | None:
    """Pull `error.message` out of an error response, falling back to the raw body."""
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return body.strip() or None
    if isinstance(error, dict):
        return error.get("message")
    return str(error) if error else None


def _diagnostic(kind: str, error: Exception, *, details: JSONObject | None = None) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=kind,
        error=DiagnosticError(
            name=error.__class__.__name__,
            message=str(error),
            code=getattr(error, "code", None),
        ),
        details=details,
    )


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _timing(partial: AssistantMessage, started_at: float) -> ResponseTiming:
    first = partial.timing.time_to_first_output_ms if partial.timing else None
    return ResponseTiming(time_to_first_output_ms=first, total_duration_ms=_elapsed_ms(started_at))
