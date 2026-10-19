from dataclasses import dataclass

from .types import JSONObject


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """What the model sees about a tool: name, description and JSON Schema.

    Sent as `tools=[{"type": "function", "function": {...}}]` in the request.
    Execution lives elsewhere; the provider only needs the definition.
    """

    name: str
    description: str
    parameters: JSONObject  # JSON Schema for the arguments object
