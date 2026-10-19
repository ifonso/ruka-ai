from dataclasses import dataclass, field
from typing import Literal, Mapping

from pydantic import SecretStr


type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]

# Accepted values vary by model (Qwen3.8 flagship: low/high/xhigh).
type ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per 1M tokens. Together doesn't return cost, so we compute it."""

    input: float
    output: float
    cache_read: float | None = None  # None -> billed at the input rate


@dataclass(frozen=True, slots=True)
class AiProviderConfig:
    api_key: SecretStr
    base_url: str = "https://api.together.xyz/v1"

    headers: Mapping[str, str] | None = field(default=None, repr=False)
    timeout_seconds: float = 120

    max_retries: int = 2
    max_retry_delay_seconds: float = 1.0
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None  # None -> model's default

    # model id -> pricing; models missing here get zero cost
    pricing: Mapping[str, ModelPricing] = field(default_factory=dict)
