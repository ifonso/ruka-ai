from time import time
from pydantic import BaseModel, ConfigDict


def current_timestamp_ms() -> int:
    """Return the current Unix timestamp in milliseconds."""
    return int(time() * 1000)


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        alias_generator=_to_camel,
    )

