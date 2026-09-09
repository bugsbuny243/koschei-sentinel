"""Unambiguous JSON for evidence ingestion; this does not authenticate a source."""

from __future__ import annotations

import json
import math


def require_json_value(value: object) -> None:
    """Reject values that JSON cannot preserve, including non-finite numbers."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if type(value) is list:
        for child in value:
            require_json_value(child)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            require_json_value(child)
        return
    raise ValueError("value is not a JSON value")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object member")
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise ValueError("JSON numbers must be finite")


def strict_json_loads(raw: str | bytes) -> object:
    # Validate after decoding as well: a valid JSON exponent such as 1e999 can
    # overflow Python's float without going through parse_constant.
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        require_json_value(payload)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc
    return payload
