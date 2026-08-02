from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

SENSITIVE_KEYS = {
    "email",
    "name",
    "phone",
    "ip",
    "password",
    "secret",
    "token",
    "api_key",
    "private_key",
}
IDENTIFIER_KEYS = {
    "address",
    "wallet",
    "owner",
    "creator",
    "mint",
    "target",
    "target_ref",
    "signature",
}

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


def anonymize_record(record: Mapping[str, Any], *, salt: str | None = None) -> dict[str, Any]:
    active_salt = salt or os.getenv("SENTINEL_DATASET_SALT", "")
    if len(active_salt) < 16:
        raise ValueError("SENTINEL_DATASET_SALT must contain at least 16 characters")
    return _walk(dict(record), active_salt)


def pseudonymize(value: str, salt: str, *, prefix: str = "id") -> str:
    digest = hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _walk(value: Any, salt: str, key: str = "") -> Any:
    normalized_key = key.lower()
    if normalized_key in SENSITIVE_KEYS or any(
        normalized_key.endswith(f"_{suffix}") for suffix in SENSITIVE_KEYS
    ):
        return "[REDACTED]"

    if isinstance(value, Mapping):
        return {str(item_key): _walk(item_value, salt, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_walk(item, salt, key) for item in value]
    if isinstance(value, str):
        if normalized_key in IDENTIFIER_KEYS or any(
            normalized_key.endswith(f"_{suffix}") for suffix in IDENTIFIER_KEYS
        ):
            return pseudonymize(value, salt, prefix=normalized_key or "id")
        return _PHONE.sub("[REDACTED_PHONE]", _EMAIL.sub("[REDACTED_EMAIL]", value))
    return value
