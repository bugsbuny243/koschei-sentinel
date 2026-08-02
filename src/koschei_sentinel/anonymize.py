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
    "session",
    "authorization",
    "cookie",
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
    "case_id",
    "evidence_id",
    "lineage_id",
    "cluster_id",
}

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_SOLANA_ADDRESS = re.compile(
    r"(?<![1-9A-HJ-NP-Za-km-z])"
    r"[1-9A-HJ-NP-Za-km-z]{32,44}"
    r"(?![1-9A-HJ-NP-Za-km-z])"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")
_OPENAI_STYLE_KEY = re.compile(r"\b(?:sk|rk|pk|api)[-_][A-Za-z0-9_-]{16,}\b", re.IGNORECASE)
_LONG_HEX_SECRET = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{64,}(?![A-Fa-f0-9])")


def require_dataset_salt(salt: str | None = None) -> str:
    active_salt = salt or os.getenv("SENTINEL_DATASET_SALT", "")
    if len(active_salt) < 16:
        raise ValueError("SENTINEL_DATASET_SALT must contain at least 16 characters")
    return active_salt


def anonymize_record(record: Mapping[str, Any], *, salt: str | None = None) -> dict[str, Any]:
    return _walk(dict(record), require_dataset_salt(salt))


def pseudonymize(value: str, salt: str, *, prefix: str = "id") -> str:
    digest = hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:24]
    safe_prefix = re.sub(r"[^a-z0-9_]+", "_", prefix.lower()).strip("_") or "id"
    return f"{safe_prefix}_{digest}"


def sanitize_text(value: str, *, salt: str) -> str:
    text = _EMAIL.sub("[REDACTED_EMAIL]", value)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    text = _JWT.sub("[REDACTED_SECRET]", text)
    text = _BEARER.sub("[REDACTED_SECRET]", text)
    text = _OPENAI_STYLE_KEY.sub("[REDACTED_SECRET]", text)
    text = _LONG_HEX_SECRET.sub("[REDACTED_SECRET]", text)
    return _SOLANA_ADDRESS.sub(
        lambda match: pseudonymize(match.group(0), salt, prefix="address"), text
    )


def detect_sensitive_text(value: str) -> list[str]:
    findings: list[str] = []
    checks = (
        ("email", _EMAIL),
        ("phone", _PHONE),
        ("solana_address", _SOLANA_ADDRESS),
        ("jwt", _JWT),
        ("bearer_credential", _BEARER),
        ("api_key", _OPENAI_STYLE_KEY),
        ("long_hex_secret", _LONG_HEX_SECRET),
    )
    for name, pattern in checks:
        if pattern.search(value):
            findings.append(name)
    return findings


def _walk(value: Any, salt: str, key: str = "") -> Any:
    normalized_key = key.lower()
    if normalized_key in SENSITIVE_KEYS or any(
        normalized_key.endswith(f"_{suffix}") for suffix in SENSITIVE_KEYS
    ):
        return "[REDACTED]"

    if isinstance(value, Mapping):
        return {
            str(item_key): _walk(item_value, salt, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_walk(item, salt, key) for item in value]
    if isinstance(value, str):
        if normalized_key in IDENTIFIER_KEYS or any(
            normalized_key.endswith(f"_{suffix}") for suffix in IDENTIFIER_KEYS
        ):
            return pseudonymize(value, salt, prefix=normalized_key or "id")
        return sanitize_text(value, salt=salt)
    return value
