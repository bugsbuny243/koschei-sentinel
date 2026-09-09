from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from koschei_sentinel.strict_json import require_json_value, strict_json_loads


def parse_timezone_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def sha256_canonical_json(payload: object) -> str:
    require_json_value(payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_regular_bytes(path: str | Path, label: str) -> bytes:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        return source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc


def load_strict_json_object(path: str | Path, label: str) -> dict[str, object]:
    raw = read_regular_bytes(path, label)
    try:
        payload = strict_json_loads(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def hash_stable_regular_file(path: str | Path, label: str) -> tuple[str, int]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        before = source.stat()
    except OSError as exc:
        raise ValueError(f"cannot stat {label}") from exc
    if before.st_size <= 0:
        raise ValueError(f"{label} must not be empty")

    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = source.stat()
    except OSError as exc:
        raise ValueError(f"cannot hash {label}") from exc

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        raise ValueError(f"{label} changed while hashing")
    return digest.hexdigest(), after.st_size
