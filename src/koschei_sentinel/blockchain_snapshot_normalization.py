from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from koschei_sentinel.anonymize import detect_sensitive_text, sanitize_text

_ALLOWED_SPDX = frozenset({"Apache-2.0", "MIT", "CC0-1.0", "CC-BY-4.0"})
_LICENSE_BASENAMES = ("license", "copying")
_LICENSE_SEPARATORS = (".", "-", "_")


@dataclass(frozen=True, slots=True)
class SafeSnapshotText:
    text: str
    detected_sensitive_kinds: tuple[str, ...]
    changed: bool


def is_license_evidence_path(path: str) -> bool:
    """Return whether a top-level path can carry repository license evidence."""

    pure = PurePosixPath(path)
    if pure.is_absolute() or len(pure.parts) != 1:
        return False
    name = pure.name.casefold()
    for basename in _LICENSE_BASENAMES:
        if name == basename:
            return True
        if any(name.startswith(basename + separator) for separator in _LICENSE_SEPARATORS):
            return True
    return False


def license_text_matches_spdx(spdx_id: str, text: str) -> bool:
    """Match allowlisted SPDX evidence by license text, not by filename alone."""

    if spdx_id not in _ALLOWED_SPDX:
        return False
    folded = " ".join(text.casefold().split())
    if spdx_id == "Apache-2.0":
        return "apache license" in folded and "version 2.0" in folded
    if spdx_id == "MIT":
        return (
            "permission is hereby granted" in folded
            and "free of charge" in folded
            and "without restriction" in folded
        )
    if spdx_id == "CC0-1.0":
        return "cc0" in folded and "1.0 universal" in folded
    return (
        "attribution 4.0 international" in folded
        or ("creative commons" in folded and "4.0 international" in folded)
    )


def select_license_evidence(
    expected_spdx: str,
    candidates: Mapping[str, str],
) -> str | None:
    """Select deterministic exact-commit license evidence from candidate texts."""

    for path in sorted(candidates):
        if not is_license_evidence_path(path):
            continue
        if license_text_matches_spdx(expected_spdx, candidates[path]):
            return path
    return None


def normalize_snapshot_text(text: str, *, salt: str) -> SafeSnapshotText:
    """Return training-safe text while preserving deterministic source lineage.

    Raw upstream text should be hashed and lineage-bound before this transformation.
    The returned text is suitable for the persisted training snapshot: supported raw
    identifiers and credentials are redacted or pseudonymized, then re-scanned.
    """

    findings = tuple(sorted(detect_sensitive_text(text)))
    if not findings:
        return SafeSnapshotText(text=text, detected_sensitive_kinds=(), changed=False)

    normalized = sanitize_text(text, salt=salt)
    remaining = detect_sensitive_text(normalized)
    if remaining:
        raise ValueError(
            "snapshot text remains sensitive after normalization: "
            + ", ".join(sorted(remaining))
        )
    if not normalized.strip():
        raise ValueError("snapshot text is empty after normalization")
    return SafeSnapshotText(
        text=normalized,
        detected_sensitive_kinds=findings,
        changed=normalized != text,
    )
