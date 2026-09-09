"""Koschei Fabric case-envelope projection for Sentinel.

Sentinel may interpret evidence, but this adapter cannot mint authority, approve a
Web3 decision, or describe an effect as verified unless the caller supplies the
separate evidence-bound state. Importing this module has no network, training,
HOLDOUT, or paid-compute side effect.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_INTERPRETATION_STATES = frozenset({"AVAILABLE", "UNAVAILABLE", "INVALID", "TIMEOUT", "NOT_REQUESTED"})
_MAPPING_STATES = frozenset({"VERIFIED", "PARTIAL", "UNVERIFIED"})


@dataclass(frozen=True, slots=True)
class FabricSentinelProjectionV1:
    sentinel: dict[str, object]
    native_binding: dict[str, object]


def _require_sha256(value: str, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be lowercase SHA-256 hex")
    return value


def build_sentinel_fabric_projection_v1(
    *,
    interpretation_state: str,
    supporting_evidence_ids: tuple[str, ...],
    uncertainty: tuple[str, ...],
    native_ref: str,
    native_digest_sha256: str,
    native_schema: str = "sentinel.web4-security-event.v1",
    adapter_version: str = "sentinel.fabric-case-adapter.v1",
    mapping_state: str = "PARTIAL",
    model_ref: str | None = None,
    checkpoint_digest_sha256: str | None = None,
) -> FabricSentinelProjectionV1:
    """Project Sentinel interpretation metadata without producing authority."""
    if interpretation_state not in _INTERPRETATION_STATES:
        raise ValueError("unsupported interpretation state")
    if mapping_state not in _MAPPING_STATES:
        raise ValueError("unsupported mapping state")
    if any(not item.strip() for item in supporting_evidence_ids):
        raise ValueError("supporting evidence IDs must be non-empty")
    if len(set(supporting_evidence_ids)) != len(supporting_evidence_ids):
        raise ValueError("supporting evidence IDs must be unique")
    if any(not item.strip() for item in uncertainty):
        raise ValueError("uncertainty entries must be non-empty")
    if not native_ref.strip() or not native_schema.strip() or not adapter_version.strip():
        raise ValueError("native schema/ref and adapter version are required")

    native_digest = _require_sha256(native_digest_sha256, "native_digest_sha256")
    checkpoint_digest = None
    if checkpoint_digest_sha256 is not None:
        checkpoint_digest = _require_sha256(checkpoint_digest_sha256, "checkpoint_digest_sha256")
        if not model_ref or not model_ref.strip():
            raise ValueError("model_ref is required when checkpoint digest is supplied")
    if interpretation_state == "AVAILABLE" and not supporting_evidence_ids:
        raise ValueError("available interpretation requires supporting evidence")

    sentinel: dict[str, object] = {
        "role": "evidence-interpretation-only",
        "modelRef": model_ref,
        "checkpointDigestSha256": checkpoint_digest,
        "interpretationState": interpretation_state,
        "supportingEvidenceIds": list(supporting_evidence_ids),
        "uncertainty": list(uncertainty),
    }
    native_binding: dict[str, object] = {
        "owner": "koschei-sentinel",
        "nativeSchema": native_schema,
        "nativeRef": native_ref,
        "nativeDigestSha256": native_digest,
        "adapterVersion": adapter_version,
        "mappingState": mapping_state,
    }
    return FabricSentinelProjectionV1(sentinel=sentinel, native_binding=native_binding)


def fabric_sentinel_projection_payload_v1(**kwargs: object) -> dict[str, object]:
    """Return the Sentinel projection as plain JSON-serializable data."""
    return asdict(build_sentinel_fabric_projection_v1(**kwargs))
