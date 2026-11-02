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
_MAX_SUPPORTING_EVIDENCE_IDS = 256
_MAX_EVIDENCE_ID_LENGTH = 160
_MAX_UNCERTAINTY_ENTRIES = 128
_MAX_UNCERTAINTY_LENGTH = 2000
_MAX_MODEL_REF_LENGTH = 512
_MAX_NATIVE_SCHEMA_LENGTH = 256
_MAX_NATIVE_REF_LENGTH = 1024
_MAX_ADAPTER_VERSION_LENGTH = 128


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
    # Bounds mirror fabric.security-case-envelope.v1 so a projection the adapter
    # accepts is never rejected by the Web3 coordinator's schema.
    if len(supporting_evidence_ids) > _MAX_SUPPORTING_EVIDENCE_IDS:
        raise ValueError(f"at most {_MAX_SUPPORTING_EVIDENCE_IDS} supporting evidence IDs are allowed")
    if any(len(item) > _MAX_EVIDENCE_ID_LENGTH for item in supporting_evidence_ids):
        raise ValueError(f"supporting evidence IDs must be at most {_MAX_EVIDENCE_ID_LENGTH} characters")
    if len(uncertainty) > _MAX_UNCERTAINTY_ENTRIES:
        raise ValueError(f"at most {_MAX_UNCERTAINTY_ENTRIES} uncertainty entries are allowed")
    if any(len(item) > _MAX_UNCERTAINTY_LENGTH for item in uncertainty):
        raise ValueError(f"uncertainty entries must be at most {_MAX_UNCERTAINTY_LENGTH} characters")
    for value, field, limit in (
        (native_ref, "native_ref", _MAX_NATIVE_REF_LENGTH),
        (native_schema, "native_schema", _MAX_NATIVE_SCHEMA_LENGTH),
        (adapter_version, "adapter_version", _MAX_ADAPTER_VERSION_LENGTH),
    ):
        if len(value) > limit:
            raise ValueError(f"{field} must be at most {limit} characters")
    if model_ref is not None and len(model_ref) > _MAX_MODEL_REF_LENGTH:
        raise ValueError(f"model_ref must be at most {_MAX_MODEL_REF_LENGTH} characters")

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
