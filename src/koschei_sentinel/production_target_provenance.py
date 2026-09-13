from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_target_binding import ProductionTargetBinding

_DIGEST = r"^[a-f0-9]{64}$"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def owner_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


class ProductionSourceEvidence(StrictModel):
    artifact_kind: Literal["architecture", "router", "expert_topology"]
    artifact_manifest_sha256: str = Field(pattern=_DIGEST)
    source_ref: str = Field(min_length=1, max_length=2048)
    source_revision: str = Field(pattern=_DIGEST)
    source_payload_sha256: str = Field(pattern=_DIGEST)
    verification_method: Literal["pinned-source-and-local-recompute"] = (
        "pinned-source-and-local-recompute"
    )
    independently_reviewed: Literal[True] = True


class ProductionTargetProvenance(StrictModel):
    schema_version: Literal["sentinel.production-target-provenance.v1"] = (
        "sentinel.production-target-provenance.v1"
    )
    model_ref: str = Field(min_length=1)
    model_revision: str = Field(pattern=_DIGEST)
    binding_sha256: str = Field(pattern=_DIGEST)
    sources: list[ProductionSourceEvidence] = Field(min_length=3, max_length=3)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    provenance_sha256: str = Field(pattern=_DIGEST)
    owner_signature_b64: str = Field(min_length=1)

    @model_validator(mode="after")
    def sources_are_complete(self) -> "ProductionTargetProvenance":
        kinds = [item.artifact_kind for item in self.sources]
        if sorted(kinds) != ["architecture", "expert_topology", "router"]:
            raise ValueError("production provenance must contain exactly one source for each manifest kind")
        return self


def production_target_binding_digest(binding: ProductionTargetBinding) -> str:
    return _canonical_digest(binding.model_dump(mode="json"))


def _unsigned_payload(provenance: ProductionTargetProvenance) -> dict[str, object]:
    payload = provenance.model_dump(mode="json")
    payload.pop("owner_signature_b64", None)
    payload.pop("provenance_sha256", None)
    return payload


def load_production_target_provenance(path: str | Path) -> ProductionTargetProvenance:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production target provenance: {source}") from exc
    return ProductionTargetProvenance.model_validate(payload)


def verify_production_target_provenance(
    provenance: ProductionTargetProvenance,
    binding: ProductionTargetBinding,
    owner_public_key: Ed25519PublicKey,
) -> ProductionTargetProvenance:
    if provenance.model_ref != binding.model_ref or provenance.model_revision != binding.model_revision:
        raise ValueError("production provenance model identity does not match target binding")
    if provenance.binding_sha256 != production_target_binding_digest(binding):
        raise ValueError("production provenance does not bind the exact target binding")

    expected_artifacts = {
        "architecture": binding.architecture_manifest_sha256,
        "router": binding.router_manifest_sha256,
        "expert_topology": binding.expert_topology_manifest_sha256,
    }
    for item in provenance.sources:
        if item.artifact_manifest_sha256 != expected_artifacts[item.artifact_kind]:
            raise ValueError(
                f"production provenance does not bind the {item.artifact_kind} manifest"
            )

    if provenance.owner_key_fingerprint != owner_key_fingerprint(owner_public_key):
        raise ValueError("production provenance owner key fingerprint mismatch")

    unsigned = _unsigned_payload(provenance)
    expected_digest = _canonical_digest(unsigned)
    if provenance.provenance_sha256 != expected_digest:
        raise ValueError("production provenance self-hash does not verify")

    try:
        signature = base64.b64decode(provenance.owner_signature_b64, validate=True)
    except ValueError as exc:
        raise ValueError("production provenance owner signature is not valid base64") from exc
    try:
        owner_public_key.verify(signature, expected_digest.encode("ascii"))
    except InvalidSignature as exc:
        raise ValueError("production provenance owner signature verification failed") from exc
    return provenance
