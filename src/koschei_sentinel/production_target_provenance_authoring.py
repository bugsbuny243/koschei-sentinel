from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.production_model_source import (
    ProductionModelSourceIntake,
    verify_production_model_source_intake,
)
from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_provenance import (
    ProductionSourceEvidence,
    ProductionTargetProvenance,
    owner_key_fingerprint,
    production_target_binding_digest,
)

_SIGNATURE_CONTEXT = b"koschei-sentinel-production-target-provenance-v1\0"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_signed_production_target_provenance(
    *,
    binding: ProductionTargetBinding,
    intake: ProductionModelSourceIntake,
    owner_private_key: Ed25519PrivateKey,
) -> ProductionTargetProvenance:
    verify_production_model_source_intake(intake)
    if intake.model_ref != binding.model_ref or intake.model_revision != binding.model_revision:
        raise ValueError("model source intake identity does not match production target binding")

    sources = [
        ProductionSourceEvidence(
            artifact_kind="architecture",
            artifact_manifest_sha256=binding.architecture_manifest_sha256,
            source_ref=intake.source_ref,
            source_revision=intake.source_revision,
            source_payload_sha256=intake.architecture_evidence_sha256,
            independently_reviewed=True,
        ),
        ProductionSourceEvidence(
            artifact_kind="router",
            artifact_manifest_sha256=binding.router_manifest_sha256,
            source_ref=intake.source_ref,
            source_revision=intake.source_revision,
            source_payload_sha256=intake.router_evidence_sha256,
            independently_reviewed=True,
        ),
        ProductionSourceEvidence(
            artifact_kind="expert_topology",
            artifact_manifest_sha256=binding.expert_topology_manifest_sha256,
            source_ref=intake.source_ref,
            source_revision=intake.source_revision,
            source_payload_sha256=intake.expert_topology_evidence_sha256,
            independently_reviewed=True,
        ),
    ]
    unsigned = {
        "schema_version": "sentinel.production-target-provenance.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "binding_sha256": production_target_binding_digest(binding),
        "sources": [item.model_dump(mode="json") for item in sources],
        "owner_key_fingerprint": owner_key_fingerprint(owner_private_key.public_key()),
    }
    provenance_sha256 = _canonical_digest(unsigned)
    signature = owner_private_key.sign(
        _SIGNATURE_CONTEXT + provenance_sha256.encode("ascii")
    )
    return ProductionTargetProvenance.model_validate(
        {
            **unsigned,
            "provenance_sha256": provenance_sha256,
            "owner_signature_b64": base64.b64encode(signature).decode("ascii"),
        }
    )


def write_provenance(path: str | Path, provenance: ProductionTargetProvenance) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
