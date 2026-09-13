from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_provenance import (
    ProductionSourceEvidence,
    ProductionTargetProvenance,
    owner_key_fingerprint,
    production_target_binding_digest,
    verify_production_target_provenance,
)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding() -> ProductionTargetBinding:
    return ProductionTargetBinding(
        model_ref="koschei/sentinel-397b-35b",
        model_revision="a" * 64,
        architecture_manifest_sha256="1" * 64,
        router_manifest_sha256="2" * 64,
        expert_topology_manifest_sha256="3" * 64,
        trainer_adapter_version="4.5.2",
        production_training_allowed=True,
        verification_status="verified",
        blockers=[],
    )


def _source(kind: str, artifact_digest: str, source_revision: str) -> ProductionSourceEvidence:
    return ProductionSourceEvidence(
        artifact_kind=kind,
        artifact_manifest_sha256=artifact_digest,
        source_ref=f"pinned://{kind}/source",
        source_revision=source_revision,
        source_payload_sha256="f" * 64,
        independently_reviewed=True,
    )


def _signed_provenance(binding: ProductionTargetBinding, private_key: Ed25519PrivateKey) -> ProductionTargetProvenance:
    public_key = private_key.public_key()
    unsigned = {
        "schema_version": "sentinel.production-target-provenance.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "binding_sha256": production_target_binding_digest(binding),
        "sources": [
            _source("architecture", binding.architecture_manifest_sha256, "4" * 64).model_dump(mode="json"),
            _source("router", binding.router_manifest_sha256, "5" * 64).model_dump(mode="json"),
            _source("expert_topology", binding.expert_topology_manifest_sha256, "6" * 64).model_dump(mode="json"),
        ],
        "owner_key_fingerprint": owner_key_fingerprint(public_key),
    }
    provenance_sha256 = _digest(unsigned)
    signature = private_key.sign(provenance_sha256.encode("ascii"))
    return ProductionTargetProvenance.model_validate(
        {
            **unsigned,
            "provenance_sha256": provenance_sha256,
            "owner_signature_b64": base64.b64encode(signature).decode("ascii"),
        }
    )


def test_valid_owner_signed_provenance_is_accepted() -> None:
    binding = _binding()
    private_key = Ed25519PrivateKey.generate()
    provenance = _signed_provenance(binding, private_key)
    verified = verify_production_target_provenance(
        provenance,
        binding,
        private_key.public_key(),
    )
    assert verified.provenance_sha256 == provenance.provenance_sha256


def test_wrong_owner_key_is_rejected() -> None:
    binding = _binding()
    private_key = Ed25519PrivateKey.generate()
    provenance = _signed_provenance(binding, private_key)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_production_target_provenance(
            provenance,
            binding,
            Ed25519PrivateKey.generate().public_key(),
        )


def test_provenance_for_different_manifest_is_rejected() -> None:
    binding = _binding()
    private_key = Ed25519PrivateKey.generate()
    provenance = _signed_provenance(binding, private_key)
    changed_binding = binding.model_copy(
        update={"architecture_manifest_sha256": "9" * 64}
    )
    with pytest.raises(ValueError, match="exact target binding|architecture manifest"):
        verify_production_target_provenance(
            provenance,
            changed_binding,
            private_key.public_key(),
        )
