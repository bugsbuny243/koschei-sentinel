from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.production_model_source import ProductionModelSourceIntake
from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_provenance import verify_production_target_provenance
from koschei_sentinel.production_target_provenance_authoring import (
    build_signed_production_target_provenance,
)


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


def _intake(model_ref: str = "koschei/sentinel-397b-35b") -> ProductionModelSourceIntake:
    payload = {
        "model_ref": model_ref,
        "model_revision": "a" * 64,
        "source_kind": "internal-artifact",
        "source_ref": "pinned://sentinel/base",
        "source_revision": "b" * 64,
        "source_payload_sha256": "c" * 64,
        "license_ref": "internal://license/reviewed",
        "architecture_evidence_sha256": "4" * 64,
        "router_evidence_sha256": "5" * 64,
        "expert_topology_evidence_sha256": "6" * 64,
        "tokenizer_template_evidence_sha256": "7" * 64,
        "framework_compatibility_evidence_sha256": "8" * 64,
        "independently_reviewed": True,
        "review_ref": "review://production-model-source/001",
        "intake_sha256": "9" * 64,
    }
    return ProductionModelSourceIntake.model_validate(payload)


def test_authoring_builds_verifiable_owner_signed_provenance() -> None:
    binding = _binding()
    private_key = Ed25519PrivateKey.generate()
    provenance = build_signed_production_target_provenance(
        binding=binding,
        intake=_intake(),
        owner_private_key=private_key,
    )
    assert verify_production_target_provenance(
        provenance,
        binding,
        private_key.public_key(),
    ) is provenance


def test_authoring_rejects_mismatched_model_identity() -> None:
    with pytest.raises(ValueError, match="identity does not match"):
        build_signed_production_target_provenance(
            binding=_binding(),
            intake=_intake("other/model"),
            owner_private_key=Ed25519PrivateKey.generate(),
        )
