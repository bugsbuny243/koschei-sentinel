from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_topology import (
    ArchitectureManifest,
    ExpertTopologyManifest,
    RouterManifest,
    verify_production_target_topology,
)
from koschei_sentinel.production_target_training_gate import (
    ProductionTargetTrainingBlocked,
    verify_production_training_gate,
)


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_payloads(model_ref: str = "Koschei/Sentinel-397B-A35B", revision: str = "a" * 64):
    architecture = {
        "schema_version": "sentinel.production-architecture.v1",
        "model_ref": model_ref,
        "model_revision": revision,
        "architecture_class": "sparse-moe",
        "dense_shared_parameters_billion": 5.0,
        "expert_count": 196,
        "expert_parameters_each_billion": 2.0,
    }
    architecture["manifest_sha256"] = _digest(architecture)

    router = {
        "schema_version": "sentinel.production-router.v1",
        "model_ref": model_ref,
        "model_revision": revision,
        "routing": "top-k",
        "experts_per_token": 15,
        "router_weights_revision": "b" * 64,
    }
    router["manifest_sha256"] = _digest(router)

    topology = {
        "schema_version": "sentinel.production-expert-topology.v1",
        "model_ref": model_ref,
        "model_revision": revision,
        "expert_count": 196,
        "expert_parameters_each_billion": 2.0,
        "experts_per_token": 15,
        "topology_revision": "c" * 64,
    }
    topology["manifest_sha256"] = _digest(topology)
    return architecture, router, topology


def test_topology_recomputes_exact_397b_35b() -> None:
    architecture_payload, router_payload, topology_payload = _manifest_payloads()
    binding = ProductionTargetBinding(
        model_ref=architecture_payload["model_ref"],
        model_revision=architecture_payload["model_revision"],
        architecture_manifest_sha256=architecture_payload["manifest_sha256"],
        router_manifest_sha256=router_payload["manifest_sha256"],
        expert_topology_manifest_sha256=topology_payload["manifest_sha256"],
        trainer_adapter_version="4.5.2",
        production_training_allowed=True,
        verification_status="verified",
        blockers=[],
    )
    verification = verify_production_target_topology(
        binding,
        ArchitectureManifest.model_validate(architecture_payload),
        RouterManifest.model_validate(router_payload),
        ExpertTopologyManifest.model_validate(topology_payload),
    )
    assert verification.valid is True
    assert verification.computed_total_parameters_billion == 397.0
    assert verification.computed_active_parameters_billion == 35.0


def test_topology_rejects_wrong_active_count() -> None:
    architecture_payload, router_payload, topology_payload = _manifest_payloads()
    router_payload["experts_per_token"] = 14
    topology_payload["experts_per_token"] = 14
    binding = ProductionTargetBinding(
        model_ref=architecture_payload["model_ref"],
        model_revision=architecture_payload["model_revision"],
        architecture_manifest_sha256=architecture_payload["manifest_sha256"],
        router_manifest_sha256=router_payload["manifest_sha256"],
        expert_topology_manifest_sha256=topology_payload["manifest_sha256"],
        trainer_adapter_version="4.5.2",
        production_training_allowed=True,
        verification_status="verified",
        blockers=[],
    )
    verification = verify_production_target_topology(
        binding,
        ArchitectureManifest.model_validate(architecture_payload),
        RouterManifest.model_validate(router_payload),
        ExpertTopologyManifest.model_validate(topology_payload),
    )
    assert verification.valid is False
    assert "recomputed active parameter count is not 35B" in verification.blockers


def test_training_gate_rejects_a17b_trainer_identity(tmp_path: Path) -> None:
    architecture, router, topology = _manifest_payloads()
    binding = {
        "schema_version": "sentinel.production-target-binding.v1",
        "total_parameters": "397B",
        "active_parameters": "35B",
        "architecture_class": "sparse-moe",
        "model_ref": architecture["model_ref"],
        "model_revision": architecture["model_revision"],
        "architecture_manifest_sha256": architecture["manifest_sha256"],
        "router_manifest_sha256": router["manifest_sha256"],
        "expert_topology_manifest_sha256": topology["manifest_sha256"],
        "trainer_adapter": "megatron-swift",
        "trainer_adapter_version": "4.5.2",
        "production_training_allowed": True,
        "verification_status": "verified",
        "blockers": [],
    }
    paths = {}
    for name, payload in (
        ("binding", binding),
        ("architecture", architecture),
        ("router", router),
        ("topology", topology),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    with pytest.raises(ProductionTargetTrainingBlocked, match="trainer model_ref"):
        verify_production_training_gate(
            binding_path=paths["binding"],
            architecture_manifest_path=paths["architecture"],
            router_manifest_path=paths["router"],
            expert_topology_manifest_path=paths["topology"],
            trainer_model_ref="Qwen/Qwen3.5-397B-A17B",
            trainer_model_revision="8472618112abcbd45acbcdc58436aff4233c23f7",
            trainer_adapter="megatron-swift",
            trainer_adapter_version="4.5.2",
        )
