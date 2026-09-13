from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from koschei_sentinel.production_megatron_gate import (
    production_target_binding_digest,
    verify_production_megatron_gate,
)
from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_topology import ProductionTopologyVerification


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


def _verification() -> ProductionTopologyVerification:
    return ProductionTopologyVerification(
        model_ref="koschei/sentinel-397b-35b",
        model_revision="a" * 64,
        computed_total_parameters_billion=397.0,
        computed_active_parameters_billion=35.0,
        valid=True,
        blockers=[],
        verification_sha256="4" * 64,
    )


def _write_config(tmp_path, binding_digest: str, topology_digest: str):
    path = tmp_path / "production.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sentinel.production-megatron-config.v1",
                "lane": "production-target-397b-35b",
                "run_id": "sentinel-prod-001",
                "model_ref": "koschei/sentinel-397b-35b",
                "model_revision": "a" * 64,
                "total_parameters": "397B",
                "active_parameters": "35B",
                "architecture_class": "sparse-moe",
                "trainer_adapter": "megatron-swift",
                "trainer_adapter_version": "4.5.2",
                "corpus_dir": "build/prod/train",
                "validation_corpus_dir": "build/prod/validation",
                "dataset_dir": "build/prod/dataset",
                "output_dir": "build/prod/output",
                "topology": {
                    "nodes": 8,
                    "gpus_per_node": 8,
                    "minimum_gpu_memory_gib": 80,
                    "tensor_model_parallel_size": 8,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 8,
                    "sequence_parallel": True,
                },
                "launch_approval_env": "KOSCHEI_397B_35B_LAUNCH_APPROVED",
                "production_target_binding_sha256": binding_digest,
                "production_topology_verification_sha256": topology_digest,
            }
        ),
        encoding="utf-8",
    )
    return path


def _patch_verified_dependencies(monkeypatch, binding, verification) -> None:
    provenance = SimpleNamespace(provenance_sha256="5" * 64)
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.load_production_target_binding",
        lambda _: binding,
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.verify_production_training_gate",
        lambda **_: (binding, verification),
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.load_production_target_provenance",
        lambda _: provenance,
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.load_owner_public_key",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.verify_production_target_provenance",
        lambda candidate, *_: candidate,
    )


def test_gate_accepts_config_bound_to_verified_artifacts(tmp_path, monkeypatch) -> None:
    binding = _binding()
    verification = _verification()
    config_path = _write_config(
        tmp_path,
        production_target_binding_digest(binding),
        verification.verification_sha256,
    )
    _patch_verified_dependencies(monkeypatch, binding, verification)
    config, returned_binding, returned_verification, provenance = verify_production_megatron_gate(
        config_path=config_path,
        binding_path="binding.json",
        architecture_manifest_path="architecture.json",
        router_manifest_path="router.json",
        expert_topology_manifest_path="topology.json",
        provenance_path="provenance.json",
        owner_public_key_path="owner.pem",
    )
    assert config.active_parameters == "35B"
    assert returned_binding is binding
    assert returned_verification is verification
    assert provenance.provenance_sha256 == "5" * 64


def test_gate_rejects_config_not_bound_to_binding(tmp_path, monkeypatch) -> None:
    binding = _binding()
    verification = _verification()
    config_path = _write_config(tmp_path, "f" * 64, verification.verification_sha256)
    _patch_verified_dependencies(monkeypatch, binding, verification)
    with pytest.raises(ValueError, match="does not bind the verified target binding"):
        verify_production_megatron_gate(
            config_path=config_path,
            binding_path="binding.json",
            architecture_manifest_path="architecture.json",
            router_manifest_path="router.json",
            expert_topology_manifest_path="topology.json",
            provenance_path="provenance.json",
            owner_public_key_path="owner.pem",
        )


def test_gate_rejects_failed_provenance_verification(tmp_path, monkeypatch) -> None:
    binding = _binding()
    verification = _verification()
    config_path = _write_config(
        tmp_path,
        production_target_binding_digest(binding),
        verification.verification_sha256,
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.load_production_target_binding",
        lambda _: binding,
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.verify_production_training_gate",
        lambda **_: (binding, verification),
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.load_production_target_provenance",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.load_owner_public_key",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "koschei_sentinel.production_megatron_gate.verify_production_target_provenance",
        lambda *_: (_ for _ in ()).throw(ValueError("owner signature verification failed")),
    )
    with pytest.raises(ValueError, match="provenance verification failed"):
        verify_production_megatron_gate(
            config_path=config_path,
            binding_path="binding.json",
            architecture_manifest_path="architecture.json",
            router_manifest_path="router.json",
            expert_topology_manifest_path="topology.json",
            provenance_path="provenance.json",
            owner_public_key_path="owner.pem",
        )
