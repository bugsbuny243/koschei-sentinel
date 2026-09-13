from __future__ import annotations

import hashlib
import json
from pathlib import Path

from koschei_sentinel.production_megatron_config import (
    ProductionMegatronConfig,
    load_production_megatron_config,
)
from koschei_sentinel.production_target_binding import (
    ProductionTargetBinding,
    load_production_target_binding,
)
from koschei_sentinel.production_target_provenance import (
    ProductionTargetProvenance,
    load_production_target_provenance,
    verify_production_target_provenance,
)
from koschei_sentinel.production_target_training_gate import (
    ProductionTargetTrainingBlocked,
    verify_production_training_gate,
)
from koschei_sentinel.production_target_topology import ProductionTopologyVerification
from koschei_sentinel.promotion import load_owner_public_key


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def production_target_binding_digest(binding: ProductionTargetBinding) -> str:
    return _canonical_digest(binding.model_dump(mode="json"))


def verify_production_megatron_gate(
    *,
    config_path: str | Path,
    binding_path: str | Path,
    architecture_manifest_path: str | Path,
    router_manifest_path: str | Path,
    expert_topology_manifest_path: str | Path,
    provenance_path: str | Path,
    owner_public_key_path: str | Path,
) -> tuple[
    ProductionMegatronConfig,
    ProductionTargetBinding,
    ProductionTopologyVerification,
    ProductionTargetProvenance,
]:
    config = load_production_megatron_config(config_path)
    binding = load_production_target_binding(binding_path)

    verified_binding, verification = verify_production_training_gate(
        binding_path=binding_path,
        architecture_manifest_path=architecture_manifest_path,
        router_manifest_path=router_manifest_path,
        expert_topology_manifest_path=expert_topology_manifest_path,
        trainer_model_ref=config.model_ref,
        trainer_model_revision=config.model_revision,
        trainer_adapter=config.trainer_adapter,
        trainer_adapter_version=config.trainer_adapter_version,
    )

    if config.production_target_binding_sha256 != production_target_binding_digest(verified_binding):
        raise ProductionTargetTrainingBlocked(
            "production Megatron config does not bind the verified target binding"
        )
    if config.production_topology_verification_sha256 != verification.verification_sha256:
        raise ProductionTargetTrainingBlocked(
            "production Megatron config does not bind the topology verification"
        )
    if config.total_parameters != verified_binding.total_parameters:
        raise ProductionTargetTrainingBlocked("production config total-parameter target mismatch")
    if config.active_parameters != verified_binding.active_parameters:
        raise ProductionTargetTrainingBlocked("production config active-parameter target mismatch")
    if config.architecture_class != verified_binding.architecture_class:
        raise ProductionTargetTrainingBlocked("production config architecture class mismatch")

    try:
        provenance = verify_production_target_provenance(
            load_production_target_provenance(provenance_path),
            verified_binding,
            load_owner_public_key(owner_public_key_path),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ProductionTargetTrainingBlocked(
            f"production target provenance verification failed: {exc}"
        ) from exc

    return config, verified_binding, verification, provenance
