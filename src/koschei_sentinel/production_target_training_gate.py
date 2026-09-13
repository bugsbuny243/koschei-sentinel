from __future__ import annotations

from pathlib import Path

from koschei_sentinel.production_target_binding import (
    ProductionTargetBinding,
    load_production_target_binding,
    require_verified_production_target_binding,
)
from koschei_sentinel.production_target_topology import (
    ProductionTopologyVerification,
    load_architecture_manifest,
    load_expert_topology_manifest,
    load_router_manifest,
    verify_production_target_topology,
)


class ProductionTargetTrainingBlocked(ValueError):
    """Raised when the 397B/35B production lane is not cryptographically bound and verified."""


def verify_production_training_gate(
    *,
    binding_path: str | Path,
    architecture_manifest_path: str | Path,
    router_manifest_path: str | Path,
    expert_topology_manifest_path: str | Path,
    trainer_model_ref: str,
    trainer_model_revision: str,
    trainer_adapter: str,
    trainer_adapter_version: str,
) -> tuple[ProductionTargetBinding, ProductionTopologyVerification]:
    try:
        binding = require_verified_production_target_binding(
            load_production_target_binding(binding_path)
        )
        verification = verify_production_target_topology(
            binding,
            load_architecture_manifest(architecture_manifest_path),
            load_router_manifest(router_manifest_path),
            load_expert_topology_manifest(expert_topology_manifest_path),
        )
    except ValueError as exc:
        raise ProductionTargetTrainingBlocked(str(exc)) from exc

    if not verification.valid:
        detail = "; ".join(verification.blockers)
        raise ProductionTargetTrainingBlocked(
            "397B/35B topology verification failed" + (f": {detail}" if detail else "")
        )
    if trainer_model_ref != binding.model_ref:
        raise ProductionTargetTrainingBlocked("trainer model_ref does not match verified 397B/35B binding")
    if trainer_model_revision != binding.model_revision:
        raise ProductionTargetTrainingBlocked("trainer model revision does not match verified binding")
    if trainer_adapter != binding.trainer_adapter:
        raise ProductionTargetTrainingBlocked("trainer adapter does not match verified binding")
    if trainer_adapter_version != binding.trainer_adapter_version:
        raise ProductionTargetTrainingBlocked("trainer adapter version does not match verified binding")
    return binding, verification
