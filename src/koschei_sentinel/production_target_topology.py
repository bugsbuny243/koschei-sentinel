from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_target_binding import ProductionTargetBinding

_DIGEST = r"^[a-f0-9]{64}$"
_TARGET_TOTAL_B = 397.0
_TARGET_ACTIVE_B = 35.0
_TOLERANCE_B = 0.5


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ArchitectureManifest(StrictModel):
    schema_version: Literal["sentinel.production-architecture.v1"] = (
        "sentinel.production-architecture.v1"
    )
    model_ref: str = Field(min_length=1)
    model_revision: str = Field(pattern=_DIGEST)
    architecture_class: Literal["sparse-moe"] = "sparse-moe"
    dense_shared_parameters_billion: float = Field(gt=0.0)
    expert_count: int = Field(gt=1)
    expert_parameters_each_billion: float = Field(gt=0.0)
    manifest_sha256: str = Field(pattern=_DIGEST)


class RouterManifest(StrictModel):
    schema_version: Literal["sentinel.production-router.v1"] = (
        "sentinel.production-router.v1"
    )
    model_ref: str = Field(min_length=1)
    model_revision: str = Field(pattern=_DIGEST)
    routing: Literal["top-k"] = "top-k"
    experts_per_token: int = Field(gt=0)
    router_weights_revision: str = Field(pattern=_DIGEST)
    manifest_sha256: str = Field(pattern=_DIGEST)


class ExpertTopologyManifest(StrictModel):
    schema_version: Literal["sentinel.production-expert-topology.v1"] = (
        "sentinel.production-expert-topology.v1"
    )
    model_ref: str = Field(min_length=1)
    model_revision: str = Field(pattern=_DIGEST)
    expert_count: int = Field(gt=1)
    expert_parameters_each_billion: float = Field(gt=0.0)
    experts_per_token: int = Field(gt=0)
    topology_revision: str = Field(pattern=_DIGEST)
    manifest_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def active_experts_fit_pool(self) -> "ExpertTopologyManifest":
        if self.experts_per_token > self.expert_count:
            raise ValueError("experts_per_token cannot exceed expert_count")
        return self


class ProductionTopologyVerification(StrictModel):
    schema_version: Literal["sentinel.production-topology-verification.v1"] = (
        "sentinel.production-topology-verification.v1"
    )
    model_ref: str
    model_revision: str = Field(pattern=_DIGEST)
    computed_total_parameters_billion: float
    computed_active_parameters_billion: float
    total_target_billion: Literal[397.0] = 397.0
    active_target_billion: Literal[35.0] = 35.0
    valid: bool
    blockers: list[str] = Field(default_factory=list)
    verification_sha256: str = Field(pattern=_DIGEST)


def _load_manifest(path: str | Path, model_type):
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production topology manifest: {source}") from exc
    model = model_type.model_validate(payload)
    claimed = payload.pop("manifest_sha256", None)
    if claimed != _canonical_digest(payload):
        raise ValueError(f"production topology manifest self-hash mismatch: {source}")
    return model


def load_architecture_manifest(path: str | Path) -> ArchitectureManifest:
    return _load_manifest(path, ArchitectureManifest)


def load_router_manifest(path: str | Path) -> RouterManifest:
    return _load_manifest(path, RouterManifest)


def load_expert_topology_manifest(path: str | Path) -> ExpertTopologyManifest:
    return _load_manifest(path, ExpertTopologyManifest)


def verify_production_target_topology(
    binding: ProductionTargetBinding,
    architecture: ArchitectureManifest,
    router: RouterManifest,
    topology: ExpertTopologyManifest,
) -> ProductionTopologyVerification:
    blockers: list[str] = []
    identities = {
        (architecture.model_ref, architecture.model_revision),
        (router.model_ref, router.model_revision),
        (topology.model_ref, topology.model_revision),
        (binding.model_ref, binding.model_revision),
    }
    if len(identities) != 1:
        blockers.append("binding and topology manifests do not share one model identity")

    if architecture.expert_count != topology.expert_count:
        blockers.append("architecture and expert topology disagree on expert_count")
    if abs(architecture.expert_parameters_each_billion - topology.expert_parameters_each_billion) > 1e-9:
        blockers.append("architecture and expert topology disagree on expert size")
    if router.experts_per_token != topology.experts_per_token:
        blockers.append("router and expert topology disagree on experts_per_token")
    if router.experts_per_token > architecture.expert_count:
        blockers.append("router selects more experts than the architecture provides")

    computed_total = (
        architecture.dense_shared_parameters_billion
        + architecture.expert_count * architecture.expert_parameters_each_billion
    )
    computed_active = (
        architecture.dense_shared_parameters_billion
        + router.experts_per_token * architecture.expert_parameters_each_billion
    )
    if abs(computed_total - _TARGET_TOTAL_B) > _TOLERANCE_B:
        blockers.append("recomputed total parameter count is not 397B")
    if abs(computed_active - _TARGET_ACTIVE_B) > _TOLERANCE_B:
        blockers.append("recomputed active parameter count is not 35B")

    if binding.architecture_manifest_sha256 != architecture.manifest_sha256:
        blockers.append("binding does not bind the architecture manifest")
    if binding.router_manifest_sha256 != router.manifest_sha256:
        blockers.append("binding does not bind the router manifest")
    if binding.expert_topology_manifest_sha256 != topology.manifest_sha256:
        blockers.append("binding does not bind the expert topology manifest")

    payload = {
        "schema_version": "sentinel.production-topology-verification.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "computed_total_parameters_billion": round(computed_total, 6),
        "computed_active_parameters_billion": round(computed_active, 6),
        "total_target_billion": 397.0,
        "active_target_billion": 35.0,
        "valid": not blockers,
        "blockers": blockers,
    }
    return ProductionTopologyVerification.model_validate(
        {**payload, "verification_sha256": _canonical_digest(payload)}
    )
