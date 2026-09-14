from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_launch_plan import ProductionMegatronLaunchPlan
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MegatronRuntimeInventory(StrictModel):
    schema_version: Literal["sentinel.megatron-runtime-inventory.v1"] = (
        "sentinel.megatron-runtime-inventory.v1"
    )
    nodes: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)
    minimum_observed_gpu_memory_gib: int = Field(gt=0)
    cuda_available_on_all_nodes: bool
    nccl_available_on_all_nodes: bool
    transformer_engine_available_on_all_nodes: bool
    megatron_core_available_on_all_nodes: bool
    homogeneous_gpu_count: bool
    high_speed_interconnect_verified: bool

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node


class MegatronRuntimePreflight(StrictModel):
    schema_version: Literal["sentinel.megatron-runtime-preflight.v1"] = (
        "sentinel.megatron-runtime-preflight.v1"
    )
    launch_plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    required_world_size: int = Field(gt=0)
    observed_world_size: int = Field(gt=0)
    runtime_ready: bool
    blockers: list[str] = Field(default_factory=list)
    launch_authorized: Literal[False] = False
    preflight_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def verify_integrity_and_no_authority(self) -> "MegatronRuntimePreflight":
        payload = self.model_dump(mode="json", exclude={"preflight_sha256"})
        if self.preflight_sha256 != _digest(payload):
            raise ValueError("production Megatron runtime-preflight self-hash mismatch")
        if self.runtime_ready != (not self.blockers):
            raise ValueError("runtime_ready must equal absence of blockers")
        if self.launch_authorized is not False:
            raise ValueError("runtime preflight cannot authorize production training")
        return self


def verify_runtime_preflight(
    spec: ProductionMegatronModelSpec,
    plan: ProductionMegatronLaunchPlan,
    inventory: MegatronRuntimeInventory,
) -> MegatronRuntimePreflight:
    blockers: list[str] = []
    required = spec.topology

    if plan.world_size != required.world_size:
        blockers.append("launch plan world size disagrees with model topology")
    if inventory.nodes < required.nodes:
        blockers.append("insufficient node count")
    if inventory.gpus_per_node < required.gpus_per_node:
        blockers.append("insufficient GPUs per node")
    if inventory.world_size < plan.world_size:
        blockers.append("insufficient total GPU world size")
    if inventory.minimum_observed_gpu_memory_gib < required.minimum_gpu_memory_gib:
        blockers.append("minimum observed GPU memory is below topology requirement")
    if not inventory.cuda_available_on_all_nodes:
        blockers.append("CUDA is not available on every node")
    if not inventory.nccl_available_on_all_nodes:
        blockers.append("NCCL is not available on every node")
    if not inventory.transformer_engine_available_on_all_nodes:
        blockers.append("Transformer Engine is not available on every node")
    if not inventory.megatron_core_available_on_all_nodes:
        blockers.append("Megatron Core is not available on every node")
    if not inventory.homogeneous_gpu_count:
        blockers.append("GPU inventory is not homogeneous across nodes")
    if not inventory.high_speed_interconnect_verified:
        blockers.append("high-speed interconnect has not been verified")

    payload = {
        "schema_version": "sentinel.megatron-runtime-preflight.v1",
        "launch_plan_sha256": plan.plan_sha256,
        "required_world_size": plan.world_size,
        "observed_world_size": inventory.world_size,
        "runtime_ready": not blockers,
        "blockers": blockers,
        "launch_authorized": False,
    }
    return MegatronRuntimePreflight.model_validate(
        {**payload, "preflight_sha256": _digest(payload)}
    )
