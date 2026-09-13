from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class TargetArchitectureContract(StrictModel):
    schema_version: Literal["sentinel.target-architecture.v1"] = "sentinel.target-architecture.v1"
    target_total_parameters: Literal["397B"] = "397B"
    target_active_parameters: Literal["35B"] = "35B"
    target_role: Literal["production-target"] = "production-target"
    current_executable_model: Literal["Qwen/Qwen3.5-397B-A17B"] = "Qwen/Qwen3.5-397B-A17B"
    current_executable_active_parameters: Literal["17B"] = "17B"
    current_executable_role: Literal["systems-validation-and-curriculum-learning"] = (
        "systems-validation-and-curriculum-learning"
    )
    production_target_training_allowed: Literal[False] = False
    blockers: list[str] = Field(
        default_factory=lambda: [
            "no verified 397B/35B-active base architecture is wired into the trainer",
            "current Megatron path is pinned to Qwen3.5-397B-A17B",
            "routing/expert topology for a 35B-active target is not yet implemented and verified",
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def lanes_must_not_be_conflated(self) -> "TargetArchitectureContract":
        if self.current_executable_active_parameters == self.target_active_parameters:
            raise ValueError("validation lane must not masquerade as the 35B-active production target")
        if self.production_target_training_allowed:
            raise ValueError("397B/35B production training must remain blocked until a verified architecture exists")
        return self


def default_target_architecture_contract() -> TargetArchitectureContract:
    return TargetArchitectureContract()
