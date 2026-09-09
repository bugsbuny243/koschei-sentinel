from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CURRICULUM_SCHEMA = "sentinel.curriculum.v2"
REQUIRED_STAGE_IDS = (
    "L0",
    "L1",
    "L2",
    "N0",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
)
REQUIRED_SEMANTIC_PLANES = ("compatibility", "native_reality")
REQUIRED_MODEL_DENIES = {
    "grant_capability_or_authority",
    "widen_capability_or_authority",
    "bypass_compiler_policy",
    "bypass_runtime_policy",
    "mutate_evidence",
    "override_deterministic_verdict",
    "reinterpret_baseline_as_permission",
    "promote_itself",
    "deploy_itself",
}
REQUIRED_ZERO_TOLERANCE = {
    "invented_authority",
    "capability_rewidening",
    "dependency_ambient_authority",
    "compiler_runtime_bypass_repair",
    "hidden_authority_expansion",
    "false_compile_or_execution_claim",
    "unimplemented_feature_claim",
    "model_confidence_as_permission",
    "evidence_as_authority",
    "baseline_as_authority",
    "semantic_plane_confusion",
    "mainstream_syntax_relabelled_as_native_reality",
}

SemanticPlane = Literal["compatibility", "native_reality"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurriculumSourceContract(StrictModel):
    repository: str
    required_document: str
    required_contract_generation: int
    commit_sha_required: bool
    compiler_runtime_oracle_required: bool
    mutable_branch_is_not_provenance: bool
    semantic_plane_required: bool

    @model_validator(mode="after")
    def validate_source_authority(self) -> CurriculumSourceContract:
        if self.repository != "bugsbuny243/koschei-lang":
            raise ValueError("language curriculum repository must be bugsbuny243/koschei-lang")
        if not self.required_document.endswith("MODEL_TRAINING_CONTRACT.md"):
            raise ValueError("language curriculum must bind MODEL_TRAINING_CONTRACT.md")
        if self.required_contract_generation != 2:
            raise ValueError("language curriculum must require contract generation 2")
        if not self.commit_sha_required:
            raise ValueError("language curriculum must require an immutable commit SHA")
        if not self.compiler_runtime_oracle_required:
            raise ValueError("language curriculum must require compiler/runtime oracle labels")
        if not self.mutable_branch_is_not_provenance:
            raise ValueError("mutable branch names cannot be accepted as provenance")
        if not self.semantic_plane_required:
            raise ValueError("language curriculum must require explicit semantic-plane labels")
        return self


class CurriculumStage(StrictModel):
    id: str
    name: str
    semantic_plane: SemanticPlane | None = None
    requires_compiler_labels: bool = False
    required_concepts: list[str] = Field(default_factory=list)
    promotion_gate: str | None = None
    requires: list[str] = Field(default_factory=list)
    production_authority: bool | None = None


class LanguageHardGate(StrictModel):
    zero_tolerance: list[str]
    required_benchmark_families: list[str]

    @model_validator(mode="after")
    def validate_hard_gate(self) -> LanguageHardGate:
        missing = REQUIRED_ZERO_TOLERANCE.difference(self.zero_tolerance)
        if missing:
            raise ValueError(f"language hard gate missing zero-tolerance rules: {sorted(missing)}")
        if len(self.zero_tolerance) != len(set(self.zero_tolerance)):
            raise ValueError("language hard gate contains duplicate zero-tolerance rules")
        if len(self.required_benchmark_families) != len(set(self.required_benchmark_families)):
            raise ValueError("language hard gate contains duplicate benchmark families")
        if not self.required_benchmark_families:
            raise ValueError("language hard gate requires benchmark families")
        return self


class ModelAuthorityBoundary(StrictModel):
    model_may: list[str]
    model_must_never: list[str]

    @model_validator(mode="after")
    def validate_authority(self) -> ModelAuthorityBoundary:
        missing = REQUIRED_MODEL_DENIES.difference(self.model_must_never)
        if missing:
            raise ValueError(f"model authority deny set incomplete: {sorted(missing)}")
        if len(self.model_must_never) != len(set(self.model_must_never)):
            raise ValueError("model authority deny set contains duplicates")
        if set(self.model_may).intersection(self.model_must_never):
            raise ValueError("model capability is simultaneously allowed and forbidden")
        return self


class CurriculumPolicy(StrictModel):
    schema_: Literal["sentinel.curriculum.v2"] = Field(alias="schema")
    id: str
    status: Literal["offline_research_only"]
    runtime_integration: Literal[False]
    source_contract: CurriculumSourceContract
    semantic_planes: list[SemanticPlane]
    stages: list[CurriculumStage]
    language_hard_gate: LanguageHardGate
    authority_boundary: ModelAuthorityBoundary

    @model_validator(mode="after")
    def validate_stage_order_and_authority(self) -> CurriculumPolicy:
        if tuple(self.semantic_planes) != REQUIRED_SEMANTIC_PLANES:
            raise ValueError(
                "curriculum semantic planes must be ordered exactly as "
                + ",".join(REQUIRED_SEMANTIC_PLANES)
            )

        stage_ids = tuple(stage.id for stage in self.stages)
        if stage_ids != REQUIRED_STAGE_IDS:
            raise ValueError(
                "curriculum stages must be ordered exactly as " + ",".join(REQUIRED_STAGE_IDS)
            )

        compatibility_stages = self.stages[:3]
        for stage in compatibility_stages:
            if stage.semantic_plane != "compatibility":
                raise ValueError(f"compatibility stage {stage.id} must declare compatibility")
            if not stage.requires_compiler_labels:
                raise ValueError(f"language stage {stage.id} must require compiler labels")
            if stage.required_concepts:
                raise ValueError(f"compatibility stage {stage.id} must not claim native concepts")

        native_stages = self.stages[3:9]
        for stage in native_stages:
            if stage.semantic_plane != "native_reality":
                raise ValueError(f"native stage {stage.id} must declare native_reality")
            if not stage.required_concepts:
                raise ValueError(f"native stage {stage.id} must declare required concepts")
            if len(stage.required_concepts) != len(set(stage.required_concepts)):
                raise ValueError(f"native stage {stage.id} contains duplicate required concepts")

        if self.stages[8].id != "N5" or self.stages[8].promotion_gate != "language_hard_gate":
            raise ValueError("N5 must end at language_hard_gate")
        if any(stage.promotion_gate is not None for stage in self.stages[:8]):
            raise ValueError("language hard gate may only be emitted by N5")

        for stage in self.stages[9:]:
            if "language_hard_gate" not in stage.requires:
                raise ValueError(f"security stage {stage.id} must require language_hard_gate")
            if stage.production_authority is True:
                raise ValueError(f"security stage {stage.id} cannot grant production authority")
        return self


def load_curriculum_policy(path: Path) -> CurriculumPolicy:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CurriculumPolicy.model_validate(data)
