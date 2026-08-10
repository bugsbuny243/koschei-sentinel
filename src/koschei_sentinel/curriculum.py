from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CURRICULUM_SCHEMA = "sentinel.curriculum.v1"
REQUIRED_STAGE_IDS = ("L0", "L1", "L2", "L3", "L4", "S0", "S1", "S2", "S3", "S4")
REQUIRED_MODEL_DENIES = {
    "grant_capability",
    "widen_capability",
    "bypass_compiler_policy",
    "bypass_runtime_policy",
    "mutate_evidence",
    "override_deterministic_verdict",
    "promote_itself",
    "deploy_itself",
}
REQUIRED_KOSCH_DENIES = {
    "raise_model_confidence",
    "suppress_abstention",
    "weaken_benchmark_policy",
    "pass_failed_language_gate",
    "promote_candidate",
    "deploy_candidate",
    "grant_compiler_runtime_authority",
}
REQUIRED_ZERO_TOLERANCE = {
    "invented_authority",
    "capability_rewidening",
    "dependency_ambient_authority",
    "compiler_bypass_repair",
    "hidden_capability_expansion",
    "false_compile_claim",
    "unimplemented_feature_claim",
    "model_confidence_as_permission",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurriculumSourceContract(StrictModel):
    repository: str
    required_document: str
    commit_sha_required: bool
    compiler_oracle_required: bool
    mutable_branch_is_not_provenance: bool

    @model_validator(mode="after")
    def validate_source_authority(self) -> "CurriculumSourceContract":
        if self.repository != "bugsbuny243/koschei-lang":
            raise ValueError("language curriculum repository must be bugsbuny243/koschei-lang")
        if not self.required_document.endswith("MODEL_TRAINING_CONTRACT.md"):
            raise ValueError("language curriculum must bind MODEL_TRAINING_CONTRACT.md")
        if not self.commit_sha_required:
            raise ValueError("language curriculum must require an immutable commit SHA")
        if not self.compiler_oracle_required:
            raise ValueError("language curriculum must require compiler-oracle labels")
        if not self.mutable_branch_is_not_provenance:
            raise ValueError("mutable branch names cannot be accepted as provenance")
        return self


class CurriculumStage(StrictModel):
    id: str
    name: str
    requires_compiler_labels: bool = False
    promotion_gate: str | None = None
    requires: list[str] = Field(default_factory=list)
    production_authority: bool | None = None


class LanguageHardGate(StrictModel):
    zero_tolerance: list[str]
    required_benchmark_families: list[str]

    @model_validator(mode="after")
    def validate_hard_gate(self) -> "LanguageHardGate":
        missing = REQUIRED_ZERO_TOLERANCE.difference(self.zero_tolerance)
        if missing:
            raise ValueError(f"language hard gate missing zero-tolerance rules: {sorted(missing)}")
        if len(self.zero_tolerance) != len(set(self.zero_tolerance)):
            raise ValueError("language hard gate contains duplicate zero-tolerance rules")
        if len(self.required_benchmark_families) != len(set(self.required_benchmark_families)):
            raise ValueError("language hard gate contains duplicate benchmark families")
        return self


class ModelAuthorityBoundary(StrictModel):
    model_may: list[str]
    model_must_never: list[str]

    @model_validator(mode="after")
    def validate_authority(self) -> "ModelAuthorityBoundary":
        missing = REQUIRED_MODEL_DENIES.difference(self.model_must_never)
        if missing:
            raise ValueError(f"model authority deny set incomplete: {sorted(missing)}")
        if set(self.model_may).intersection(self.model_must_never):
            raise ValueError("model capability is simultaneously allowed and forbidden")
        return self


class KOSCHBoundary(StrictModel):
    allowed_future_roles: list[str]
    must_never: list[str]

    @model_validator(mode="after")
    def validate_kosch_boundary(self) -> "KOSCHBoundary":
        missing = REQUIRED_KOSCH_DENIES.difference(self.must_never)
        if missing:
            raise ValueError(f"KOSCH authority deny set incomplete: {sorted(missing)}")
        return self


class CurriculumPolicy(StrictModel):
    schema_: Literal[CURRICULUM_SCHEMA] = Field(alias="schema")
    id: str
    status: Literal["offline_research_only"]
    runtime_integration: Literal[False]
    source_contract: CurriculumSourceContract
    stages: list[CurriculumStage]
    language_hard_gate: LanguageHardGate
    authority_boundary: ModelAuthorityBoundary
    kosch_boundary: KOSCHBoundary

    @model_validator(mode="after")
    def validate_stage_order_and_authority(self) -> "CurriculumPolicy":
        stage_ids = tuple(stage.id for stage in self.stages)
        if stage_ids != REQUIRED_STAGE_IDS:
            raise ValueError(
                "curriculum stages must be ordered exactly as " + ",".join(REQUIRED_STAGE_IDS)
            )
        for stage in self.stages[:5]:
            if not stage.requires_compiler_labels:
                raise ValueError(f"language stage {stage.id} must require compiler labels")
        if self.stages[4].promotion_gate != "language_hard_gate":
            raise ValueError("L4 must end at language_hard_gate")
        for stage in self.stages[5:]:
            if "language_hard_gate" not in stage.requires:
                raise ValueError(f"security stage {stage.id} must require language_hard_gate")
            if stage.production_authority is True:
                raise ValueError(f"security stage {stage.id} cannot grant production authority")
        return self


def load_curriculum_policy(path: Path) -> CurriculumPolicy:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CurriculumPolicy.model_validate(data)
