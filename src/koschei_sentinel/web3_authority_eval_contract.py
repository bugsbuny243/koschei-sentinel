from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

CaseMode = Literal["positive", "hard_negative", "ambiguous", "abstention"]
ConfidenceLabel = Literal["low", "medium", "high"]


class TargetArchitecture(StrictModel):
    total_parameters: Literal["397B"]
    active_parameters: Literal["35B"]
    small_model_lane_role: Literal["systems-validation-and-curriculum-learning"]


class SafetyPolicy(StrictModel):
    defensive_authorized_use_only: Literal[True]
    exclude_operational_unauthorized_targeting: Literal[True]
    exclude_credential_theft: Literal[True]
    exclude_persistence_or_evasion: Literal[True]
    exclude_destructive_actions: Literal[True]
    require_evidence_grounded_conclusions: Literal[True]
    require_abstention_when_authority_or_evidence_is_missing: Literal[True]


class ReasoningContract(StrictModel):
    required_fields: list[str] = Field(min_length=1)
    forbidden_shortcuts: list[str] = Field(min_length=1)


class EvalFamily(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    maturity: Literal["core", "research_to_core"]
    actors: list[str] = Field(min_length=1)
    trust_boundaries: list[str] = Field(min_length=1)
    required_evidence: list[str] = Field(min_length=1)
    failure_modes: list[str] = Field(min_length=1)
    expected_remediation: list[str] = Field(min_length=1)
    confidence_ceiling_without_required_evidence: float = Field(ge=0.0, le=0.75)


class Scoring(StrictModel):
    dimensions: dict[str, float]
    hard_fail_conditions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "Scoring":
        if abs(sum(self.dimensions.values()) - 1.0) > 1e-9:
            raise ValueError("scoring dimension weights must sum to 1.0")
        if any(weight <= 0 for weight in self.dimensions.values()):
            raise ValueError("scoring dimension weights must be positive")
        return self


class EvalGeneration(StrictModel):
    minimum_cases_per_family: int = Field(ge=1)
    required_distribution: dict[CaseMode, float]
    require_counterfactual_pair: Literal[True]
    require_evidence_removed_variant: Literal[True]
    require_remediation_regression_case: Literal[True]

    @model_validator(mode="after")
    def distribution_is_complete(self) -> "EvalGeneration":
        expected = {"positive", "hard_negative", "ambiguous", "abstention"}
        if set(self.required_distribution) != expected:
            raise ValueError("required_distribution must define all four case modes")
        if abs(sum(self.required_distribution.values()) - 1.0) > 1e-9:
            raise ValueError("required_distribution weights must sum to 1.0")
        if any(weight <= 0 for weight in self.required_distribution.values()):
            raise ValueError("required_distribution weights must be positive")
        return self


class Web3AuthorityEvalConfig(StrictModel):
    schema_version: Literal["sentinel.web3-authority-evals.v1"]
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    target_architecture: TargetArchitecture
    safety_policy: SafetyPolicy
    reasoning_contract: ReasoningContract
    case_modes: list[CaseMode] = Field(min_length=4, max_length=4)
    families: list[EvalFamily] = Field(min_length=1)
    scoring: Scoring
    eval_generation: EvalGeneration

    @model_validator(mode="after")
    def config_is_internally_consistent(self) -> "Web3AuthorityEvalConfig":
        expected_modes = {"positive", "hard_negative", "ambiguous", "abstention"}
        if set(self.case_modes) != expected_modes:
            raise ValueError("case_modes must contain each supported mode exactly once")
        family_ids = [family.id for family in self.families]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("eval family ids must be unique")
        required_fields = set(self.reasoning_contract.required_fields)
        mandatory = {"actors", "effective_authority", "evidence", "evidence_limits", "confidence"}
        if not mandatory.issubset(required_fields):
            raise ValueError("reasoning contract is missing mandatory authority/evidence fields")
        return self


class EvalScenario(StrictModel):
    claim: str = Field(min_length=1)
    observed: list[str] = Field(min_length=1)


class EvalSeedCase(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    family: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    mode: CaseMode
    scenario: EvalScenario
    expected_reasoning: list[str] = Field(min_length=1)
    expected_conclusion: str = Field(min_length=1)
    expected_confidence: ConfidenceLabel
    expected_remediation: list[str] = Field(default_factory=list)
    must_abstain_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def abstention_cases_define_boundary(self) -> "EvalSeedCase":
        if self.mode == "abstention" and not self.must_abstain_from:
            raise ValueError("abstention cases must define must_abstain_from")
        return self


class Web3AuthorityEvalSeeds(StrictModel):
    schema_version: Literal["sentinel.web3-authority-eval-seeds.v1"]
    curriculum_ref: Literal["configs/training/web3-authority-evals.v1.json"]
    purpose: str = Field(min_length=1)
    cases: list[EvalSeedCase] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> "Web3AuthorityEvalSeeds":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("eval seed case ids must be unique")
        return self


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load valid JSON from {path}: {exc}") from exc


def load_eval_config(path: Path) -> Web3AuthorityEvalConfig:
    return Web3AuthorityEvalConfig.model_validate(_load_json(path))


def load_eval_seeds(path: Path) -> Web3AuthorityEvalSeeds:
    return Web3AuthorityEvalSeeds.model_validate(_load_json(path))


def validate_eval_bundle(
    config: Web3AuthorityEvalConfig,
    seeds: Web3AuthorityEvalSeeds,
) -> None:
    family_ids = {family.id for family in config.families}
    seed_families = {case.family for case in seeds.cases}

    unknown = sorted(seed_families - family_ids)
    if unknown:
        raise ValueError(f"seed cases reference unknown families: {unknown}")

    missing = sorted(family_ids - seed_families)
    if missing:
        raise ValueError(f"every eval family requires at least one reviewed seed case: {missing}")

    unsupported_modes = sorted({case.mode for case in seeds.cases} - set(config.case_modes))
    if unsupported_modes:
        raise ValueError(f"seed cases use unsupported modes: {unsupported_modes}")

    if "invented_evidence" not in config.scoring.hard_fail_conditions:
        raise ValueError("invented_evidence must remain a hard-fail condition")
    if "invented_authority" not in config.scoring.hard_fail_conditions:
        raise ValueError("invented_authority must remain a hard-fail condition")


def load_and_validate_eval_bundle(config_path: Path, seeds_path: Path) -> tuple[Web3AuthorityEvalConfig, Web3AuthorityEvalSeeds]:
    config = load_eval_config(config_path)
    seeds = load_eval_seeds(seeds_path)
    validate_eval_bundle(config, seeds)
    return config, seeds
