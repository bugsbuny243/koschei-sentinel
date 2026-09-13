from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

Mode = Literal["positive", "hard_negative", "ambiguous", "abstention"]
Confidence = Literal["low", "medium", "high"]


class TargetArchitecture(StrictModel):
    total_parameters: Literal["397B"]
    active_parameters: Literal["35B"]


class SafetyPolicy(StrictModel):
    defensive_authorized_use_only: Literal[True]
    exclude_credential_theft: Literal[True]
    exclude_unauthorized_targeting: Literal[True]
    exclude_persistence_evasion_or_destructive_actions: Literal[True]


class EvalFamily(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    required_evidence: list[str] = Field(min_length=1)
    failure_modes: list[str] = Field(min_length=1)


class Generation(StrictModel):
    minimum_cases_per_family: int = Field(ge=32)
    modes: list[Mode] = Field(min_length=4, max_length=4)
    require_counterfactual_pair: Literal[True]
    require_evidence_removed_variant: Literal[True]
    require_cross_layer_causal_case: Literal[True]

    @model_validator(mode="after")
    def all_modes_once(self) -> "Generation":
        if set(self.modes) != {"positive", "hard_negative", "ambiguous", "abstention"}:
            raise ValueError("generation modes must contain all four supported modes")
        return self


class Config(StrictModel):
    schema_version: Literal["sentinel.agentic-security-protocol-state-evals.v1"]
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    target_architecture: TargetArchitecture
    safety_policy: SafetyPolicy
    reasoning_chain: list[str] = Field(min_length=10)
    forbidden_shortcuts: list[str] = Field(min_length=1)
    families: list[EvalFamily] = Field(min_length=1)
    scoring: dict[str, float]
    hard_fail_conditions: list[str] = Field(min_length=1)
    generation: Generation

    @model_validator(mode="after")
    def fail_closed_contract(self) -> "Config":
        if abs(sum(self.scoring.values()) - 1.0) > 1e-9 or any(v <= 0 for v in self.scoring.values()):
            raise ValueError("scoring weights must be positive and sum to 1.0")
        ids = [family.id for family in self.families]
        if len(ids) != len(set(ids)):
            raise ValueError("family ids must be unique")
        mandatory = {"valid_credential_implies_correct_principal", "feature_present_in_code_implies_activated_on_chain"}
        if not mandatory.issubset(set(self.forbidden_shortcuts)):
            raise ValueError("mandatory authority/protocol-state shortcuts cannot be removed")
        return self


class Seed(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    family: str = Field(pattern=r"^[a-z0-9_]+$")
    mode: Mode
    claim: str = Field(min_length=1)
    observed: list[str] = Field(min_length=1)
    expected: list[str] = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    confidence: Confidence
    must_abstain_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def abstention_has_boundary(self) -> "Seed":
        if self.mode == "abstention" and not self.must_abstain_from:
            raise ValueError("abstention seed must define must_abstain_from")
        return self


class Seeds(StrictModel):
    schema_version: Literal["sentinel.agentic-security-protocol-state-seeds.v1"]
    curriculum_ref: Literal["configs/training/agentic-security-protocol-state-evals.v1.json"]
    purpose: str = Field(min_length=1)
    cases: list[Seed] = Field(min_length=1)


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load valid JSON from {path}") from exc


def load_and_validate(config_path: Path, seeds_path: Path) -> tuple[Config, Seeds]:
    config = Config.model_validate(_json(config_path))
    seeds = Seeds.model_validate(_json(seeds_path))
    family_ids = {family.id for family in config.families}
    used = {case.family for case in seeds.cases}
    if used - family_ids:
        raise ValueError(f"unknown seed families: {sorted(used - family_ids)}")
    if family_ids - used:
        raise ValueError(f"every family requires a reviewed seed: {sorted(family_ids - used)}")
    ids = [case.id for case in seeds.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("seed ids must be unique")
    return config, seeds
