from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

_TARGET_TOTAL_B = 397.0
_TARGET_ACTIVE_B = 35.0
_TOLERANCE_B = 0.5


class CandidateFrameworkSupport(StrictModel):
    transformers: str = Field(min_length=1)
    ms_swift: str = Field(min_length=1)
    megatron_swift: str = Field(min_length=1)


class CandidateFit(StrictModel):
    total_target_match: bool
    active_target_match: bool
    exact_target_match: bool


class ProductionTargetCandidate(StrictModel):
    model_ref: str = Field(min_length=1)
    total_parameters_billion: float = Field(gt=0)
    active_parameters_billion: float = Field(gt=0)
    architecture: str = Field(min_length=1)
    license: str = Field(min_length=1)
    framework: CandidateFrameworkSupport
    fit: CandidateFit
    role: str = Field(min_length=1)
    production_binding_allowed: Literal[False] = False
    evidence: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def fit_is_derived_not_claimed(self) -> "ProductionTargetCandidate":
        total_match = abs(self.total_parameters_billion - _TARGET_TOTAL_B) <= _TOLERANCE_B
        active_match = abs(self.active_parameters_billion - _TARGET_ACTIVE_B) <= _TOLERANCE_B
        exact_match = total_match and active_match
        if self.fit.total_target_match != total_match:
            raise ValueError("candidate total_target_match does not match measured parameter count")
        if self.fit.active_target_match != active_match:
            raise ValueError("candidate active_target_match does not match measured parameter count")
        if self.fit.exact_target_match != exact_match:
            raise ValueError("candidate exact_target_match must be derived from total and active matches")
        if self.active_parameters_billion > self.total_parameters_billion:
            raise ValueError("active parameters cannot exceed total parameters")
        return self


class CandidateMatrixTarget(StrictModel):
    total_parameters_billion: Literal[397.0] = 397.0
    active_parameters_billion: Literal[35.0] = 35.0
    architecture_class: Literal["sparse-moe"] = "sparse-moe"


class CandidateMatrixDecision(StrictModel):
    exact_public_base_found: bool
    production_binding_allowed: Literal[False] = False
    systems_validation_primary: str
    active_width_reference: str
    balanced_moe_reference: str
    notes: list[str] = Field(min_length=1)


class ProductionTargetCandidateMatrix(StrictModel):
    schema_version: Literal["sentinel.production-target-candidate-matrix.v1"] = (
        "sentinel.production-target-candidate-matrix.v1"
    )
    as_of: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    target: CandidateMatrixTarget
    decision: CandidateMatrixDecision
    candidates: list[ProductionTargetCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def decision_is_fail_closed(self) -> "ProductionTargetCandidateMatrix":
        refs = [candidate.model_ref for candidate in self.candidates]
        if len(refs) != len(set(refs)):
            raise ValueError("candidate matrix contains duplicate model_ref entries")
        exact = [candidate for candidate in self.candidates if candidate.fit.exact_target_match]
        if self.decision.exact_public_base_found != bool(exact):
            raise ValueError("exact_public_base_found does not match candidate evidence")
        if self.decision.systems_validation_primary not in refs:
            raise ValueError("systems_validation_primary must reference a reviewed candidate")
        if self.decision.active_width_reference not in refs:
            raise ValueError("active_width_reference must reference a reviewed candidate")
        if self.decision.balanced_moe_reference not in refs:
            raise ValueError("balanced_moe_reference must reference a reviewed candidate")
        if exact:
            raise ValueError(
                "an exact candidate requires a separate pinned-source intake and cannot be auto-bound from research metadata"
            )
        return self


def load_production_target_candidate_matrix(path: str | Path) -> ProductionTargetCandidateMatrix:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production target candidate matrix: {source}") from exc
    return ProductionTargetCandidateMatrix.model_validate(payload)
