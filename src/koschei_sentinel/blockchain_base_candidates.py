from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write

_COMMIT = r"^[a-f0-9]{40}$"
_DIGEST = r"^[a-f0-9]{64}$"
_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MODEL_ID = r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$"


class BaseCandidateLane(StrEnum):
    FEASIBILITY = "FEASIBILITY"
    EFFICIENCY = "EFFICIENCY"
    POWER = "POWER"


class LicenseReviewStatus(StrEnum):
    ALLOWLISTED_OPEN_LICENSE = "ALLOWLISTED_OPEN_LICENSE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class RuntimeIntegrationStatus(StrEnum):
    NATIVE_TRANSFORMERS_EXPECTED = "NATIVE_TRANSFORMERS_EXPECTED"
    CUSTOM_CODE_REVIEW_REQUIRED = "CUSTOM_CODE_REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class PreflightStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    PASSED = "PASSED"
    FAILED = "FAILED"


class BlockchainBaseCandidate(StrictModel):
    schema_version: Literal["sentinel.blockchain-base-candidate.v1"] = (
        "sentinel.blockchain-base-candidate.v1"
    )
    candidate_id: str = Field(pattern=_ID)
    lane: BaseCandidateLane
    model_id: str = Field(pattern=_MODEL_ID)
    revision: str = Field(pattern=_COMMIT)
    training_stage: Literal["BASE_PRETRAINING"] = "BASE_PRETRAINING"
    declared_total_parameters_billion: float = Field(gt=0.0, le=10_000.0)
    declared_active_parameters_billion: float = Field(gt=0.0, le=10_000.0)
    declared_context_length_tokens: int | None = Field(default=None, ge=1024, le=10_000_000)
    license_id: str = Field(min_length=2, max_length=128)
    license_review_status: LicenseReviewStatus
    commercial_use_declared: bool
    trust_remote_code_required: bool
    runtime_integration_status: RuntimeIntegrationStatus
    metadata_source: Literal["OFFICIAL_MODEL_CARD"] = "OFFICIAL_MODEL_CARD"
    preflight_status: PreflightStatus = PreflightStatus.NOT_RUN
    hardware_plan_approved: bool = False
    training_authorized: Literal[False] = False
    blocked_reasons: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def candidate_is_fail_closed(self) -> BlockchainBaseCandidate:
        if self.declared_active_parameters_billion > self.declared_total_parameters_billion:
            raise ValueError("active parameter count may not exceed total parameter count")
        if self.license_review_status is LicenseReviewStatus.REJECTED:
            if "license_rejected" not in self.blocked_reasons:
                raise ValueError("rejected license must be represented in blocked_reasons")
        if self.runtime_integration_status is RuntimeIntegrationStatus.REJECTED:
            if "runtime_rejected" not in self.blocked_reasons:
                raise ValueError("rejected runtime must be represented in blocked_reasons")
        if self.trust_remote_code_required and (
            self.runtime_integration_status
            is not RuntimeIntegrationStatus.CUSTOM_CODE_REVIEW_REQUIRED
        ):
            raise ValueError(
                "remote-code model must remain behind custom-code runtime review"
            )
        if not self.trust_remote_code_required and (
            self.runtime_integration_status
            is RuntimeIntegrationStatus.CUSTOM_CODE_REVIEW_REQUIRED
        ):
            raise ValueError("custom-code review may only be required for remote-code models")
        if self.preflight_status is PreflightStatus.PASSED and not self.hardware_plan_approved:
            raise ValueError("passed preflight requires an approved hardware plan")
        return self

    @property
    def admitted_for_preflight(self) -> bool:
        return (
            self.license_review_status is not LicenseReviewStatus.REJECTED
            and self.runtime_integration_status is not RuntimeIntegrationStatus.REJECTED
            and self.preflight_status is not PreflightStatus.FAILED
        )

    @property
    def ready_for_training_authorization_review(self) -> bool:
        return (
            self.license_review_status is LicenseReviewStatus.ALLOWLISTED_OPEN_LICENSE
            and self.runtime_integration_status
            is RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED
            and self.preflight_status is PreflightStatus.PASSED
            and self.hardware_plan_approved
            and not self.blocked_reasons
        )


class BlockchainBaseCandidateRegistry(StrictModel):
    schema_version: Literal["sentinel.blockchain-base-candidate-registry.v1"] = (
        "sentinel.blockchain-base-candidate-registry.v1"
    )
    registry_id: str = Field(pattern=_ID)
    candidates: list[BlockchainBaseCandidate] = Field(min_length=1, max_length=128)
    registry_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def registry_is_unique_and_bound(self) -> BlockchainBaseCandidateRegistry:
        identifiers = [item.candidate_id for item in self.candidates]
        model_ids = [item.model_id for item in self.candidates]
        pins = [(item.model_id, item.revision) for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("base candidate registry contains duplicate candidate_id")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("base candidate registry contains duplicate model_id")
        if len(pins) != len(set(pins)):
            raise ValueError("base candidate registry contains duplicate model revision pin")
        payload = self.model_dump(mode="json")
        expected = payload.pop("registry_digest")
        if _digest(payload) != expected:
            raise ValueError("base candidate registry digest mismatch")
        return self


class BlockchainBaseCandidatePolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-base-candidate-policy.v1"] = (
        "sentinel.blockchain-base-candidate-policy.v1"
    )
    policy_id: str = Field(pattern=_ID)
    min_candidates: int = Field(default=3, ge=2, le=128)
    required_lanes: list[BaseCandidateLane] = Field(min_length=1)
    require_exact_revision_pin: Literal[True] = True
    require_base_pretraining_stage: Literal[True] = True
    require_unique_models: Literal[True] = True
    require_training_unauthorized_until_preflight: Literal[True] = True
    allow_remote_code_without_review: Literal[False] = False
    allow_custom_license_without_review: Literal[False] = False

    @model_validator(mode="after")
    def lanes_are_unique(self) -> BlockchainBaseCandidatePolicy:
        if len(self.required_lanes) != len(set(self.required_lanes)):
            raise ValueError("required_lanes must be unique")
        return self


class BlockchainBaseCandidateAudit(StrictModel):
    schema_version: Literal["sentinel.blockchain-base-candidate-audit.v1"] = (
        "sentinel.blockchain-base-candidate-audit.v1"
    )
    ready_for_preflight: bool
    policy_id: str
    registry_id: str
    registry_digest: str = Field(pattern=_DIGEST)
    candidates: int = Field(ge=1)
    admitted_for_preflight: int = Field(ge=0)
    ready_for_training_authorization_review: int = Field(ge=0)
    lane_counts: dict[str, int]
    license_review_counts: dict[str, int]
    runtime_integration_counts: dict[str, int]
    preflight_counts: dict[str, int]
    blocked_candidates: dict[str, list[str]]
    violations: list[str]
    training_started: Literal[False] = False
    production_authority: Literal[False] = False


def build_base_candidate_registry(
    registry_id: str,
    candidates: list[BlockchainBaseCandidate],
) -> BlockchainBaseCandidateRegistry:
    payload = {
        "schema_version": "sentinel.blockchain-base-candidate-registry.v1",
        "registry_id": registry_id,
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    return BlockchainBaseCandidateRegistry.model_validate(
        {**payload, "registry_digest": _digest(payload)}
    )


def load_base_candidate_registry(path: str | Path) -> BlockchainBaseCandidateRegistry:
    try:
        return BlockchainBaseCandidateRegistry.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain base candidate registry") from exc


def load_base_candidate_policy(path: str | Path) -> BlockchainBaseCandidatePolicy:
    try:
        return BlockchainBaseCandidatePolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain base candidate policy") from exc


def audit_base_candidate_registry(
    registry: BlockchainBaseCandidateRegistry,
    policy: BlockchainBaseCandidatePolicy,
) -> BlockchainBaseCandidateAudit:
    violations: list[str] = []
    if len(registry.candidates) < policy.min_candidates:
        violations.append(
            f"candidates {len(registry.candidates)} below minimum {policy.min_candidates}"
        )

    lane_counts: dict[str, int] = {}
    license_counts: dict[str, int] = {}
    runtime_counts: dict[str, int] = {}
    preflight_counts: dict[str, int] = {}
    blocked: dict[str, list[str]] = {}
    admitted = 0
    training_review_ready = 0

    for item in registry.candidates:
        lane_counts[item.lane.value] = lane_counts.get(item.lane.value, 0) + 1
        license_counts[item.license_review_status.value] = (
            license_counts.get(item.license_review_status.value, 0) + 1
        )
        runtime_counts[item.runtime_integration_status.value] = (
            runtime_counts.get(item.runtime_integration_status.value, 0) + 1
        )
        preflight_counts[item.preflight_status.value] = (
            preflight_counts.get(item.preflight_status.value, 0) + 1
        )
        if item.admitted_for_preflight:
            admitted += 1
        if item.ready_for_training_authorization_review:
            training_review_ready += 1
        if item.blocked_reasons:
            blocked[item.candidate_id] = list(item.blocked_reasons)
        if item.training_authorized is not False:
            violations.append(f"candidate {item.candidate_id} is prematurely training-authorized")
        if (
            item.trust_remote_code_required
            and item.runtime_integration_status
            is not RuntimeIntegrationStatus.CUSTOM_CODE_REVIEW_REQUIRED
        ):
            violations.append(f"candidate {item.candidate_id} bypasses remote-code review")
        if (
            item.license_review_status is LicenseReviewStatus.MANUAL_REVIEW_REQUIRED
            and "license_review_pending" not in item.blocked_reasons
        ):
            violations.append(
                f"candidate {item.candidate_id} lacks explicit pending license-review block"
            )

    missing_lanes = sorted(
        item.value for item in policy.required_lanes if lane_counts.get(item.value, 0) == 0
    )
    if missing_lanes:
        violations.append("required candidate lanes missing: " + ", ".join(missing_lanes))

    if admitted < policy.min_candidates:
        violations.append(
            f"preflight-admitted candidates {admitted} below minimum {policy.min_candidates}"
        )

    return BlockchainBaseCandidateAudit(
        ready_for_preflight=not violations,
        policy_id=policy.policy_id,
        registry_id=registry.registry_id,
        registry_digest=registry.registry_digest,
        candidates=len(registry.candidates),
        admitted_for_preflight=admitted,
        ready_for_training_authorization_review=training_review_ready,
        lane_counts=dict(sorted(lane_counts.items())),
        license_review_counts=dict(sorted(license_counts.items())),
        runtime_integration_counts=dict(sorted(runtime_counts.items())),
        preflight_counts=dict(sorted(preflight_counts.items())),
        blocked_candidates=dict(sorted(blocked.items())),
        violations=violations,
    )


def write_base_candidate_audit(
    audit: BlockchainBaseCandidateAudit,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"blockchain base candidate audit already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
