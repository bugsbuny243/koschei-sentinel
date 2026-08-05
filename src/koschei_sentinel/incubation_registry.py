from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.matrix import ComparisonMatrix
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import AdapterManifest, atomic_write, model_digest

_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_DIGEST = r"^[a-f0-9]{64}$"


class CandidateRegistrationBlocked(ValueError):
    """Raised when valid artifacts do not qualify for incubation registration."""


class IncubationCandidateRecord(StrictModel):
    schema_version: Literal["sentinel.incubation-candidate.v1"] = (
        "sentinel.incubation-candidate.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    stage: Literal["incubation_candidate"] = "incubation_candidate"
    authority: Literal["explanation_only"] = "explanation_only"
    run_id: str = Field(pattern=_CANDIDATE_ID)
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    adapter_digest: str = Field(pattern=_DIGEST)
    adapter_manifest_digest: str = Field(pattern=_DIGEST)
    dataset_manifest_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    autotrain_plan_digest: str = Field(pattern=_DIGEST)
    comparison_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    benchmark_report_digest: str = Field(pattern=_DIGEST)
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    record_digest: str = Field(pattern=_DIGEST)


class IncubationRegistry(StrictModel):
    schema_version: Literal["sentinel.incubation-registry.v1"] = (
        "sentinel.incubation-registry.v1"
    )
    records: list[IncubationCandidateRecord] = Field(default_factory=list, max_length=10_000)
    registry_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def records_are_unique(self) -> IncubationRegistry:
        for field in ("candidate_id", "adapter_digest", "record_digest"):
            values = [getattr(record, field) for record in self.records]
            if len(values) != len(set(values)):
                raise ValueError(f"registry contains duplicate {field} values")
        return self


def load_autotrain_plan(path: str | Path) -> AutotrainPlan:
    return _load_model(path, AutotrainPlan, "autotrain plan")


def load_adapter_manifest(path: str | Path) -> AdapterManifest:
    return _load_model(path, AdapterManifest, "adapter manifest")


def load_comparison_matrix(path: str | Path) -> ComparisonMatrix:
    return _load_model(path, ComparisonMatrix, "comparison matrix")


def build_candidate_record(
    candidate_id: str,
    autotrain: AutotrainPlan,
    adapter: AdapterManifest,
    comparison: ComparisonMatrix,
) -> IncubationCandidateRecord:
    if autotrain.decision != "ready_for_offline_training":
        raise CandidateRegistrationBlocked("autotrain plan did not authorize offline training")
    if autotrain.training_run_id != candidate_id:
        raise CandidateRegistrationBlocked("candidate_id does not match autotrain training_run_id")
    if adapter.run_id != candidate_id:
        raise CandidateRegistrationBlocked("candidate_id does not match adapter run_id")
    if adapter.dataset_manifest_digest != autotrain.dataset_manifest_digest:
        raise CandidateRegistrationBlocked(
            "adapter dataset digest does not match the approved autotrain plan"
        )

    outcome = next(
        (item for item in comparison.candidates if item.candidate_id == candidate_id),
        None,
    )
    if outcome is None:
        raise CandidateRegistrationBlocked("candidate is missing from comparison matrix")
    if outcome.status != "passed" or outcome.report is None or not outcome.report.gate_passed:
        raise CandidateRegistrationBlocked("candidate did not pass the benchmark hard gates")
    if candidate_id not in comparison.eligible_candidates:
        raise CandidateRegistrationBlocked("candidate is not listed as benchmark eligible")
    if outcome.report.suite_digest != comparison.suite_digest:
        raise CandidateRegistrationBlocked(
            "candidate report suite does not match comparison matrix"
        )

    payload = {
        "schema_version": "sentinel.incubation-candidate.v1",
        "candidate_id": candidate_id,
        "stage": "incubation_candidate",
        "authority": "explanation_only",
        "run_id": adapter.run_id,
        "base_model": adapter.base_model,
        "base_revision": adapter.base_revision,
        "adapter_digest": adapter.adapter_digest,
        "adapter_manifest_digest": model_digest(adapter),
        "dataset_manifest_digest": adapter.dataset_manifest_digest,
        "training_config_digest": adapter.training_config_digest,
        "autotrain_plan_digest": model_digest(autotrain),
        "comparison_digest": comparison.comparison_digest,
        "benchmark_suite_digest": comparison.suite_digest,
        "benchmark_report_digest": model_digest(outcome.report),
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    return IncubationCandidateRecord.model_validate(
        {**payload, "record_digest": _digest(payload)}
    )


def empty_registry() -> IncubationRegistry:
    return _build_registry([])


def load_registry(path: str | Path) -> IncubationRegistry:
    source = Path(path)
    if not source.exists():
        return empty_registry()
    registry = _load_model(source, IncubationRegistry, "incubation registry")
    if registry.registry_digest != _registry_digest(registry.records):
        raise ValueError("incubation registry digest does not match its records")
    for record in registry.records:
        if record.record_digest != _record_digest(record):
            raise ValueError(f"candidate record digest mismatch: {record.candidate_id}")
    return registry


def append_candidate(
    registry: IncubationRegistry,
    record: IncubationCandidateRecord,
) -> IncubationRegistry:
    if any(item.candidate_id == record.candidate_id for item in registry.records):
        raise ValueError(f"candidate_id is already registered: {record.candidate_id}")
    if any(item.adapter_digest == record.adapter_digest for item in registry.records):
        raise ValueError("adapter digest is already registered")
    return _build_registry([*registry.records, record])


def write_registry(registry: IncubationRegistry, path: str | Path) -> None:
    payload = json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)


def _build_registry(records: list[IncubationCandidateRecord]) -> IncubationRegistry:
    return IncubationRegistry(records=records, registry_digest=_registry_digest(records))


def _registry_digest(records: list[IncubationCandidateRecord]) -> str:
    payload = [record.model_dump(mode="json") for record in records]
    return _digest(payload)


def _record_digest(record: IncubationCandidateRecord) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("record_digest")
    return _digest(payload)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_model(path: str | Path, model_type: type[StrictModel], label: str):
    try:
        return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
