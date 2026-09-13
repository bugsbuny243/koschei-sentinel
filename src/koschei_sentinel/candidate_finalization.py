from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.incubation_registry import (
    IncubationCandidateRecord,
    IncubationRegistry,
    append_candidate,
    build_candidate_record,
)
from koschei_sentinel.matrix import ComparisonMatrix
from koschei_sentinel.models import StrictModel
from koschei_sentinel.offline_receipt import OfflineTrainingReceipt
from koschei_sentinel.training import AdapterManifest, model_digest

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class CandidateFinalizationBlocked(ValueError):
    """Raised when a receipt, benchmark, or registry lineage cannot be finalized."""


class CandidateFinalization(StrictModel):
    schema_version: Literal["sentinel.candidate-finalization.v1"] = (
        "sentinel.candidate-finalization.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    state: Literal["finalized_incubation"] = "finalized_incubation"
    authority: Literal["explanation_only"] = "explanation_only"
    offline_receipt_digest: str = Field(pattern=_DIGEST)
    job_digest: str = Field(pattern=_DIGEST)
    adapter_manifest_digest: str = Field(pattern=_DIGEST)
    adapter_digest: str = Field(pattern=_DIGEST)
    dataset_manifest_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    comparison_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    benchmark_report_digest: str = Field(pattern=_DIGEST)
    candidate_record_digest: str = Field(pattern=_DIGEST)
    previous_registry_digest: str = Field(pattern=_DIGEST)
    updated_registry_digest: str = Field(pattern=_DIGEST)
    automatic_registry_replacement_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    finalization_digest: str = Field(pattern=_DIGEST)


def verify_candidate_finalization(finalization: CandidateFinalization) -> CandidateFinalization:
    payload = finalization.model_dump(mode="json")
    claimed = payload.pop("finalization_digest")
    if claimed != _digest(payload):
        raise CandidateFinalizationBlocked(
            "candidate finalization digest does not match its contents"
        )
    return finalization


def load_candidate_finalization(path: str | Path) -> CandidateFinalization:
    try:
        finalization = CandidateFinalization.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise CandidateFinalizationBlocked("invalid candidate finalization") from exc
    return verify_candidate_finalization(finalization)


def load_offline_training_receipt(path: str | Path) -> OfflineTrainingReceipt:
    try:
        receipt = OfflineTrainingReceipt.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid offline training receipt") from exc
    _require_model_digest(receipt, "receipt_digest", "offline training receipt")
    return receipt


def finalize_candidate(
    candidate_id: str,
    receipt: OfflineTrainingReceipt,
    autotrain: AutotrainPlan,
    adapter: AdapterManifest,
    comparison: ComparisonMatrix,
    registry: IncubationRegistry,
) -> tuple[CandidateFinalization, IncubationRegistry, IncubationCandidateRecord]:
    _require_model_digest(receipt, "receipt_digest", "offline training receipt")
    _require_comparison_digest(comparison)

    if receipt.training_run_id != candidate_id or receipt.receipt_id != candidate_id:
        raise CandidateFinalizationBlocked("receipt candidate ID does not match finalization")
    if receipt.adapter_manifest_digest != model_digest(adapter):
        raise CandidateFinalizationBlocked("receipt does not match the adapter manifest")
    if receipt.adapter_digest != adapter.adapter_digest:
        raise CandidateFinalizationBlocked("receipt adapter digest does not match adapter")
    if receipt.dataset_manifest_digest != adapter.dataset_manifest_digest:
        raise CandidateFinalizationBlocked("receipt dataset lineage does not match adapter")
    if receipt.training_config_digest != adapter.training_config_digest:
        raise CandidateFinalizationBlocked("receipt training config does not match adapter")
    if receipt.base_model != adapter.base_model or receipt.base_revision != adapter.base_revision:
        raise CandidateFinalizationBlocked("receipt base model lineage does not match adapter")
    if receipt.output_dir != adapter.output_dir:
        raise CandidateFinalizationBlocked("receipt output directory does not match adapter")

    record = build_candidate_record(candidate_id, autotrain, adapter, comparison)
    if record.adapter_manifest_digest != receipt.adapter_manifest_digest:
        raise CandidateFinalizationBlocked("candidate record does not match receipt manifest")
    if record.adapter_digest != receipt.adapter_digest:
        raise CandidateFinalizationBlocked("candidate record does not match receipt adapter")

    updated_registry = append_candidate(registry, record)
    outcome = next(
        item for item in comparison.candidates if item.candidate_id == candidate_id
    )
    if outcome.report is None:
        raise CandidateFinalizationBlocked("benchmark report is missing")

    payload = {
        "schema_version": "sentinel.candidate-finalization.v1",
        "candidate_id": candidate_id,
        "state": "finalized_incubation",
        "authority": "explanation_only",
        "offline_receipt_digest": receipt.receipt_digest,
        "job_digest": receipt.job_digest,
        "adapter_manifest_digest": receipt.adapter_manifest_digest,
        "adapter_digest": receipt.adapter_digest,
        "dataset_manifest_digest": receipt.dataset_manifest_digest,
        "training_config_digest": receipt.training_config_digest,
        "comparison_digest": comparison.comparison_digest,
        "benchmark_suite_digest": comparison.suite_digest,
        "benchmark_report_digest": model_digest(outcome.report),
        "candidate_record_digest": record.record_digest,
        "previous_registry_digest": registry.registry_digest,
        "updated_registry_digest": updated_registry.registry_digest,
        "automatic_registry_replacement_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    finalization = CandidateFinalization.model_validate(
        {**payload, "finalization_digest": _digest(payload)}
    )
    return finalization, updated_registry, record


def write_finalization_bundle(
    finalization: CandidateFinalization,
    registry: IncubationRegistry,
    output_dir: str | Path,
) -> None:
    verify_candidate_finalization(finalization)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"candidate finalization already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_json(staging / "candidate-finalization.json", finalization)
        _write_json(staging / "incubation-registry.json", registry)
        _fsync_tree(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _require_model_digest(model: StrictModel, field: str, label: str) -> None:
    payload = model.model_dump(mode="json")
    claimed = payload.pop(field)
    if claimed != _digest(payload):
        raise CandidateFinalizationBlocked(f"{label} digest does not match its contents")


def _require_comparison_digest(comparison: ComparisonMatrix) -> None:
    payload = {
        "suite_digest": comparison.suite_digest,
        "registry_digest": comparison.registry_digest,
        "thresholds": comparison.thresholds.model_dump(mode="json"),
        "ranking": comparison.ranking,
        "candidates": [item.model_dump(mode="json") for item in comparison.candidates],
    }
    if comparison.comparison_digest != _digest(payload):
        raise CandidateFinalizationBlocked(
            "comparison matrix digest does not match its contents"
        )


def _write_json(path: Path, model: StrictModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()