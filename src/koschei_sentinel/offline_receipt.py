from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.offline_job import OfflineTrainingJob
from koschei_sentinel.training import AdapterManifest, atomic_write, model_digest

_DIGEST = r"^[a-f0-9]{64}$"
_RUN_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class OfflineReceiptBlocked(ValueError):
    """Raised when a completed adapter does not match its sealed offline job."""


class OfflineTrainingReceipt(StrictModel):
    schema_version: Literal["sentinel.offline-training-receipt.v1"] = (
        "sentinel.offline-training-receipt.v1"
    )
    receipt_id: str = Field(pattern=_RUN_ID)
    state: Literal["completed_offline"] = "completed_offline"
    candidate_stage: Literal["incubation_candidate"] = "incubation_candidate"
    training_run_id: str = Field(pattern=_RUN_ID)
    job_digest: str = Field(pattern=_DIGEST)
    adapter_manifest_digest: str = Field(pattern=_DIGEST)
    adapter_digest: str = Field(pattern=_DIGEST)
    dataset_manifest_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    output_dir: str
    adapter_files: list[str] = Field(min_length=1, max_length=256)
    adapter_files_verified: Literal[True] = True
    benchmark_required: Literal[True] = True
    benchmark_passed: Literal[False] = False
    automatic_registration_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    receipt_digest: str = Field(pattern=_DIGEST)


def load_offline_job(path: str | Path) -> OfflineTrainingJob:
    try:
        job = OfflineTrainingJob.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid offline training job") from exc
    if job.job_digest != _model_payload_digest(job, "job_digest"):
        raise ValueError("offline training job digest does not match its contents")
    return job


def load_adapter_manifest(path: str | Path) -> AdapterManifest:
    try:
        return AdapterManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid adapter manifest") from exc


def build_offline_training_receipt(
    job: OfflineTrainingJob,
    adapter: AdapterManifest,
    *,
    root: str | Path = ".",
) -> OfflineTrainingReceipt:
    if job.training_run_id != adapter.run_id:
        raise OfflineReceiptBlocked("adapter run ID does not match offline job")
    if job.base_model != adapter.base_model or job.base_revision != adapter.base_revision:
        raise OfflineReceiptBlocked("adapter base model lineage does not match offline job")
    if job.dataset_manifest_digest != adapter.dataset_manifest_digest:
        raise OfflineReceiptBlocked("adapter dataset digest does not match offline job")
    if job.training_config_digest != adapter.training_config_digest:
        raise OfflineReceiptBlocked("adapter training config digest does not match offline job")
    if job.output_dir != adapter.output_dir:
        raise OfflineReceiptBlocked("adapter output directory does not match offline job")

    verified_digest = verify_adapter_files(adapter, root=root)
    if verified_digest != adapter.adapter_digest:
        raise OfflineReceiptBlocked("adapter files do not match adapter manifest digest")

    payload = {
        "schema_version": "sentinel.offline-training-receipt.v1",
        "receipt_id": adapter.run_id,
        "state": "completed_offline",
        "candidate_stage": "incubation_candidate",
        "training_run_id": adapter.run_id,
        "job_digest": job.job_digest,
        "adapter_manifest_digest": model_digest(adapter),
        "adapter_digest": adapter.adapter_digest,
        "dataset_manifest_digest": adapter.dataset_manifest_digest,
        "training_config_digest": adapter.training_config_digest,
        "base_model": adapter.base_model,
        "base_revision": adapter.base_revision,
        "output_dir": adapter.output_dir,
        "adapter_files": list(adapter.adapter_files),
        "adapter_files_verified": True,
        "benchmark_required": True,
        "benchmark_passed": False,
        "automatic_registration_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    return OfflineTrainingReceipt.model_validate(
        {**payload, "receipt_digest": _digest(payload)}
    )


def verify_adapter_files(
    adapter: AdapterManifest,
    *,
    root: str | Path = ".",
) -> str:
    root_path = Path(root).resolve()
    output_path = _resolve_under_root(root_path, adapter.output_dir)
    digest = hashlib.sha256()
    seen: set[str] = set()
    for relative in adapter.adapter_files:
        safe_relative = _safe_relative_path(relative, "adapter file")
        if safe_relative in seen:
            raise ValueError(f"duplicate adapter file: {safe_relative}")
        seen.add(safe_relative)
        path = _resolve_under_root(output_path, safe_relative)
        if not path.is_file():
            raise ValueError(f"adapter file is missing: {safe_relative}")
        digest.update(safe_relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_offline_training_receipt(
    receipt: OfflineTrainingReceipt,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"offline training receipt already exists: {destination}")
    payload = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(destination, payload)


def _model_payload_digest(model: StrictModel, digest_field: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(digest_field)
    return _digest(payload)


def _resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes the repository root")
    return candidate


def _safe_relative_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{label} must stay within its artifact root")
    return path.as_posix()


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
