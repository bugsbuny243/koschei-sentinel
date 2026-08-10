from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from koschei_sentinel.continued_pretraining import ContinuedPretrainingPlan
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"
_MAX_CHECKPOINT_FILES = 10_000
_MANIFEST_NAME = "checkpoint-manifest.json"
_EXECUTOR_ID = "koschei-stage2-executor/v1"


class ContinuedPretrainingExecutionEnvelope(StrictModel):
    schema_version: Literal["sentinel.continued-pretraining-execution.v1"] = (
        "sentinel.continued-pretraining-execution.v1"
    )
    lineage_stage: Literal["stage2_checkpoint_execution"] = "stage2_checkpoint_execution"
    executor_id: Literal["koschei-stage2-executor/v1"] = _EXECUTOR_ID
    run_id: str
    plan_path: str
    plan_file_digest: str = Field(pattern=_DIGEST)
    plan_digest: str = Field(pattern=_DIGEST)
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    corpus_file_digest: str = Field(pattern=_DIGEST)
    corpus_digest: str = Field(pattern=_DIGEST)
    corpus_audit_file_digest: str = Field(pattern=_DIGEST)
    corpus_policy_digest: str = Field(pattern=_DIGEST)
    holdout_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    max_sequence_length: int
    epochs: float
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    estimated_optimizer_steps: int
    seed: int
    output_dir: str


class CheckpointFile(StrictModel):
    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=_DIGEST)
    bytes: int = Field(ge=0)


class Stage2CheckpointManifest(StrictModel):
    schema_version: Literal["sentinel.stage2-checkpoint-manifest.v1"] = (
        "sentinel.stage2-checkpoint-manifest.v1"
    )
    lineage_stage: Literal["stage2_checkpoint"] = "stage2_checkpoint"
    executor_id: Literal["koschei-stage2-executor/v1"] = _EXECUTOR_ID
    run_id: str
    plan_file_digest: str = Field(pattern=_DIGEST)
    plan_digest: str = Field(pattern=_DIGEST)
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    corpus_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    files: list[CheckpointFile] = Field(min_length=1, max_length=_MAX_CHECKPOINT_FILES)
    checkpoint_digest: str = Field(pattern=_DIGEST)


def prepare_execution_envelope(
    plan_path: str | Path,
    *,
    root: str | Path = ".",
) -> ContinuedPretrainingExecutionEnvelope:
    root_path = Path(root).resolve()
    relative_plan = _relative_path(root_path, plan_path, "plan_path")
    plan, raw = _load_plan(root_path, relative_plan)
    output_path = _resolve_under_root(root_path, plan.output_dir)
    if output_path.exists():
        raise FileExistsError(
            f"continued-pretraining output already exists: {plan.output_dir}"
        )
    return _envelope_from_plan(plan, relative_plan, raw)


def finalize_checkpoint(
    envelope: ContinuedPretrainingExecutionEnvelope,
    *,
    root: str | Path = ".",
) -> Stage2CheckpointManifest:
    root_path = Path(root).resolve()
    _verify_envelope_plan(envelope, root_path)
    output = _resolve_under_root(root_path, envelope.output_dir)
    if not output.is_dir():
        raise ValueError("continued-pretraining checkpoint output directory is missing")

    files = _checkpoint_files(output)
    if not files:
        raise ValueError("checkpoint output contains no model artifacts")
    payload = _checkpoint_payload(envelope, files)
    return Stage2CheckpointManifest(
        run_id=envelope.run_id,
        plan_file_digest=envelope.plan_file_digest,
        plan_digest=envelope.plan_digest,
        base_model=envelope.base_model,
        base_revision=envelope.base_revision,
        corpus_digest=envelope.corpus_digest,
        benchmark_suite_digest=envelope.benchmark_suite_digest,
        training_config_digest=envelope.training_config_digest,
        files=files,
        checkpoint_digest=_digest(payload),
    )


def write_checkpoint_manifest(
    manifest: Stage2CheckpointManifest,
    envelope: ContinuedPretrainingExecutionEnvelope,
    *,
    root: str | Path = ".",
) -> Path:
    output = _resolve_under_root(Path(root).resolve(), envelope.output_dir)
    if not output.is_dir():
        raise ValueError("checkpoint output directory is missing")
    destination = output / _MANIFEST_NAME
    if destination.exists():
        raise FileExistsError(f"checkpoint manifest already exists: {destination}")
    _atomic_write(
        destination,
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    return destination


def verify_checkpoint_manifest(
    manifest_path: str | Path,
    envelope: ContinuedPretrainingExecutionEnvelope,
    *,
    root: str | Path = ".",
) -> Stage2CheckpointManifest:
    root_path = Path(root).resolve()
    _verify_envelope_plan(envelope, root_path)
    resolved = _resolve_file(
        root_path,
        _relative_path(root_path, manifest_path, "manifest_path"),
        "checkpoint manifest",
    )
    try:
        stored = Stage2CheckpointManifest.model_validate_json(resolved.read_bytes())
    except ValueError as exc:
        raise ValueError("invalid Stage 2 checkpoint manifest") from exc
    recomputed = finalize_checkpoint(envelope, root=root_path)
    if stored != recomputed:
        raise ValueError("checkpoint manifest is stale or checkpoint files were modified")
    return stored


def _load_plan(root: Path, relative_plan: str) -> tuple[ContinuedPretrainingPlan, bytes]:
    resolved_plan = _resolve_file(root, relative_plan, "continued-pretraining plan")
    raw = resolved_plan.read_bytes()
    try:
        plan = ContinuedPretrainingPlan.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("invalid continued-pretraining plan") from exc
    return plan, raw


def _envelope_from_plan(
    plan: ContinuedPretrainingPlan,
    relative_plan: str,
    raw: bytes,
) -> ContinuedPretrainingExecutionEnvelope:
    return ContinuedPretrainingExecutionEnvelope(
        run_id=plan.run_id,
        plan_path=relative_plan,
        plan_file_digest=hashlib.sha256(raw).hexdigest(),
        plan_digest=_model_digest(plan),
        base_model=plan.base_model,
        base_revision=plan.base_revision,
        corpus_file_digest=plan.corpus_file_digest,
        corpus_digest=plan.corpus_digest,
        corpus_audit_file_digest=plan.corpus_audit_file_digest,
        corpus_policy_digest=plan.corpus_policy_digest,
        holdout_digest=plan.holdout_digest,
        benchmark_suite_digest=plan.benchmark_suite_digest,
        training_config_digest=plan.training_config_digest,
        max_sequence_length=plan.max_sequence_length,
        epochs=plan.epochs,
        learning_rate=plan.learning_rate,
        per_device_batch_size=plan.per_device_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        effective_batch_size=plan.effective_batch_size,
        estimated_optimizer_steps=plan.estimated_optimizer_steps,
        seed=plan.seed,
        output_dir=plan.output_dir,
    )


def _verify_envelope_plan(
    envelope: ContinuedPretrainingExecutionEnvelope,
    root: Path,
) -> None:
    plan, raw = _load_plan(root, envelope.plan_path)
    if hashlib.sha256(raw).hexdigest() != envelope.plan_file_digest:
        raise ValueError("continued-pretraining plan file digest changed")
    if _model_digest(plan) != envelope.plan_digest:
        raise ValueError("continued-pretraining semantic plan digest changed")
    expected = _envelope_from_plan(plan, envelope.plan_path, raw)
    if envelope != expected:
        raise ValueError("execution envelope does not match continued-pretraining plan")


def _checkpoint_files(output: Path) -> list[CheckpointFile]:
    rows: list[CheckpointFile] = []
    for path in sorted(output.rglob("*")):
        if path.name == _MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise ValueError("checkpoint output may not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("checkpoint output contains an unsupported filesystem entry")
        relative = path.relative_to(output).as_posix()
        _validate_artifact_path(relative)
        rows.append(
            CheckpointFile(
                path=relative,
                sha256=_hash_file(path),
                bytes=path.stat().st_size,
            )
        )
        if len(rows) > _MAX_CHECKPOINT_FILES:
            raise ValueError("checkpoint output contains too many files")
    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_payload(
    envelope: ContinuedPretrainingExecutionEnvelope,
    files: list[CheckpointFile],
) -> dict[str, object]:
    return {
        "executor_id": envelope.executor_id,
        "run_id": envelope.run_id,
        "plan_file_digest": envelope.plan_file_digest,
        "plan_digest": envelope.plan_digest,
        "base_model": envelope.base_model,
        "base_revision": envelope.base_revision,
        "corpus_digest": envelope.corpus_digest,
        "benchmark_suite_digest": envelope.benchmark_suite_digest,
        "training_config_digest": envelope.training_config_digest,
        "files": [item.model_dump(mode="json") for item in files],
    }


def _model_digest(model: StrictModel) -> str:
    return _digest(model.model_dump(mode="json"))


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_file(root: Path, relative: str, label: str) -> Path:
    path = _resolve_under_root(root, relative)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {relative}")
    return path


def _resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes the repository root")
    return candidate


def _relative_path(root: Path, value: str | Path, field: str) -> str:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field} must stay within the repository root") from exc
        value = relative.as_posix()
    else:
        value = PurePosixPath(str(path)).as_posix()
    _validate_artifact_path(value)
    return value


def _validate_artifact_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError("artifact path must stay within its repository/checkpoint root")


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
