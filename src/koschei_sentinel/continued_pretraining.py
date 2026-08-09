from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.pretraining_corpus import (
    PretrainingCorpusAudit,
    audit_pretraining_corpus,
    load_pretraining_documents,
    load_pretraining_holdout,
    load_pretraining_policy,
)

_DIGEST = r"^[a-f0-9]{64}$"


class ContinuedPretrainingConfig(StrictModel):
    schema_version: Literal["sentinel.continued-pretraining-config.v1"] = (
        "sentinel.continued-pretraining-config.v1"
    )
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    base_model: str = Field(min_length=3, max_length=256)
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    corpus_path: str = Field(min_length=1, max_length=1024)
    corpus_audit_path: str = Field(min_length=1, max_length=1024)
    holdout_path: str = Field(min_length=1, max_length=1024)
    policy_path: str = Field(min_length=1, max_length=1024)
    output_dir: str = Field(min_length=1, max_length=1024)
    max_sequence_length: int = Field(default=4096, ge=512, le=32768)
    epochs: float = Field(default=1.0, gt=0.0, le=10.0)
    learning_rate: float = Field(default=0.00005, gt=0.0, le=0.01)
    per_device_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=32, ge=1, le=4096)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    trust_remote_code: Literal[False] = False

    @model_validator(mode="after")
    def inputs_are_immutable_and_local(self) -> ContinuedPretrainingConfig:
        for field in (
            "corpus_path",
            "corpus_audit_path",
            "holdout_path",
            "policy_path",
            "output_dir",
        ):
            _validate_relative_path(getattr(self, field), field)
        if self.base_model.startswith(("http://", "https://")):
            raise ValueError("base_model must be a registry identifier, not a URL")
        if self.base_model.count("/") != 1:
            raise ValueError("base_model must use owner/model format")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


class ContinuedPretrainingPlan(StrictModel):
    schema_version: Literal["sentinel.continued-pretraining-plan.v1"] = (
        "sentinel.continued-pretraining-plan.v1"
    )
    lineage_stage: Literal["stage2_continued_pretraining"] = (
        "stage2_continued_pretraining"
    )
    dry_run: Literal[True] = True
    run_id: str
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    corpus_file_digest: str = Field(pattern=_DIGEST)
    corpus_digest: str = Field(pattern=_DIGEST)
    corpus_audit_file_digest: str = Field(pattern=_DIGEST)
    corpus_policy_digest: str = Field(pattern=_DIGEST)
    holdout_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    documents: int = Field(ge=1)
    unique_families: int = Field(ge=0)
    effective_batch_size: int = Field(ge=1)
    estimated_optimizer_steps: int = Field(ge=1)
    max_sequence_length: int
    output_dir: str


def load_continued_pretraining_config(
    path: str | Path,
) -> ContinuedPretrainingConfig:
    try:
        return ContinuedPretrainingConfig.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid continued-pretraining config") from exc


def plan_continued_pretraining(
    config: ContinuedPretrainingConfig,
    *,
    root: str | Path = ".",
) -> ContinuedPretrainingPlan:
    root_path = Path(root).resolve()
    corpus_path = _resolve_file(root_path, config.corpus_path, "pretraining corpus")
    audit_path = _resolve_file(
        root_path,
        config.corpus_audit_path,
        "pretraining corpus audit",
    )
    holdout_path = _resolve_file(root_path, config.holdout_path, "pretraining holdout")
    policy_path = _resolve_file(root_path, config.policy_path, "pretraining policy")
    output_path = _resolve_under_root(root_path, config.output_dir)
    if output_path.exists():
        raise FileExistsError(
            f"continued-pretraining output already exists: {config.output_dir}"
        )

    documents = load_pretraining_documents(corpus_path)
    holdout = load_pretraining_holdout(holdout_path)
    policy = load_pretraining_policy(policy_path)
    recomputed = audit_pretraining_corpus(documents, holdout, policy)
    try:
        stored = PretrainingCorpusAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid pretraining corpus audit") from exc

    if not stored.ready:
        raise ValueError("continued pretraining requires a passing corpus audit")
    if stored != recomputed:
        raise ValueError(
            "pretraining corpus audit is stale or does not match corpus/policy/holdout"
        )
    if not recomputed.ready:
        raise ValueError("recomputed pretraining corpus audit did not pass")

    steps_per_epoch = max(
        1,
        (len(documents) + config.effective_batch_size - 1)
        // config.effective_batch_size,
    )
    estimated_steps = max(1, int(steps_per_epoch * config.epochs + 0.999999))

    return ContinuedPretrainingPlan(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_file_digest=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        corpus_digest=recomputed.corpus_digest,
        corpus_audit_file_digest=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        corpus_policy_digest=recomputed.policy_digest,
        holdout_digest=recomputed.holdout_digest,
        benchmark_suite_digest=recomputed.benchmark_suite_digest,
        training_config_digest=_model_digest(config),
        documents=recomputed.documents,
        unique_families=recomputed.unique_families,
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=estimated_steps,
        max_sequence_length=config.max_sequence_length,
        output_dir=config.output_dir,
    )


def write_continued_pretraining_plan(
    plan: ContinuedPretrainingPlan,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"continued-pretraining plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{field} must stay within the repository root")


def _model_digest(model: StrictModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
