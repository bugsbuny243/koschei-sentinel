from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    CyberSFTPlan,
    load_cyber_sft_config,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class CyberSFTTrainingSourceBinding(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-training-source.v1"] = (
        "sentinel.cyber-sft-training-source.v1"
    )
    run_id: str
    repository_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_binding_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_sha256(config: CyberSFTConfig) -> str:
    return _sha256_bytes(canonical_json(config.model_dump(mode="json")).encode("utf-8"))


def _source_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("source_binding_sha256", None)
    return _sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def _validate_repository_commit(value: str) -> None:
    valid = len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)
    if not valid:
        raise ValueError("repository_commit must be a lowercase 40-character Git commit SHA")


def build_training_source_binding(
    *,
    config_path: str | Path,
    plan_path: str | Path,
    repository_commit: str,
) -> CyberSFTTrainingSourceBinding:
    _validate_repository_commit(repository_commit)
    config = load_cyber_sft_config(config_path)
    plan_raw = Path(plan_path).read_bytes()
    plan = CyberSFTPlan.model_validate_json(plan_raw)

    checks = (
        ("run_id", plan.run_id, config.run_id),
        ("stage", plan.stage, config.stage),
        ("execution_profile", plan.execution_profile, config.execution_profile),
        ("base_model", plan.base_model, config.base_model),
        ("base_revision", plan.base_revision, config.base_revision),
        ("output_dir", plan.output_dir, config.output_dir),
        ("input_adapter_dir", plan.input_adapter_dir, config.input_adapter_dir),
    )
    for label, observed, expected in checks:
        if observed != expected:
            raise ValueError(f"training source plan/config mismatch: {label}")

    payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-sft-training-source.v1",
        "run_id": config.run_id,
        "repository_commit": repository_commit,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "config_sha256": _config_sha256(config),
        "plan_sha256": _sha256_bytes(plan_raw),
        "corpus_examples_sha256": plan.corpus_examples_sha256,
        "corpus_manifest_sha256": plan.corpus_manifest_sha256,
    }
    payload["source_binding_sha256"] = _source_digest(payload)
    return CyberSFTTrainingSourceBinding.model_validate(payload)


def verify_training_source_binding(
    binding: CyberSFTTrainingSourceBinding,
    *,
    config_path: str | Path,
    plan_path: str | Path,
    repository_commit: str,
) -> None:
    expected = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit=repository_commit,
    )
    if binding.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("training source binding differs from current repo/config/plan")
    if _source_digest(binding.model_dump(mode="json")) != binding.source_binding_sha256:
        raise ValueError("training source binding self-hash does not verify")
