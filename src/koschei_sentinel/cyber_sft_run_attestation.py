from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_model_access_preflight import CyberModelAccessPreflight
from koschei_sentinel.cyber_sft_artifact_verify import (
    CyberSFTArtifactVerification,
    verify_cyber_sft_run,
)
from koschei_sentinel.cyber_sft_trainer import (
    CyberSFTAdapterManifest,
    CyberSFTTrainingReceipt,
)
from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    CyberSFTPlan,
    load_cyber_sft_config,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json, resolve_under_root


class CyberSFTRunAttestation(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-run-attestation.v1"] = (
        "sentinel.cyber-sft-run-attestation.v1"
    )
    run_id: str
    selected_profile: Literal["normal", "lowmem"]
    repository_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    resolved_model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_preflight_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_runtime_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    resume_runtime_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    global_step: int = Field(gt=0)
    resumed: bool
    resume_checkpoint: str | None
    smoke_only: bool
    promotion_eligible: bool | None
    attestation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_sha256(config: CyberSFTConfig) -> str:
    return _sha256_bytes(canonical_json(config.model_dump(mode="json")).encode("utf-8"))


def _attestation_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("attestation_sha256", None)
    return _sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw


def build_cyber_sft_run_attestation(
    *,
    config_path: str | Path,
    plan_path: str | Path,
    run_dir: str,
    model_preflight_path: str | Path,
    verification_path: str | Path,
    selected_profile: Literal["normal", "lowmem"],
    repository_commit: str,
    root: str | Path = ".",
) -> CyberSFTRunAttestation:
    commit_is_valid = len(repository_commit) == 40 and all(
        ch in "0123456789abcdef" for ch in repository_commit
    )
    if not commit_is_valid:
        raise ValueError("repository_commit must be a lowercase 40-character Git commit SHA")

    root_path = Path(root).resolve()
    config = load_cyber_sft_config(config_path)
    plan_raw = Path(plan_path).read_bytes()
    plan = CyberSFTPlan.model_validate_json(plan_raw)
    run_path = resolve_under_root(root_path, run_dir)
    configured_run_path = resolve_under_root(root_path, config.output_dir)
    if run_path != configured_run_path:
        raise ValueError(
            "attestation run_dir differs from the selected training config output_dir"
        )

    manifest = CyberSFTAdapterManifest.model_validate_json(
        (run_path / "adapter-manifest.json").read_bytes()
    )
    receipt = CyberSFTTrainingReceipt.model_validate_json(
        (run_path / "training-receipt.json").read_bytes()
    )

    plan_checks = (
        ("run_id", plan.run_id, config.run_id),
        ("stage", plan.stage, config.stage),
        ("execution_profile", plan.execution_profile, config.execution_profile),
        ("base_model", plan.base_model, config.base_model),
        ("base_revision", plan.base_revision, config.base_revision),
        ("output_dir", plan.output_dir, config.output_dir),
        ("input_adapter_dir", plan.input_adapter_dir, config.input_adapter_dir),
        (
            "corpus examples",
            plan.corpus_examples_sha256,
            manifest.corpus_examples_sha256,
        ),
        (
            "corpus manifest",
            plan.corpus_manifest_sha256,
            manifest.corpus_manifest_sha256,
        ),
    )
    for label, observed, expected in plan_checks:
        if observed != expected:
            raise ValueError(f"Cyber SFT plan binding mismatch: {label}")
    if not plan.executable_with_current_trainer:
        raise ValueError("Cyber SFT plan is not executable with the current trainer")

    model_preflight_raw = Path(model_preflight_path).read_bytes()
    model_preflight = CyberModelAccessPreflight.model_validate_json(model_preflight_raw)
    if not model_preflight.ready or not model_preflight.public_ungated:
        raise ValueError("model preflight is not ready/public-ungated")
    if (
        model_preflight.base_model != config.base_model
        or model_preflight.requested_revision != config.base_revision
        or model_preflight.resolved_revision != config.base_revision
    ):
        raise ValueError(
            "model preflight does not bind the selected config to its exact revision"
        )
    if (
        model_preflight.model_type != "qwen3_5"
        or model_preflight.causal_lm_class != "Qwen3_5ForCausalLM"
    ):
        raise ValueError(
            "model preflight does not prove the expected Qwen3.5 text CausalLM mapping"
        )

    supplied_verification_raw = Path(verification_path).read_bytes()
    supplied_verification = CyberSFTArtifactVerification.model_validate_json(
        supplied_verification_raw
    )
    fresh_verification = verify_cyber_sft_run(run_dir, root=root_path)
    if not fresh_verification.valid:
        raise ValueError("fresh Cyber SFT artifact verification is invalid")
    supplied_payload = supplied_verification.model_dump(mode="json")
    fresh_payload = fresh_verification.model_dump(mode="json")
    if supplied_payload != fresh_payload:
        raise ValueError("supplied verification report differs from a fresh run verification")

    model_runtime, model_runtime_raw = _load_json_object(
        run_path / "model-runtime.json",
        label="model-runtime.json",
    )
    if model_runtime != {
        "expected_model_class": "Qwen3_5ForCausalLM",
        "loader": "AutoModelForCausalLM",
        "model_class": "Qwen3_5ForCausalLM",
        "text_only": True,
    }:
        raise ValueError("model-runtime.json does not prove the expected text-only runtime")

    resume_runtime, resume_runtime_raw = _load_json_object(
        run_path / "resume-runtime.json",
        label="resume-runtime.json",
    )
    expected_binding = {
        "schema_version": "sentinel.cyber-sft-resume-binding.v1",
        "run_id": config.run_id,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "corpus_examples_sha256": manifest.corpus_examples_sha256,
        "corpus_manifest_sha256": manifest.corpus_manifest_sha256,
        "config_sha256": _config_sha256(config),
    }
    if resume_runtime.get("resume_binding") != expected_binding:
        raise ValueError(
            "resume-runtime.json is not bound to the selected model/corpus/config"
        )
    resumed = resume_runtime.get("resumed")
    resume_checkpoint = resume_runtime.get("resume_checkpoint")
    if not isinstance(resumed, bool):
        raise ValueError("resume-runtime.json resumed must be boolean")
    if resume_checkpoint is not None and not isinstance(resume_checkpoint, str):
        raise ValueError("resume-runtime.json resume_checkpoint must be a string or null")
    if resumed != (resume_checkpoint is not None):
        raise ValueError("resume-runtime.json resumed flag and checkpoint disagree")

    binding_checks = (
        ("run_id", manifest.run_id, config.run_id),
        ("receipt run_id", receipt.run_id, config.run_id),
        ("base_model", manifest.base_model, config.base_model),
        ("base_revision", manifest.base_revision, config.base_revision),
        ("receipt base_revision", receipt.base_revision, config.base_revision),
        ("receipt digest", receipt.adapter_digest, manifest.adapter_digest),
    )
    for label, observed, expected in binding_checks:
        if observed != expected:
            raise ValueError(f"run attestation binding mismatch: {label}")
    if receipt.global_step <= 0:
        raise ValueError("run attestation requires at least one completed optimizer step")

    payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-sft-run-attestation.v1",
        "run_id": config.run_id,
        "selected_profile": selected_profile,
        "repository_commit": repository_commit,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "resolved_model_revision": model_preflight.resolved_revision,
        "config_sha256": _config_sha256(config),
        "plan_sha256": _sha256_bytes(plan_raw),
        "model_preflight_sha256": _sha256_bytes(model_preflight_raw),
        "verification_sha256": _sha256_bytes(supplied_verification_raw),
        "model_runtime_sha256": _sha256_bytes(model_runtime_raw),
        "resume_runtime_sha256": _sha256_bytes(resume_runtime_raw),
        "receipt_sha256": receipt.receipt_sha256,
        "adapter_digest": manifest.adapter_digest,
        "corpus_examples_sha256": manifest.corpus_examples_sha256,
        "corpus_manifest_sha256": manifest.corpus_manifest_sha256,
        "global_step": receipt.global_step,
        "resumed": resumed,
        "resume_checkpoint": resume_checkpoint,
        "smoke_only": manifest.corpus_promotion_eligible is False,
        "promotion_eligible": manifest.corpus_promotion_eligible,
    }
    payload["attestation_sha256"] = _attestation_digest(payload)
    return CyberSFTRunAttestation.model_validate(payload)
