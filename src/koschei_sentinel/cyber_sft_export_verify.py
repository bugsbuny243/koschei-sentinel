from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from koschei_sentinel.cyber_model_access_preflight import CyberModelAccessPreflight
from koschei_sentinel.cyber_sft_artifact_verify import (
    CyberSFTArtifactVerification,
    verify_cyber_sft_run,
)
from koschei_sentinel.cyber_sft_run_attestation import (
    CyberSFTRunAttestation,
    _attestation_digest,
    _config_sha256,
)
from koschei_sentinel.cyber_sft_trainer import (
    CyberSFTAdapterManifest,
    CyberSFTTrainingReceipt,
)
from koschei_sentinel.cyber_sft_training import CyberSFTPlan, load_cyber_sft_config
from koschei_sentinel.models import StrictModel


class CyberSFTExportVerification(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-export-verification.v1"] = (
        "sentinel.cyber-sft-export-verification.v1"
    )
    run_id: str | None
    valid: bool
    attestation_sha256_verified: bool
    config_sha256_verified: bool
    plan_sha256_verified: bool
    model_preflight_sha256_verified: bool
    verification_sha256_verified: bool
    model_runtime_sha256_verified: bool
    resume_runtime_sha256_verified: bool
    fresh_run_verification_valid: bool
    receipt_binding_verified: bool
    adapter_digest_verified: bool
    corpus_examples_sha256_verified: bool
    corpus_manifest_sha256_verified: bool
    profile_binding_verified: bool
    repository_commit_binding_verified: bool
    violations: list[str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _invalid(
    *,
    violations: list[str],
    run_id: str | None = None,
) -> CyberSFTExportVerification:
    return CyberSFTExportVerification(
        run_id=run_id,
        valid=False,
        attestation_sha256_verified=False,
        config_sha256_verified=False,
        plan_sha256_verified=False,
        model_preflight_sha256_verified=False,
        verification_sha256_verified=False,
        model_runtime_sha256_verified=False,
        resume_runtime_sha256_verified=False,
        fresh_run_verification_valid=False,
        receipt_binding_verified=False,
        adapter_digest_verified=False,
        corpus_examples_sha256_verified=False,
        corpus_manifest_sha256_verified=False,
        profile_binding_verified=False,
        repository_commit_binding_verified=False,
        violations=violations,
    )


def verify_cyber_sft_export(export_dir: str | Path) -> CyberSFTExportVerification:
    root = Path(export_dir).resolve()
    required_files = {
        "attestation": root / "run-attestation.json",
        "config": root / "selected-training-config.json",
        "plan": root / "qwen35-9b-smoke.plan.json",
        "model_preflight": root / "model-preflight.json",
        "verification": root / "verification.json",
        "profile": root / "selected-profile.txt",
        "repository_commit": root / "repository-commit.txt",
        "corpus_examples": root / "defense-reflex-v3.examples.jsonl",
        "corpus_manifest": root / "defense-reflex-v3.manifest.json",
        "adapter_manifest": root / "run" / "adapter-manifest.json",
        "receipt": root / "run" / "training-receipt.json",
        "model_runtime": root / "run" / "model-runtime.json",
        "resume_runtime": root / "run" / "resume-runtime.json",
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        return _invalid(
            violations=["missing export files: " + ", ".join(sorted(missing))]
        )

    try:
        attestation = CyberSFTRunAttestation.model_validate_json(
            required_files["attestation"].read_bytes()
        )
    except ValueError as exc:
        return _invalid(violations=[f"run attestation is invalid: {exc}"])

    violations: list[str] = []
    attestation_payload = attestation.model_dump(mode="json")
    attestation_verified = (
        _attestation_digest(attestation_payload) == attestation.attestation_sha256
    )
    if not attestation_verified:
        violations.append("run attestation self-hash does not verify")

    try:
        config = load_cyber_sft_config(required_files["config"])
        plan = CyberSFTPlan.model_validate_json(required_files["plan"].read_bytes())
        model_preflight = CyberModelAccessPreflight.model_validate_json(
            required_files["model_preflight"].read_bytes()
        )
        supplied_verification = CyberSFTArtifactVerification.model_validate_json(
            required_files["verification"].read_bytes()
        )
        manifest = CyberSFTAdapterManifest.model_validate_json(
            required_files["adapter_manifest"].read_bytes()
        )
        receipt = CyberSFTTrainingReceipt.model_validate_json(
            required_files["receipt"].read_bytes()
        )
        model_runtime = _json_object(
            required_files["model_runtime"],
            label="model-runtime.json",
        )
        resume_runtime = _json_object(
            required_files["resume_runtime"],
            label="resume-runtime.json",
        )
        corpus_manifest = _json_object(
            required_files["corpus_manifest"],
            label="defense-reflex-v3.manifest.json",
        )
    except (OSError, TypeError, ValueError) as exc:
        return _invalid(
            violations=[f"export artifact parsing failed: {exc}"],
            run_id=attestation.run_id,
        )

    config_verified = _config_sha256(config) == attestation.config_sha256
    if not config_verified:
        violations.append("selected training config SHA-256 differs from attestation")

    plan_verified = _sha256(required_files["plan"]) == attestation.plan_sha256
    if not plan_verified:
        violations.append("execution plan SHA-256 differs from attestation")

    model_preflight_verified = (
        _sha256(required_files["model_preflight"])
        == attestation.model_preflight_sha256
    )
    if not model_preflight_verified:
        violations.append("model preflight SHA-256 differs from attestation")

    verification_verified = (
        _sha256(required_files["verification"]) == attestation.verification_sha256
    )
    if not verification_verified:
        violations.append("verification report SHA-256 differs from attestation")

    model_runtime_verified = (
        _sha256(required_files["model_runtime"]) == attestation.model_runtime_sha256
    )
    if not model_runtime_verified:
        violations.append("model runtime SHA-256 differs from attestation")

    resume_runtime_verified = (
        _sha256(required_files["resume_runtime"])
        == attestation.resume_runtime_sha256
    )
    if not resume_runtime_verified:
        violations.append("resume runtime SHA-256 differs from attestation")

    fresh_verification = verify_cyber_sft_run("run", root=root)
    fresh_valid = fresh_verification.valid
    if not fresh_valid:
        violations.append("fresh verification of exported run is invalid")
    supplied_payload = supplied_verification.model_dump(mode="json")
    fresh_payload = fresh_verification.model_dump(mode="json")
    if supplied_payload != fresh_payload:
        violations.append(
            "exported verification report differs from fresh exported-run verification"
        )

    receipt_binding_verified = (
        receipt.run_id == attestation.run_id
        and receipt.base_model == attestation.base_model
        and receipt.base_revision == attestation.base_revision
        and receipt.receipt_sha256 == attestation.receipt_sha256
        and receipt.corpus_examples_sha256 == attestation.corpus_examples_sha256
        and receipt.corpus_manifest_sha256 == attestation.corpus_manifest_sha256
        and receipt.global_step == attestation.global_step
        and receipt.corpus_promotion_eligible == attestation.promotion_eligible
    )
    if not receipt_binding_verified:
        violations.append("training receipt bindings differ from attestation")

    adapter_verified = (
        manifest.adapter_digest == attestation.adapter_digest
        and receipt.adapter_digest == attestation.adapter_digest
    )
    if not adapter_verified:
        violations.append("adapter digest differs from attestation")

    examples_verified = (
        _sha256(required_files["corpus_examples"])
        == attestation.corpus_examples_sha256
    )
    if not examples_verified:
        violations.append("exported corpus examples SHA-256 differs from attestation")

    corpus_manifest_verified = (
        _sha256(required_files["corpus_manifest"])
        == attestation.corpus_manifest_sha256
        and corpus_manifest.get("examples_sha256")
        == attestation.corpus_examples_sha256
    )
    if not corpus_manifest_verified:
        violations.append("exported corpus manifest does not bind the attested examples")

    profile = required_files["profile"].read_text(encoding="utf-8").strip()
    profile_verified = profile == attestation.selected_profile
    if not profile_verified:
        violations.append("selected profile differs from attestation")

    repository_commit = (
        required_files["repository_commit"].read_text(encoding="utf-8").strip()
    )
    repository_verified = repository_commit == attestation.repository_commit
    if not repository_verified:
        violations.append("repository commit differs from attestation")

    plan_bindings = (
        plan.run_id == attestation.run_id
        and plan.base_model == attestation.base_model
        and plan.base_revision == attestation.base_revision
        and plan.corpus_examples_sha256 == attestation.corpus_examples_sha256
        and plan.corpus_manifest_sha256 == attestation.corpus_manifest_sha256
        and plan.executable_with_current_trainer
    )
    if not plan_bindings:
        violations.append("execution plan semantic bindings differ from attestation")

    preflight_bindings = (
        model_preflight.ready
        and model_preflight.public_ungated
        and model_preflight.base_model == attestation.base_model
        and model_preflight.requested_revision == attestation.base_revision
        and model_preflight.resolved_revision == attestation.resolved_model_revision
        and model_preflight.causal_lm_class == "Qwen3_5ForCausalLM"
        and model_preflight.model_type == "qwen3_5"
    )
    if not preflight_bindings:
        violations.append("model preflight semantic bindings differ from attestation")

    runtime_bindings = (
        model_runtime.get("loader") == "AutoModelForCausalLM"
        and model_runtime.get("model_class") == "Qwen3_5ForCausalLM"
        and model_runtime.get("expected_model_class") == "Qwen3_5ForCausalLM"
        and model_runtime.get("text_only") is True
        and resume_runtime.get("resumed") == attestation.resumed
        and resume_runtime.get("resume_checkpoint") == attestation.resume_checkpoint
    )
    if not runtime_bindings:
        violations.append("runtime semantic bindings differ from attestation")

    config_bindings = (
        config.run_id == attestation.run_id
        and config.base_model == attestation.base_model
        and config.base_revision == attestation.base_revision
    )
    if not config_bindings:
        violations.append("selected training config semantic bindings differ from attestation")

    valid = (
        attestation_verified
        and config_verified
        and plan_verified
        and model_preflight_verified
        and verification_verified
        and model_runtime_verified
        and resume_runtime_verified
        and fresh_valid
        and receipt_binding_verified
        and adapter_verified
        and examples_verified
        and corpus_manifest_verified
        and profile_verified
        and repository_verified
        and plan_bindings
        and preflight_bindings
        and runtime_bindings
        and config_bindings
        and not violations
    )
    return CyberSFTExportVerification(
        run_id=attestation.run_id,
        valid=valid,
        attestation_sha256_verified=attestation_verified,
        config_sha256_verified=config_verified,
        plan_sha256_verified=plan_verified,
        model_preflight_sha256_verified=model_preflight_verified,
        verification_sha256_verified=verification_verified,
        model_runtime_sha256_verified=model_runtime_verified,
        resume_runtime_sha256_verified=resume_runtime_verified,
        fresh_run_verification_valid=fresh_valid,
        receipt_binding_verified=receipt_binding_verified,
        adapter_digest_verified=adapter_verified,
        corpus_examples_sha256_verified=examples_verified,
        corpus_manifest_sha256_verified=corpus_manifest_verified,
        profile_binding_verified=profile_verified,
        repository_commit_binding_verified=repository_verified,
        violations=violations,
    )
