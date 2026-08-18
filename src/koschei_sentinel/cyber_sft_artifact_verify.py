from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_trainer import (
    CyberSFTAdapterManifest,
    CyberSFTTrainingReceipt,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json, resolve_under_root

_EXPECTED_TEXT_RUNTIME = {
    "loader": "AutoModelForCausalLM",
    "model_class": "Qwen3_5ForCausalLM",
    "expected_model_class": "Qwen3_5ForCausalLM",
    "text_only": True,
}
_SUPPORTED_LORA_TARGET_TYPES = {
    "torch.nn.modules.linear.Linear",
    "bitsandbytes.nn.modules.Linear4bit",
    "bitsandbytes.nn.modules.Linear8bitLt",
}


class CyberSFTArtifactVerification(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-artifact-verification.v1"] = (
        "sentinel.cyber-sft-artifact-verification.v1"
    )
    run_id: str | None
    valid: bool
    smoke_only: bool | None
    adapter_digest_verified: bool
    receipt_digest_verified: bool
    receipt_bindings_verified: bool
    model_runtime_verified: bool
    global_step: int | None = Field(default=None, ge=0)
    violations: list[str]


def _adapter_digest(run_path: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        candidate = (run_path / relative).resolve()
        if candidate != run_path and run_path not in candidate.parents:
            raise ValueError(f"adapter file escapes run directory: {relative}")
        if not candidate.is_file():
            raise ValueError(f"adapter file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _receipt_digest(receipt: CyberSFTTrainingReceipt) -> str:
    payload = receipt.model_dump(mode="json")
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _verify_model_runtime(
    path: Path,
    *,
    require_lora_target_types: bool,
) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "model-runtime.json is missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"model-runtime.json is invalid: {exc}"
    if not isinstance(payload, dict):
        return False, "model-runtime.json must contain a JSON object"
    for key, expected in _EXPECTED_TEXT_RUNTIME.items():
        if payload.get(key) != expected:
            return False, f"model runtime mismatch: {key}"

    requested = payload.get("requested_compute_dtype")
    if requested not in {"float16", "bfloat16"}:
        return False, "model runtime lacks a supported requested compute dtype"
    observed = payload.get("observed_floating_dtypes_before_kbit_prepare")
    if not isinstance(observed, list) or not observed:
        return False, "model runtime lacks pre-kbit floating dtype evidence"
    if any(not isinstance(row, str) for row in observed):
        return False, "model runtime floating dtype evidence must contain strings"
    if requested not in observed:
        return False, "requested compute dtype was not observed before k-bit preparation"
    competing = "bfloat16" if requested == "float16" else "float16"
    if competing in observed:
        return False, "competing low-precision dtype leaked into loaded model weights"

    target_types = payload.get("lora_target_module_types_before_peft")
    if require_lora_target_types:
        if not isinstance(target_types, dict) or not target_types:
            return False, "model runtime lacks pre-PEFT LoRA target module type evidence"
        for module_type, count in target_types.items():
            if not isinstance(module_type, str) or not isinstance(count, int) or count <= 0:
                return False, "model runtime LoRA target module type evidence is malformed"
            if module_type not in _SUPPORTED_LORA_TARGET_TYPES:
                return False, f"unsupported LoRA target module type in runtime proof: {module_type}"
    elif target_types is not None:
        if not isinstance(target_types, dict):
            return False, "model runtime LoRA target module type evidence must be an object"
        for module_type, count in target_types.items():
            if not isinstance(module_type, str) or not isinstance(count, int) or count <= 0:
                return False, "model runtime LoRA target module type evidence is malformed"
            if module_type not in _SUPPORTED_LORA_TARGET_TYPES:
                return False, f"unsupported LoRA target module type in runtime proof: {module_type}"
    return True, None


def _invalid_report(
    *,
    violations: list[str],
    run_id: str | None = None,
    smoke_only: bool | None = None,
) -> CyberSFTArtifactVerification:
    return CyberSFTArtifactVerification(
        run_id=run_id,
        valid=False,
        smoke_only=smoke_only,
        adapter_digest_verified=False,
        receipt_digest_verified=False,
        receipt_bindings_verified=False,
        model_runtime_verified=False,
        global_step=None,
        violations=violations,
    )


def verify_cyber_sft_run(
    run_dir: str,
    *,
    root: str | Path = ".",
) -> CyberSFTArtifactVerification:
    root_path = Path(root).resolve()
    run_path = resolve_under_root(root_path, run_dir)
    violations: list[str] = []
    manifest_path = run_path / "adapter-manifest.json"
    receipt_path = run_path / "training-receipt.json"
    runtime_path = run_path / "model-runtime.json"
    if not manifest_path.is_file():
        violations.append("adapter-manifest.json is missing")
    if not receipt_path.is_file():
        violations.append("training-receipt.json is missing")
    if violations:
        return _invalid_report(violations=violations)

    try:
        manifest = CyberSFTAdapterManifest.model_validate_json(manifest_path.read_bytes())
    except ValueError as exc:
        violations.append(f"adapter manifest is invalid: {exc}")
        return _invalid_report(violations=violations)
    try:
        receipt = CyberSFTTrainingReceipt.model_validate_json(receipt_path.read_bytes())
    except ValueError as exc:
        violations.append(f"training receipt is invalid: {exc}")
        return _invalid_report(
            violations=violations,
            run_id=manifest.run_id,
            smoke_only=(manifest.corpus_promotion_eligible is False),
        )

    adapter_verified = False
    try:
        observed_adapter_digest = _adapter_digest(run_path, manifest.adapter_files)
        adapter_verified = observed_adapter_digest == manifest.adapter_digest
        if not adapter_verified:
            violations.append("adapter directory digest differs from adapter manifest")
    except ValueError as exc:
        violations.append(str(exc))

    receipt_verified = _receipt_digest(receipt) == receipt.receipt_sha256
    if not receipt_verified:
        violations.append("training receipt self-hash does not verify")

    binding_pairs = [
        ("run_id", receipt.run_id, manifest.run_id),
        ("stage", receipt.stage, manifest.stage),
        ("base_model", receipt.base_model, manifest.base_model),
        ("base_revision", receipt.base_revision, manifest.base_revision),
        (
            "corpus_examples_sha256",
            receipt.corpus_examples_sha256,
            manifest.corpus_examples_sha256,
        ),
        (
            "corpus_manifest_sha256",
            receipt.corpus_manifest_sha256,
            manifest.corpus_manifest_sha256,
        ),
        (
            "corpus_promotion_eligible",
            receipt.corpus_promotion_eligible,
            manifest.corpus_promotion_eligible,
        ),
        ("adapter_digest", receipt.adapter_digest, manifest.adapter_digest),
        ("optimizer", receipt.optimizer, manifest.optimizer),
    ]
    bindings_verified = True
    for field, receipt_value, manifest_value in binding_pairs:
        if receipt_value != manifest_value:
            bindings_verified = False
            violations.append(f"training receipt binding mismatch: {field}")

    runtime_verified, runtime_violation = _verify_model_runtime(
        runtime_path,
        require_lora_target_types=manifest.input_adapter_dir is None,
    )
    if runtime_violation is not None:
        violations.append(runtime_violation)

    if receipt.global_step <= 0:
        violations.append("training receipt reports zero optimizer steps")

    valid = (
        adapter_verified
        and receipt_verified
        and bindings_verified
        and runtime_verified
        and receipt.global_step > 0
        and not violations
    )
    return CyberSFTArtifactVerification(
        run_id=manifest.run_id,
        valid=valid,
        smoke_only=(manifest.corpus_promotion_eligible is False),
        adapter_digest_verified=adapter_verified,
        receipt_digest_verified=receipt_verified,
        receipt_bindings_verified=bindings_verified,
        model_runtime_verified=runtime_verified,
        global_step=receipt.global_step,
        violations=violations,
    )
