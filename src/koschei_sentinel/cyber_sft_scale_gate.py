from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.cyber_sft_run_attestation import CyberSFTRunAttestation
from koschei_sentinel.cyber_sft_trainer import CyberSFTTrainingReceipt
from koschei_sentinel.cyber_sft_training import CyberSFTConfig, load_cyber_sft_config
from koschei_sentinel.models import StrictModel

_MICRO_MODEL = "Qwen/Qwen3.5-0.8B-Base"
_MICRO_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
_TARGET_MODEL = "Qwen/Qwen3.5-9B-Base"
_TARGET_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"


class CyberSFTScaleDisposition(StrEnum):
    PROCEED_9B = "PROCEED_9B"
    PROCEED_9B_CAUTION = "PROCEED_9B_CAUTION"
    BLOCK_9B = "BLOCK_9B"


class CyberSFTScaleGate(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-scale-gate.v1"] = (
        "sentinel.cyber-sft-scale-gate.v1"
    )
    micro_run_id: str | None
    micro_export_valid: bool
    micro_global_step: int | None = Field(default=None, ge=0)
    micro_gpu_name: str | None
    micro_gpu_total_memory_gb: float | None = Field(default=None, ge=0.0)
    micro_peak_allocated_gb: float | None = Field(default=None, ge=0.0)
    micro_peak_reserved_gb: float | None = Field(default=None, ge=0.0)
    micro_max_sequence_length: int | None = Field(default=None, ge=1)
    target_max_sequence_length: int = Field(ge=1)
    quantization_bits_match: bool
    compute_dtype_match: bool
    lora_target_coverage: bool
    missing_target_suffixes: list[str]
    target_model: str
    target_revision: str
    target_minimum_cuda_memory_gb: float = Field(ge=0.0)
    fast_kernel_warnings: list[str]
    blockers: list[str]
    warnings: list[str]
    allowed_to_attempt_9b: bool
    disposition: CyberSFTScaleDisposition


def _runtime_warnings(run_dir: Path) -> list[str]:
    path = run_dir / "runtime-warnings.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("micro runtime-warnings.json is invalid JSON") from exc
    warnings = payload.get("warnings") if isinstance(payload, dict) else None
    if not isinstance(warnings, list) or any(
        not isinstance(row, str) for row in warnings
    ):
        raise ValueError("micro runtime-warnings.json warnings must be a string list")
    return warnings


def _recipe_checks(
    micro: CyberSFTConfig | None,
    target: CyberSFTConfig,
    *,
    blockers: list[str],
    warnings: list[str],
) -> tuple[bool, bool, bool, list[str]]:
    if micro is None:
        blockers.append("micro training-config.json is missing or invalid")
        return False, False, False, sorted(set(target.lora.target_suffixes))

    bits_match = micro.quantization.bits == target.quantization.bits
    dtype_match = (
        micro.quantization.compute_dtype == target.quantization.compute_dtype
    )
    target_suffixes = set(target.lora.target_suffixes)
    micro_suffixes = set(micro.lora.target_suffixes)
    missing_suffixes = sorted(target_suffixes - micro_suffixes)
    target_coverage = not missing_suffixes

    if not bits_match:
        blockers.append(
            "micro quantization bits differ from the 9B training recipe"
        )
    if not dtype_match:
        blockers.append(
            "micro compute dtype differs from the 9B training recipe"
        )
    if not target_coverage:
        blockers.append(
            "micro LoRA target coverage does not exercise all 9B target classes: "
            + ", ".join(missing_suffixes)
        )
    if micro.max_sequence_length < target.max_sequence_length:
        warnings.append(
            "micro context length is shorter than the 9B target context; "
            "the full target sequence length was not proven by micro-smoke"
        )

    return bits_match, dtype_match, target_coverage, missing_suffixes


def evaluate_9b_scale_gate(
    *,
    micro_export_dir: str | Path,
    target_config_path: str | Path,
) -> CyberSFTScaleGate:
    root = Path(micro_export_dir).resolve()
    blockers: list[str] = []
    warnings: list[str] = []

    export_report = verify_cyber_sft_export(root)
    if not export_report.valid:
        blockers.append("micro Cyber SFT portable export is not valid")

    target = load_cyber_sft_config(target_config_path)
    if target.base_model != _TARGET_MODEL or target.base_revision != _TARGET_REVISION:
        blockers.append("target config is not the pinned Qwen3.5-9B-Base smoke target")

    micro_config: CyberSFTConfig | None = None
    micro_config_path = root / "training-config.json"
    try:
        if micro_config_path.is_file():
            micro_config = load_cyber_sft_config(micro_config_path)
    except (OSError, TypeError, ValueError) as exc:
        blockers.append(f"micro training config cannot be parsed: {exc}")

    bits_match, dtype_match, target_coverage, missing_suffixes = _recipe_checks(
        micro_config,
        target,
        blockers=blockers,
        warnings=warnings,
    )

    if micro_config is not None:
        if (
            micro_config.base_model != _MICRO_MODEL
            or micro_config.base_revision != _MICRO_REVISION
        ):
            blockers.append(
                "micro training config does not bind the pinned Qwen3.5-0.8B-Base"
            )
        if micro_config.stage != target.stage:
            blockers.append("micro and 9B training stages differ")
        if micro_config.execution_profile != target.execution_profile:
            blockers.append("micro and 9B execution profiles differ")

    attestation: CyberSFTRunAttestation | None = None
    receipt: CyberSFTTrainingReceipt | None = None
    attestation_path = root / "run-attestation.json"
    receipt_path = root / "run" / "training-receipt.json"
    try:
        if attestation_path.is_file():
            attestation = CyberSFTRunAttestation.model_validate_json(
                attestation_path.read_bytes()
            )
        if receipt_path.is_file():
            receipt = CyberSFTTrainingReceipt.model_validate_json(
                receipt_path.read_bytes()
            )
    except ValueError as exc:
        blockers.append(f"micro run evidence cannot be parsed: {exc}")

    if attestation is None or receipt is None:
        blockers.append("micro attestation or training receipt is missing")
    else:
        if attestation.selected_profile != "micro":
            blockers.append("scale gate requires a micro-profile attestation")
        if (
            attestation.base_model != _MICRO_MODEL
            or attestation.base_revision != _MICRO_REVISION
        ):
            blockers.append(
                "micro attestation does not bind the pinned Qwen3.5-0.8B-Base"
            )
        if receipt.global_step <= 0:
            blockers.append("micro run completed zero optimizer steps")
        if receipt.cuda_total_memory_gb + 1e-9 < target.minimum_cuda_memory_gb:
            blockers.append(
                "micro GPU memory is below the 9B config minimum: "
                f"{receipt.cuda_total_memory_gb:.1f} GiB < "
                f"{target.minimum_cuda_memory_gb:.1f} GiB"
            )

    fast_warnings: list[str] = []
    if (root / "run").is_dir():
        fast_warnings = _runtime_warnings(root / "run")
    if fast_warnings:
        warnings.append(
            "Qwen3.5 fast DeltaNet kernels are incomplete; 9B may use slower, "
            "more memory-hungry fallback ops"
        )

    allowed = not blockers
    if not allowed:
        disposition = CyberSFTScaleDisposition.BLOCK_9B
    elif warnings:
        disposition = CyberSFTScaleDisposition.PROCEED_9B_CAUTION
    else:
        disposition = CyberSFTScaleDisposition.PROCEED_9B

    return CyberSFTScaleGate(
        micro_run_id=attestation.run_id if attestation is not None else None,
        micro_export_valid=export_report.valid,
        micro_global_step=receipt.global_step if receipt is not None else None,
        micro_gpu_name=receipt.cuda_device_name if receipt is not None else None,
        micro_gpu_total_memory_gb=(
            receipt.cuda_total_memory_gb if receipt is not None else None
        ),
        micro_peak_allocated_gb=(
            receipt.max_cuda_memory_allocated_gb if receipt is not None else None
        ),
        micro_peak_reserved_gb=(
            receipt.max_cuda_memory_reserved_gb if receipt is not None else None
        ),
        micro_max_sequence_length=(
            micro_config.max_sequence_length if micro_config is not None else None
        ),
        target_max_sequence_length=target.max_sequence_length,
        quantization_bits_match=bits_match,
        compute_dtype_match=dtype_match,
        lora_target_coverage=target_coverage,
        missing_target_suffixes=missing_suffixes,
        target_model=target.base_model,
        target_revision=target.base_revision,
        target_minimum_cuda_memory_gb=target.minimum_cuda_memory_gb,
        fast_kernel_warnings=fast_warnings,
        blockers=blockers,
        warnings=warnings,
        allowed_to_attempt_9b=allowed,
        disposition=disposition,
    )
