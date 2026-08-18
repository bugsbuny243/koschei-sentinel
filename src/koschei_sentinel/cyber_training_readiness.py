from __future__ import annotations

import importlib.util
import re
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    CyberSFTPlan,
    cyber_sft_messages,
    load_cyber_sft_examples,
    load_cyber_sft_validation_examples,
    plan_cyber_sft,
)
from koschei_sentinel.models import StrictModel

_MIN_TRANSFORMERS_VERSION = (5, 12, 0)


class CyberTrainingUseClass(StrEnum):
    SMOKE_ONLY = "SMOKE_ONLY"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CyberTrainingReadinessReport(StrictModel):
    schema_version: Literal["sentinel.cyber-training-readiness-report.v1"] = (
        "sentinel.cyber-training-readiness-report.v1"
    )
    run_id: str
    stage: str
    execution_profile: str
    use_class: CyberTrainingUseClass
    static_plan_ready: bool
    runtime_checked: bool
    runtime_dependencies_ready: bool | None
    transformers_version: str | None = None
    tokenization_checked: bool
    tokenization_ready: bool | None
    max_observed_sequence_tokens: int | None = Field(default=None, ge=0)
    overlength_example_ids: list[str]
    cuda_available: bool | None
    cuda_device_index: int | None = Field(default=None, ge=0)
    cuda_device_name: str | None = None
    cuda_compute_capability_major: int | None = Field(default=None, ge=0)
    cuda_compute_capability_minor: int | None = Field(default=None, ge=0)
    visible_cuda_memory_gb: float | None = Field(default=None, ge=0.0)
    minimum_cuda_memory_gb: float = Field(ge=0.0)
    ready_to_execute: bool
    blockers: list[str]
    warnings: list[str]
    plan: CyberSFTPlan | None = None


def _use_class(plan: CyberSFTPlan) -> CyberTrainingUseClass:
    if plan.corpus_promotion_eligible is True:
        return CyberTrainingUseClass.PROMOTION_ELIGIBLE
    if plan.corpus_promotion_eligible is False:
        return CyberTrainingUseClass.SMOKE_ONLY
    return CyberTrainingUseClass.NOT_APPLICABLE


def _runtime_dependency_names() -> list[str]:
    return [
        "torch",
        "datasets",
        "peft",
        "transformers",
        "huggingface_hub",
        "bitsandbytes",
        "accelerate",
    ]


def _release_version_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value.strip())
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def _is_prerelease_version(value: str) -> bool:
    normalized = value.strip().lower()
    return re.search(r"(?:^|[.\-+])(dev|a|b|rc)\d*", normalized) is not None or re.search(
        r"\d(?:dev|a|b|rc)\d*",
        normalized,
    ) is not None


def _transformers_version_blocker(version: str) -> str | None:
    parsed = _release_version_tuple(version)
    if parsed is None:
        return f"cannot parse installed transformers version: {version!r}"
    if _is_prerelease_version(version):
        return (
            "prerelease/nightly transformers builds are not accepted for Cyber SFT; "
            f"install a final transformers>=5.12,<6 release instead of {version}"
        )
    if parsed < _MIN_TRANSFORMERS_VERSION:
        return (
            "transformers>=5.12 is required for the fixed Qwen3.5 composite-to-text "
            f"dtype loading path; installed version is {version}"
        )
    return None


def _quantization_capability_blocker(
    config: CyberSFTConfig,
    capability: tuple[int, int],
) -> str | None:
    if config.quantization.bits == 4 and capability < (6, 0):
        return (
            "configured NF4/FP4 quantization requires NVIDIA compute capability 6.0+; "
            f"current device reports {capability[0]}.{capability[1]}"
        )
    if config.quantization.bits == 8 and capability < (7, 5):
        return (
            "configured load_in_8bit path requires NVIDIA compute capability 7.5+; "
            f"current device reports {capability[0]}.{capability[1]}"
        )
    return None


def _tokenization_rows(
    config: CyberSFTConfig,
    *,
    root: str | Path,
) -> list[object]:
    training_rows, _examples_sha, _manifest_sha, _promotion_eligible = (
        load_cyber_sft_examples(config, root=root)
    )
    rows = list(training_rows)
    validation_loaded = load_cyber_sft_validation_examples(config, root=root)
    if validation_loaded is not None:
        validation_rows, _validation_examples_sha, _validation_manifest_sha, _validation_promotion = (
            validation_loaded
        )
        rows.extend(validation_rows)
    return rows


def _tokenization_preflight(
    config: CyberSFTConfig,
    *,
    root: str | Path,
) -> tuple[bool, int, list[str]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenization preflight") from exc

    rows = _tokenization_rows(config, root=root)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
        use_fast=True,
    )
    maximum = 0
    overlength: list[str] = []
    for row in rows:
        messages = cyber_sft_messages(row)
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Cyber SFT example {row.example_id} has a chat-template prefix mismatch"
            )
        if len(prompt_ids) >= len(full_ids):
            raise ValueError(
                f"Cyber SFT example {row.example_id} has no supervised answer tokens"
            )
        maximum = max(maximum, len(full_ids))
        if len(full_ids) > config.max_sequence_length:
            overlength.append(row.example_id)
    return not overlength, maximum, sorted(overlength)


def audit_cyber_training_readiness(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
    check_runtime: bool = False,
    check_tokenization: bool = False,
) -> CyberTrainingReadinessReport:
    blockers: list[str] = []
    warnings: list[str] = []
    plan: CyberSFTPlan | None = None
    try:
        plan = plan_cyber_sft(config, root=root)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        blockers.append(str(exc))

    if plan is not None:
        warnings.extend(plan.warnings)
        if not plan.executable_with_current_trainer:
            blockers.append(
                "configured execution profile is not supported by the current Cyber SFT trainer"
            )

    dependencies_ready: bool | None = None
    transformers_version: str | None = None
    cuda_available: bool | None = None
    cuda_device_index: int | None = None
    cuda_device_name: str | None = None
    cuda_capability: tuple[int, int] | None = None
    cuda_memory: float | None = None
    if check_runtime:
        missing = [
            name for name in _runtime_dependency_names() if importlib.util.find_spec(name) is None
        ]
        dependencies_ready = not missing
        if missing:
            blockers.append("missing training runtime packages: " + ", ".join(missing))
        if dependencies_ready:
            try:
                transformers_version = metadata.version("transformers")
            except metadata.PackageNotFoundError:
                blockers.append("installed transformers package metadata is unavailable")
                dependencies_ready = False
            else:
                version_blocker = _transformers_version_blocker(transformers_version)
                if version_blocker is not None:
                    blockers.append(version_blocker)
                    dependencies_ready = False
        if dependencies_ready:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if not cuda_available:
                blockers.append("CUDA is not available")
            else:
                cuda_device_index = int(torch.cuda.current_device())
                properties = torch.cuda.get_device_properties(cuda_device_index)
                cuda_device_name = str(properties.name)
                cuda_memory = properties.total_memory / (1024**3)
                capability = torch.cuda.get_device_capability(cuda_device_index)
                cuda_capability = (int(capability[0]), int(capability[1]))
                capability_blocker = _quantization_capability_blocker(
                    config,
                    cuda_capability,
                )
                if capability_blocker is not None:
                    blockers.append(capability_blocker)
                if cuda_memory + 1e-9 < config.minimum_cuda_memory_gb:
                    blockers.append(
                        "current CUDA device memory below configured minimum: "
                        f"{cuda_memory:.1f} GiB < {config.minimum_cuda_memory_gb:.1f} GiB"
                    )
                if (
                    config.quantization.compute_dtype == "bfloat16"
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and not torch.cuda.is_bf16_supported()
                ):
                    blockers.append(
                        "configured bfloat16 is not supported by current CUDA hardware"
                    )
    else:
        warnings.append(
            "runtime packages and CUDA were not checked; execution readiness remains false"
        )

    tokenization_ready: bool | None = None
    maximum_tokens: int | None = None
    overlength: list[str] = []
    if check_tokenization:
        if importlib.util.find_spec("transformers") is None:
            tokenization_ready = False
            blockers.append("transformers is missing; tokenization preflight cannot run")
        elif plan is not None:
            try:
                tokenization_ready, maximum_tokens, overlength = _tokenization_preflight(
                    config,
                    root=root,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                tokenization_ready = False
                blockers.append(f"tokenization preflight failed: {exc}")
            if overlength:
                blockers.append(
                    f"{len(overlength)} Cyber SFT examples exceed "
                    f"max_sequence_length={config.max_sequence_length}"
                )
    else:
        warnings.append(
            "tokenization was not checked against the pinned base model; execution readiness remains false"
        )

    static_ready = plan is not None
    ready = (
        static_ready
        and check_runtime
        and dependencies_ready is True
        and cuda_available is True
        and check_tokenization
        and tokenization_ready is True
        and not blockers
    )
    return CyberTrainingReadinessReport(
        run_id=config.run_id,
        stage=config.stage.value,
        execution_profile=config.execution_profile.value,
        use_class=(
            _use_class(plan)
            if plan is not None
            else CyberTrainingUseClass.NOT_APPLICABLE
        ),
        static_plan_ready=static_ready,
        runtime_checked=check_runtime,
        runtime_dependencies_ready=dependencies_ready,
        transformers_version=transformers_version,
        tokenization_checked=check_tokenization,
        tokenization_ready=tokenization_ready,
        max_observed_sequence_tokens=maximum_tokens,
        overlength_example_ids=overlength,
        cuda_available=cuda_available,
        cuda_device_index=cuda_device_index,
        cuda_device_name=cuda_device_name,
        cuda_compute_capability_major=(
            cuda_capability[0] if cuda_capability is not None else None
        ),
        cuda_compute_capability_minor=(
            cuda_capability[1] if cuda_capability is not None else None
        ),
        visible_cuda_memory_gb=cuda_memory,
        minimum_cuda_memory_gb=config.minimum_cuda_memory_gb,
        ready_to_execute=ready,
        blockers=blockers,
        warnings=warnings,
        plan=plan,
    )
