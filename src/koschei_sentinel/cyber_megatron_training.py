from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_sft_training import (
    CyberExecutionProfile,
    CyberSFTConfig,
    CyberSFTStage,
    cyber_sft_messages,
    load_cyber_sft_examples,
    load_cyber_sft_validation_examples,
    split_cyber_sft_examples,
)
from koschei_sentinel.defense_reflex_gold_release_audit import (
    audit_gold_defense_release,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json, resolve_under_root

QWEN35_397B_MODEL = "Qwen/Qwen3.5-397B-A17B"
QWEN35_397B_REVISION = "8472618112abcbd45acbcdc58436aff4233c23f7"
MS_SWIFT_VERSION = "4.5.2"
QWEN35_EXPERT_COUNT = 512
LAUNCH_APPROVAL_ENV = "KOSCHEI_397B_LAUNCH_APPROVED"
LAUNCH_SESSION_ENV = "KOSCHEI_397B_LAUNCH_SESSION"
RUN_IDENTITY_FILENAME = "koschei-run-identity.json"
LAUNCH_STATE_FILENAME = "koschei-launch-state.json"


def _relative_path(value: str, field_name: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{field_name} must stay inside the repository root")


class CyberMegatronTopology(StrictModel):
    nodes: int = Field(ge=1, le=128)
    gpus_per_node: int = Field(ge=1, le=16)
    minimum_gpu_memory_gib: int = Field(ge=40, le=256)
    tensor_model_parallel_size: int = Field(ge=1, le=64)
    pipeline_model_parallel_size: int = Field(ge=1, le=60)
    context_parallel_size: int = Field(default=1, ge=1, le=64)
    expert_model_parallel_size: int = Field(ge=1, le=512)
    sequence_parallel: Literal[True] = True

    @model_validator(mode="after")
    def topology_is_coherent(self) -> CyberMegatronTopology:
        model_parallel = (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
        )
        if self.world_size % model_parallel:
            raise ValueError("world size must be divisible by TP * PP * CP")
        if self.data_parallel_size % self.expert_model_parallel_size:
            raise ValueError(
                "data parallel size must be divisible by expert parallel size"
            )
        if QWEN35_EXPERT_COUNT % self.expert_model_parallel_size:
            raise ValueError("Qwen3.5 expert count must be divisible by expert parallel size")
        return self

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def data_parallel_size(self) -> int:
        denominator = (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
        )
        return self.world_size // denominator


class CyberMegatronLora(StrictModel):
    tuner_type: Literal["lora"] = "lora"
    rank: int = Field(default=64, ge=1, le=256)
    alpha: int = Field(default=128, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: Literal["all-linear"] = "all-linear"
    language_model_only: Literal[True] = True
    freeze_vit: Literal[True] = True
    freeze_aligner: Literal[True] = True


class CyberMegatronCheckpointing(StrictModel):
    save_steps: int = Field(default=25, ge=1, le=100_000)
    eval_steps: int = Field(default=25, ge=1, le=100_000)
    save_total_limit: int = Field(default=3, ge=2, le=100)
    save_optimizer: Literal[True] = True
    save_rng: Literal[True] = True
    save_safetensors: Literal[True] = True
    merge_lora: Literal[True] = True


class CyberMegatronOptimization(StrictModel):
    max_length: int = Field(default=8192, ge=512, le=262_144)
    epochs: float = Field(default=1.0, gt=0.0, le=20.0)
    learning_rate: float = Field(default=0.0001, gt=0.0, le=0.01)
    minimum_learning_rate: float = Field(default=0.00001, ge=0.0, le=0.01)
    warmup_fraction: float = Field(default=0.03, ge=0.0, le=0.5)
    micro_batch_size: int = Field(default=1, ge=1, le=32)
    global_batch_size: int = Field(default=32, ge=1, le=65_536)
    moe_aux_loss_coeff: float = Field(default=0.000001, gt=0.0, le=1.0)
    recompute_granularity: Literal["full", "selective"] = "full"
    recompute_method: Literal["uniform", "block"] = "uniform"
    recompute_num_layers: int = Field(default=1, ge=1, le=60)
    packing: Literal[True] = True
    attention_backend: Literal["flash"] = "flash"
    dataset_num_proc: int = Field(default=16, ge=1, le=256)
    dataloader_num_workers: int = Field(default=4, ge=0, le=128)
    logging_steps: int = Field(default=5, ge=1, le=100_000)

    @model_validator(mode="after")
    def learning_rates_are_coherent(self) -> CyberMegatronOptimization:
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        return self


class CyberMegatronSFTConfig(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-sft-config.v1"] = (
        "sentinel.cyber-megatron-sft-config.v1"
    )
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    model: str = QWEN35_397B_MODEL
    model_revision: str = QWEN35_397B_REVISION
    backend: Literal["megatron-swift"] = "megatron-swift"
    ms_swift_version: str = MS_SWIFT_VERSION
    stage: CyberSFTStage = CyberSFTStage.DEFENSE_REFLEX
    corpus_dir: str = Field(min_length=1, max_length=1024)
    validation_corpus_dir: str | None = Field(default=None, min_length=1, max_length=1024)
    validation_ratio: float = Field(default=0.10, ge=0.0, le=0.5)
    dataset_dir: str = Field(min_length=1, max_length=1024)
    output_dir: str = Field(min_length=1, max_length=1024)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    topology: CyberMegatronTopology
    lora: CyberMegatronLora = Field(default_factory=CyberMegatronLora)
    optimization: CyberMegatronOptimization = Field(
        default_factory=CyberMegatronOptimization
    )
    checkpointing: CyberMegatronCheckpointing = Field(
        default_factory=CyberMegatronCheckpointing
    )
    launch_approval_env: Literal["KOSCHEI_397B_LAUNCH_APPROVED"] = LAUNCH_APPROVAL_ENV
    launch_session_env: Literal["KOSCHEI_397B_LAUNCH_SESSION"] = LAUNCH_SESSION_ENV

    @model_validator(mode="after")
    def config_is_single_model_and_safe(self) -> CyberMegatronSFTConfig:
        if self.model != QWEN35_397B_MODEL:
            raise ValueError(f"only {QWEN35_397B_MODEL} is an active training target")
        if self.model_revision != QWEN35_397B_REVISION:
            raise ValueError(
                "Qwen3.5-397B-A17B model revision must match the pinned weight revision"
            )
        if self.ms_swift_version != MS_SWIFT_VERSION:
            raise ValueError(f"ms-swift must be pinned to {MS_SWIFT_VERSION}")
        for field_name, value in (
            ("corpus_dir", self.corpus_dir),
            ("validation_corpus_dir", self.validation_corpus_dir),
            ("dataset_dir", self.dataset_dir),
            ("output_dir", self.output_dir),
        ):
            if value is not None:
                _relative_path(value, field_name)
        if self.validation_corpus_dir is not None:
            if self.validation_corpus_dir == self.corpus_dir:
                raise ValueError("validation_corpus_dir must differ from corpus_dir")
            if self.validation_ratio != 0.0:
                raise ValueError(
                    "explicit validation_corpus_dir requires validation_ratio=0.0"
                )
        if self.dataset_dir == self.output_dir:
            raise ValueError("dataset_dir must differ from output_dir")
        effective_micro_batch = (
            self.optimization.micro_batch_size * self.topology.data_parallel_size
        )
        if self.optimization.global_batch_size % effective_micro_batch:
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size * data parallel size"
            )
        return self

    @property
    def gradient_accumulation_steps(self) -> int:
        denominator = self.optimization.micro_batch_size * self.topology.data_parallel_size
        return self.optimization.global_batch_size // denominator


class CyberMegatronDatasetManifest(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-dataset.v1"] = (
        "sentinel.cyber-megatron-dataset.v1"
    )
    run_id: str
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    stage: CyberSFTStage
    seed: int
    source_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_source_examples_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    validation_source_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    source_promotion_eligible: bool | None = None
    validation_source_promotion_eligible: bool | None = None
    gold_release_audit_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    training_examples: int = Field(gt=0)
    validation_examples: int = Field(ge=0)
    max_observed_sequence_tokens: int = Field(ge=1)
    train_jsonl_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_jsonl_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CyberMegatronDatasetVerification(StrictModel):
    valid: bool
    manifest: CyberMegatronDatasetManifest | None
    manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    blockers: list[str]


class CyberMegatronResume(StrictModel):
    mcore_model: str = Field(min_length=1, max_length=4096)
    mcore_adapter: str = Field(min_length=1, max_length=4096)
    binding_manifest: str = Field(min_length=1, max_length=4096)


class CyberMegatronRunIdentity(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-run-identity.v1"] = (
        "sentinel.cyber-megatron-run-identity.v1"
    )
    run_id: str
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_source_examples_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    validation_source_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    train_jsonl_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_jsonl_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_dir: str
    identity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CyberMegatronLaunchState(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-launch-state.v1"] = (
        "sentinel.cyber-megatron-launch-state.v1"
    )
    run_identity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    launch_session_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    state: Literal["launching", "completed", "failed"]
    nodes: int = Field(gt=0)


class CyberMegatronPlan(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-plan.v1"] = (
        "sentinel.cyber-megatron-plan.v1"
    )
    run_id: str
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    backend: Literal["megatron-swift"]
    ms_swift_version: Literal["4.5.2"]
    world_size: int = Field(gt=0)
    minimum_gpu_memory_gib: int = Field(gt=0)
    data_parallel_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    dataset_verified: bool
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    source_examples_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    source_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    validation_source_examples_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    validation_source_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    train_jsonl_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    validation_jsonl_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    gold_release_audit_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    max_observed_sequence_tokens: int | None = Field(default=None, ge=1)
    run_identity_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    output_available: bool
    static_ready: bool
    requires_explicit_launch_approval: Literal[True] = True
    launch_approval_env: Literal["KOSCHEI_397B_LAUNCH_APPROVED"]
    launch_session_env: Literal["KOSCHEI_397B_LAUNCH_SESSION"]
    command: list[str] = Field(min_length=3)
    blockers: list[str]
    warnings: list[str]


def load_cyber_megatron_config(path: str | Path) -> CyberMegatronSFTConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Cyber Megatron SFT config is not valid JSON") from exc
    return CyberMegatronSFTConfig.model_validate(payload)


def _source_config(config: CyberMegatronSFTConfig) -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id=config.run_id,
        stage=config.stage,
        execution_profile=CyberExecutionProfile.MOE_DISTRIBUTED_REQUIRED,
        base_model=config.model,
        base_revision=config.model_revision,
        corpus_dir=config.corpus_dir,
        validation_corpus_dir=config.validation_corpus_dir,
        output_dir=config.output_dir,
        max_sequence_length=min(config.optimization.max_length, 32_768),
        epochs=config.optimization.epochs,
        learning_rate=config.optimization.learning_rate,
        per_device_batch_size=config.optimization.micro_batch_size,
        gradient_accumulation_steps=1,
        validation_ratio=config.validation_ratio,
        warmup_ratio=config.optimization.warmup_fraction,
        logging_steps=config.optimization.logging_steps,
        seed=config.seed,
        gradient_checkpointing=False,
        enable_router_aux_loss=True,
    )


def _assert_disjoint(training_rows: list[StrictModel], validation_rows: list[StrictModel]) -> None:
    for attribute in ("example_id", "scenario_id"):
        training = {str(getattr(row, attribute)) for row in training_rows}
        validation = {str(getattr(row, attribute)) for row in validation_rows}
        overlap = sorted(training & validation)
        if overlap:
            raise ValueError(
                f"explicit Cyber SFT TRAIN/VALIDATION {attribute} values overlap: "
                + ", ".join(overlap[:8])
            )


def _jsonl(rows: list[StrictModel]) -> bytes:
    lines = [canonical_json({"messages": cyber_sft_messages(row)}) for row in rows]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _sha256_canonical(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _config_sha256(config: CyberMegatronSFTConfig) -> str:
    return _sha256_canonical(config.model_dump(mode="json"))


def _combined_promotion_eligibility(
    training: bool | None,
    validation: bool | None,
) -> bool | None:
    if training is False or validation is False:
        return False
    if training is True and validation is True:
        return True
    return None


def _gold_release_root(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path,
) -> Path | None:
    if config.stage is not CyberSFTStage.DEFENSE_REFLEX:
        return None
    if config.validation_corpus_dir is None:
        return None
    root_path = Path(root).resolve()
    train_dir = resolve_under_root(root_path, config.corpus_dir)
    validation_dir = resolve_under_root(root_path, config.validation_corpus_dir)
    if train_dir.name != "train" or validation_dir.name != "validation":
        return None
    if train_dir.parent != validation_dir.parent:
        return None
    return train_dir.parent


def _load_pinned_tokenizer(config: CyberMegatronSFTConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to preflight the pinned 397B tokenizer"
        ) from exc
    return AutoTokenizer.from_pretrained(
        config.model,
        revision=config.model_revision,
        trust_remote_code=False,
        use_fast=True,
    )


def _tokenization_preflight(
    config: CyberMegatronSFTConfig,
    rows: list[StrictModel],
) -> int:
    tokenizer = _load_pinned_tokenizer(config)
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
        example_id = str(getattr(row, "example_id", "unknown"))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Cyber Megatron example {example_id} has a chat-template prefix mismatch"
            )
        if len(prompt_ids) >= len(full_ids):
            raise ValueError(
                f"Cyber Megatron example {example_id} has no supervised answer tokens"
            )
        maximum = max(maximum, len(full_ids))
        if len(full_ids) > config.optimization.max_length:
            overlength.append(example_id)
    if overlength:
        raise ValueError(
            "Cyber Megatron examples exceed optimization.max_length: "
            + ", ".join(sorted(overlength)[:8])
        )
    if maximum == 0:
        raise ValueError("Cyber Megatron tokenization preflight received no examples")
    return maximum


def _expected_dataset(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path,
) -> tuple[bytes, bytes, CyberMegatronDatasetManifest]:
    source = _source_config(config)
    rows, examples_sha, manifest_sha, source_promotion = load_cyber_sft_examples(
        source, root=root
    )
    explicit = load_cyber_sft_validation_examples(source, root=root)
    validation_examples_sha: str | None = None
    validation_manifest_sha: str | None = None
    validation_promotion: bool | None = None
    if explicit is None:
        training_rows, validation_rows = split_cyber_sft_examples(
            rows,
            validation_ratio=config.validation_ratio,
            seed=config.seed,
        )
        validation_promotion = source_promotion
    else:
        (
            validation_rows,
            validation_examples_sha,
            validation_manifest_sha,
            validation_promotion,
        ) = explicit
        training_rows = rows
        _assert_disjoint(training_rows, validation_rows)
    if not training_rows:
        raise ValueError("Cyber Megatron dataset contains no training examples")

    gold_audit_sha: str | None = None
    promotion_eligible = _combined_promotion_eligibility(
        source_promotion,
        validation_promotion,
    )
    if promotion_eligible is True and config.stage is CyberSFTStage.DEFENSE_REFLEX:
        release_root = _gold_release_root(config, root=root)
        if release_root is None:
            raise ValueError(
                "promotion-eligible Defense Reflex training requires explicit TRAIN/VALIDATION "
                "directories from the same Gold release"
            )
        audit = audit_gold_defense_release(release_root)
        if not audit.valid:
            detail = "; ".join(audit.violations[:8]) or "unknown Gold audit failure"
            raise ValueError("Gold Defense release audit failed: " + detail)
        gold_audit_sha = audit.audit_sha256

    maximum_tokens = _tokenization_preflight(
        config,
        list(training_rows) + list(validation_rows),
    )
    train_payload = _jsonl(training_rows)
    validation_payload = _jsonl(validation_rows)
    manifest = CyberMegatronDatasetManifest(
        run_id=config.run_id,
        model=config.model,
        model_revision=config.model_revision,
        stage=config.stage,
        seed=config.seed,
        source_examples_sha256=examples_sha,
        source_manifest_sha256=manifest_sha,
        validation_source_examples_sha256=validation_examples_sha,
        validation_source_manifest_sha256=validation_manifest_sha,
        source_promotion_eligible=source_promotion,
        validation_source_promotion_eligible=validation_promotion,
        gold_release_audit_sha256=gold_audit_sha,
        training_examples=len(training_rows),
        validation_examples=len(validation_rows),
        max_observed_sequence_tokens=maximum_tokens,
        train_jsonl_sha256=hashlib.sha256(train_payload).hexdigest(),
        validation_jsonl_sha256=hashlib.sha256(validation_payload).hexdigest(),
    )
    return train_payload, validation_payload, manifest


def _dataset_manifest_text(manifest: CyberMegatronDatasetManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def materialize_cyber_megatron_dataset(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path = ".",
) -> CyberMegatronDatasetManifest:
    root_path = Path(root).resolve()
    destination = resolve_under_root(root_path, config.dataset_dir)
    if destination.exists():
        raise FileExistsError(f"Cyber Megatron dataset already exists: {config.dataset_dir}")
    train_payload, validation_payload, manifest = _expected_dataset(config, root=root_path)
    destination.mkdir(parents=True)
    atomic_write(destination / "train.jsonl", train_payload.decode("utf-8"))
    atomic_write(
        destination / "validation.jsonl",
        validation_payload.decode("utf-8"),
    )
    atomic_write(
        destination / "manifest.json",
        _dataset_manifest_text(manifest),
    )
    return manifest


def verify_cyber_megatron_dataset(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path = ".",
) -> CyberMegatronDatasetVerification:
    root_path = Path(root).resolve()
    destination = resolve_under_root(root_path, config.dataset_dir)
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    manifest_path = destination / "manifest.json"
    missing = [
        path.name
        for path in (train_path, validation_path, manifest_path)
        if not path.is_file()
    ]
    if missing:
        return CyberMegatronDatasetVerification(
            valid=False,
            manifest=None,
            manifest_sha256=None,
            blockers=["materialized dataset is missing: " + ", ".join(missing)],
        )
    blockers: list[str] = []
    try:
        manifest_raw = manifest_path.read_bytes()
        actual_manifest = CyberMegatronDatasetManifest.model_validate_json(
            manifest_raw
        )
        _, _, expected_manifest = _expected_dataset(config, root=root_path)
    except (OSError, TypeError, ValueError) as exc:
        return CyberMegatronDatasetVerification(
            valid=False,
            manifest=None,
            manifest_sha256=None,
            blockers=[f"materialized dataset cannot be verified: {exc}"],
        )
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    validation_sha = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    if actual_manifest != expected_manifest:
        blockers.append("materialized dataset manifest differs from the current source corpus")
    if manifest_raw != _dataset_manifest_text(expected_manifest).encode("utf-8"):
        blockers.append("materialized dataset manifest is not canonical byte-for-byte")
    if train_sha != actual_manifest.train_jsonl_sha256:
        blockers.append("train.jsonl digest differs from materialized dataset manifest")
    if validation_sha != actual_manifest.validation_jsonl_sha256:
        blockers.append("validation.jsonl digest differs from materialized dataset manifest")
    return CyberMegatronDatasetVerification(
        valid=not blockers,
        manifest=actual_manifest,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        blockers=blockers,
    )


def _run_identity(
    config: CyberMegatronSFTConfig,
    dataset: CyberMegatronDatasetVerification,
) -> CyberMegatronRunIdentity:
    if not dataset.valid or dataset.manifest is None or dataset.manifest_sha256 is None:
        raise ValueError("verified dataset is required to derive the 397B run identity")
    manifest = dataset.manifest
    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-run-identity.v1",
        "run_id": config.run_id,
        "model": config.model,
        "model_revision": config.model_revision,
        "config_sha256": _config_sha256(config),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "source_examples_sha256": manifest.source_examples_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "validation_source_examples_sha256": (
            manifest.validation_source_examples_sha256
        ),
        "validation_source_manifest_sha256": (
            manifest.validation_source_manifest_sha256
        ),
        "train_jsonl_sha256": manifest.train_jsonl_sha256,
        "validation_jsonl_sha256": manifest.validation_jsonl_sha256,
        "output_dir": config.output_dir,
    }
    return CyberMegatronRunIdentity(
        **unsigned,
        identity_sha256=_sha256_canonical(unsigned),
    )


def _load_run_identity(path: Path) -> CyberMegatronRunIdentity:
    try:
        identity = CyberMegatronRunIdentity.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Cyber Megatron run identity: {path}") from exc
    unsigned = identity.model_dump(mode="json")
    observed = str(unsigned.pop("identity_sha256"))
    if _sha256_canonical(unsigned) != observed:
        raise ValueError(f"Cyber Megatron run identity self-hash does not verify: {path}")
    return identity


def _resolve_runtime_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_resume_binding(
    config: CyberMegatronSFTConfig,
    resume: CyberMegatronResume,
    identity: CyberMegatronRunIdentity,
    *,
    root: Path,
) -> None:
    output = resolve_under_root(root, config.output_dir).resolve()
    if not output.is_dir():
        raise ValueError("resume requires the existing bound output directory")
    identity_path = _resolve_runtime_path(root, resume.binding_manifest)
    expected_identity_path = (output / RUN_IDENTITY_FILENAME).resolve()
    if identity_path != expected_identity_path:
        raise ValueError(
            f"resume binding manifest must be {config.output_dir}/{RUN_IDENTITY_FILENAME}"
        )
    bound_identity = _load_run_identity(identity_path)
    if bound_identity != identity:
        raise ValueError("resume binding does not match the run, config and dataset digests")
    for label, value in (
        ("mcore_model", resume.mcore_model),
        ("mcore_adapter", resume.mcore_adapter),
    ):
        checkpoint = _resolve_runtime_path(root, value)
        if not checkpoint.is_dir():
            raise ValueError(f"resume {label} directory is missing: {value}")
        if not _path_is_within(checkpoint, output):
            raise ValueError(f"resume {label} must stay inside the bound output directory")


def _write_run_identity(path: Path, identity: CyberMegatronRunIdentity) -> None:
    atomic_write(
        path,
        json.dumps(identity.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _write_launch_state(
    output: Path,
    *,
    identity: CyberMegatronRunIdentity,
    launch_session: str,
    nodes: int,
    state: Literal["launching", "completed", "failed"],
) -> None:
    payload = CyberMegatronLaunchState(
        run_identity_sha256=identity.identity_sha256,
        launch_session_sha256=hashlib.sha256(launch_session.encode("utf-8")).hexdigest(),
        state=state,
        nodes=nodes,
    )
    atomic_write(
        output / LAUNCH_STATE_FILENAME,
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _coordinate_launch(
    config: CyberMegatronSFTConfig,
    identity: CyberMegatronRunIdentity,
    *,
    root: Path,
    node_rank: int,
    launch_session: str,
    resume: CyberMegatronResume | None,
) -> None:
    output = resolve_under_root(root, config.output_dir)
    identity_path = output / RUN_IDENTITY_FILENAME
    state_path = output / LAUNCH_STATE_FILENAME
    if node_rank == 0:
        if resume is None:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"Cyber Megatron output already exists: {config.output_dir}"
                ) from exc
            _write_run_identity(identity_path, identity)
        else:
            _validate_resume_binding(config, resume, identity, root=root)
        _write_launch_state(
            output,
            identity=identity,
            launch_session=launch_session,
            nodes=config.topology.nodes,
            state="launching",
        )
        return

    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if identity_path.is_file() and state_path.is_file():
            try:
                observed_identity = _load_run_identity(identity_path)
                state = CyberMegatronLaunchState.model_validate_json(
                    state_path.read_bytes()
                )
            except (OSError, ValueError):
                time.sleep(0.25)
                continue
            expected_session_sha = hashlib.sha256(
                launch_session.encode("utf-8")
            ).hexdigest()
            if observed_identity != identity:
                raise RuntimeError(
                    "shared output belongs to a different 397B run identity"
                )
            if state.run_identity_sha256 != identity.identity_sha256:
                raise RuntimeError("shared launch state has a different run identity")
            if state.launch_session_sha256 != expected_session_sha:
                raise RuntimeError("shared output belongs to a different launch session")
            if state.nodes != config.topology.nodes or state.state != "launching":
                raise RuntimeError("shared launch state is not active for this topology")
            return
        time.sleep(0.25)
    raise RuntimeError("timed out waiting for node 0 to bind the shared output directory")


def build_cyber_megatron_command(
    config: CyberMegatronSFTConfig,
    *,
    resume: CyberMegatronResume | None = None,
) -> list[str]:
    train_path = str(PurePosixPath(config.dataset_dir) / "train.jsonl")
    validation_path = str(PurePosixPath(config.dataset_dir) / "validation.jsonl")
    optimization = config.optimization
    topology = config.topology
    lora = config.lora
    checkpointing = config.checkpointing
    command = [
        "megatron",
        "sft",
        "--model",
        config.model,
        "--model_revision",
        config.model_revision,
        "--use_hf",
        "true",
        "--dataset",
        train_path,
        "--val_dataset",
        validation_path,
        "--split_dataset_ratio",
        "0",
        "--load_from_cache_file",
        "true",
        "--add_non_thinking_prefix",
        "true",
        "--loss_scale",
        "ignore_empty_think",
        "--tuner_type",
        lora.tuner_type,
        "--lora_rank",
        str(lora.rank),
        "--lora_alpha",
        str(lora.alpha),
        "--lora_dropout",
        str(lora.dropout),
        "--target_modules",
        lora.target_modules,
        "--language_model_only",
        "true",
        "--freeze_llm",
        "false",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--tensor_model_parallel_size",
        str(topology.tensor_model_parallel_size),
        "--pipeline_model_parallel_size",
        str(topology.pipeline_model_parallel_size),
        "--context_parallel_size",
        str(topology.context_parallel_size),
        "--expert_model_parallel_size",
        str(topology.expert_model_parallel_size),
        "--sequence_parallel",
        "true",
        "--moe_permute_fusion",
        "true",
        "--moe_grouped_gemm",
        "true",
        "--moe_shared_expert_overlap",
        "true",
        "--moe_aux_loss_coeff",
        str(optimization.moe_aux_loss_coeff),
        "--micro_batch_size",
        str(optimization.micro_batch_size),
        "--global_batch_size",
        str(optimization.global_batch_size),
        "--recompute_granularity",
        optimization.recompute_granularity,
        "--recompute_method",
        optimization.recompute_method,
        "--recompute_num_layers",
        str(optimization.recompute_num_layers),
        "--num_train_epochs",
        str(optimization.epochs),
        "--cross_entropy_loss_fusion",
        "true",
        "--lr",
        str(optimization.learning_rate),
        "--min_lr",
        str(optimization.minimum_learning_rate),
        "--lr_warmup_fraction",
        str(optimization.warmup_fraction),
        "--max_length",
        str(optimization.max_length),
        "--packing",
        "true",
        "--attention_backend",
        optimization.attention_backend,
        "--dataset_num_proc",
        str(optimization.dataset_num_proc),
        "--dataloader_num_workers",
        str(optimization.dataloader_num_workers),
        "--logging_steps",
        str(optimization.logging_steps),
        "--seed",
        str(config.seed),
        "--data_seed",
        str(config.seed),
        "--output_dir",
        config.output_dir,
        "--add_version",
        "true",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(checkpointing.save_steps),
        "--eval_steps",
        str(checkpointing.eval_steps),
        "--save_total_limit",
        str(checkpointing.save_total_limit),
        "--no_save_optim",
        "false",
        "--no_save_rng",
        "false",
        "--no_load_optim",
        "false",
        "--no_load_rng",
        "false",
        "--save_safetensors",
        "true",
        "--merge_lora",
        "true",
        "--use_distributed_optimizer",
        "true",
        "--check_model",
        "true",
    ]
    if resume is None:
        command.extend(["--finetune", "true"])
    else:
        command.extend(
            [
                "--mcore_model",
                resume.mcore_model,
                "--mcore_adapter",
                resume.mcore_adapter,
                "--finetune",
                "false",
            ]
        )
    return command


def plan_cyber_megatron_sft(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path = ".",
    resume: CyberMegatronResume | None = None,
    _coordinated_identity_sha256: str | None = None,
) -> CyberMegatronPlan:
    root_path = Path(root).resolve()
    dataset = verify_cyber_megatron_dataset(config, root=root_path)
    blockers = list(dataset.blockers)
    output = resolve_under_root(root_path, config.output_dir)
    identity: CyberMegatronRunIdentity | None = None
    if dataset.valid:
        try:
            identity = _run_identity(config, dataset)
        except ValueError as exc:
            blockers.append(str(exc))

    output_available = not output.exists()
    if output.exists():
        output_available = False
        if resume is not None and identity is not None:
            try:
                _validate_resume_binding(
                    config,
                    resume,
                    identity,
                    root=root_path,
                )
            except ValueError as exc:
                blockers.append(str(exc))
            else:
                output_available = True
        elif identity is not None and (
            _coordinated_identity_sha256 == identity.identity_sha256
        ):
            try:
                observed_identity = _load_run_identity(
                    output / RUN_IDENTITY_FILENAME
                )
            except ValueError as exc:
                blockers.append(str(exc))
            else:
                if observed_identity != identity:
                    blockers.append(
                        "existing Cyber Megatron output has a different run identity"
                    )
                else:
                    output_available = True
        else:
            blockers.append(f"Cyber Megatron output already exists: {config.output_dir}")
    elif resume is not None:
        output_available = False
        blockers.append("resume requires an existing bound output directory")
    if dataset.manifest is not None and dataset.manifest.validation_examples == 0:
        blockers.append("Cyber Megatron training requires a non-empty validation dataset")
    warnings = [
        "397B topology is a candidate contract; validate memory, NCCL and storage "
        "on the paid cluster",
        f"execution requires {config.launch_approval_env}={config.run_id}",
        f"all nodes require the same unique {config.launch_session_env}",
    ]
    if dataset.manifest is not None and dataset.manifest.training_examples < 100:
        warnings.append("training corpus has fewer than 100 examples; treat the run as smoke-only")
    command = build_cyber_megatron_command(config, resume=resume)
    return CyberMegatronPlan(
        run_id=config.run_id,
        model=config.model,
        model_revision=config.model_revision,
        backend=config.backend,
        ms_swift_version=config.ms_swift_version,
        world_size=config.topology.world_size,
        minimum_gpu_memory_gib=config.topology.minimum_gpu_memory_gib,
        data_parallel_size=config.topology.data_parallel_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataset_verified=dataset.valid,
        config_sha256=_config_sha256(config),
        dataset_manifest_sha256=dataset.manifest_sha256,
        source_examples_sha256=(
            dataset.manifest.source_examples_sha256
            if dataset.manifest is not None
            else None
        ),
        source_manifest_sha256=(
            dataset.manifest.source_manifest_sha256
            if dataset.manifest is not None
            else None
        ),
        validation_source_examples_sha256=(
            dataset.manifest.validation_source_examples_sha256
            if dataset.manifest is not None
            else None
        ),
        validation_source_manifest_sha256=(
            dataset.manifest.validation_source_manifest_sha256
            if dataset.manifest is not None
            else None
        ),
        train_jsonl_sha256=(
            dataset.manifest.train_jsonl_sha256
            if dataset.manifest is not None
            else None
        ),
        validation_jsonl_sha256=(
            dataset.manifest.validation_jsonl_sha256
            if dataset.manifest is not None
            else None
        ),
        gold_release_audit_sha256=(
            dataset.manifest.gold_release_audit_sha256
            if dataset.manifest is not None
            else None
        ),
        max_observed_sequence_tokens=(
            dataset.manifest.max_observed_sequence_tokens
            if dataset.manifest is not None
            else None
        ),
        run_identity_sha256=(
            identity.identity_sha256 if identity is not None else None
        ),
        output_available=output_available,
        static_ready=not blockers,
        launch_approval_env=config.launch_approval_env,
        launch_session_env=config.launch_session_env,
        command=command,
        blockers=blockers,
        warnings=warnings,
    )


def _assert_runtime_contract(config: CyberMegatronSFTConfig) -> tuple[int, str]:
    if os.environ.get(config.launch_approval_env) != config.run_id:
        raise RuntimeError(
            f"paid 397B launch requires {config.launch_approval_env}={config.run_id}"
        )
    launch_session = os.environ.get(config.launch_session_env, "")
    allowed_session_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not 16 <= len(launch_session) <= 128 or any(
        char not in allowed_session_chars for char in launch_session
    ):
        raise RuntimeError(
            f"distributed launch requires a unique 16-128 character "
            f"{config.launch_session_env}"
        )
    if shutil.which("megatron") is None:
        raise RuntimeError("Megatron-SWIFT executable 'megatron' is not installed")
    try:
        installed_version = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("ms-swift is not installed") from exc
    if installed_version != config.ms_swift_version:
        raise RuntimeError(
            f"ms-swift version mismatch: {installed_version} != {config.ms_swift_version}"
        )
    expected = {
        "NNODES": config.topology.nodes,
        "NPROC_PER_NODE": config.topology.gpus_per_node,
    }
    for name, required in expected.items():
        try:
            observed = int(os.environ[name])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"distributed launch requires integer {name}") from exc
        if observed != required:
            raise RuntimeError(f"{name} mismatch: {observed} != {required}")
    try:
        node_rank = int(os.environ["NODE_RANK"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("distributed launch requires integer NODE_RANK") from exc
    if not 0 <= node_rank < config.topology.nodes:
        raise RuntimeError("NODE_RANK is outside the configured cluster topology")
    if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
        raise RuntimeError("distributed launch requires MASTER_ADDR and MASTER_PORT")
    try:
        master_port = int(os.environ["MASTER_PORT"])
    except ValueError as exc:
        raise RuntimeError("MASTER_PORT must be an integer") from exc
    if not 1 <= master_port <= 65_535:
        raise RuntimeError("MASTER_PORT is outside the valid TCP port range")
    visible = [row for row in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if row]
    if len(visible) != config.topology.gpus_per_node:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match gpus_per_node")
    if any(not row.isdigit() for row in visible):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must use numeric GPU indices")
    visible_indices = [int(row) for row in visible]
    if len(set(visible_indices)) != len(visible_indices):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate GPU indices")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for the 397B GPU memory preflight")
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        memory_by_index = {
            int(index.strip()): float(memory.strip()) / 1024.0
            for index, memory in (line.split(",", 1) for line in completed.stdout.splitlines())
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError("nvidia-smi GPU memory preflight failed") from exc
    undersized = [
        index
        for index in visible_indices
        if memory_by_index.get(index, 0.0) + 0.5
        < config.topology.minimum_gpu_memory_gib
    ]
    if undersized:
        raise RuntimeError(
            "visible GPUs below minimum_gpu_memory_gib: "
            + ", ".join(str(index) for index in undersized)
        )
    return node_rank, launch_session


def execute_cyber_megatron_sft(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path = ".",
    resume: CyberMegatronResume | None = None,
) -> CyberMegatronPlan:
    root_path = Path(root).resolve()
    dataset = verify_cyber_megatron_dataset(config, root=root_path)
    if not dataset.valid:
        raise RuntimeError(
            "Cyber Megatron dataset is blocked: " + "; ".join(dataset.blockers)
        )
    if dataset.manifest is None or dataset.manifest.validation_examples == 0:
        raise RuntimeError("Cyber Megatron training requires a non-empty validation dataset")
    identity = _run_identity(config, dataset)
    node_rank, launch_session = _assert_runtime_contract(config)
    if resume is not None:
        try:
            _validate_resume_binding(
                config,
                resume,
                identity,
                root=root_path,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    _coordinate_launch(
        config,
        identity,
        root=root_path,
        node_rank=node_rank,
        launch_session=launch_session,
        resume=resume,
    )
    plan = plan_cyber_megatron_sft(
        config,
        root=root_path,
        resume=resume,
        _coordinated_identity_sha256=identity.identity_sha256,
    )
    if not plan.static_ready:
        if node_rank == 0:
            _write_launch_state(
                resolve_under_root(root_path, config.output_dir),
                identity=identity,
                launch_session=launch_session,
                nodes=config.topology.nodes,
                state="failed",
            )
        raise RuntimeError("Cyber Megatron plan is blocked: " + "; ".join(plan.blockers))
    try:
        subprocess.run(plan.command, cwd=root_path, check=True)
    except (OSError, subprocess.CalledProcessError):
        if node_rank == 0:
            _write_launch_state(
                resolve_under_root(root_path, config.output_dir),
                identity=identity,
                launch_session=launch_session,
                nodes=config.topology.nodes,
                state="failed",
            )
        raise
    if node_rank == 0:
        _write_launch_state(
            resolve_under_root(root_path, config.output_dir),
            identity=identity,
            launch_session=launch_session,
            nodes=config.topology.nodes,
            state="completed",
        )
    return plan
