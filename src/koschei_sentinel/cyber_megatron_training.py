from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Literal

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
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json, resolve_under_root

QWEN35_397B_MODEL = "Qwen/Qwen3.5-397B-A17B"
QWEN35_397B_REVISION = "8472618112abcbd45acbcdc58436aff4233c23f7"
MS_SWIFT_VERSION = "4.5.2"
QWEN35_EXPERT_COUNT = 512
LAUNCH_APPROVAL_ENV = "KOSCHEI_397B_LAUNCH_APPROVED"


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
        if self.world_size % self.expert_model_parallel_size:
            raise ValueError("world size must be divisible by expert parallel size")
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
    training_examples: int = Field(gt=0)
    validation_examples: int = Field(ge=0)
    train_jsonl_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_jsonl_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CyberMegatronDatasetVerification(StrictModel):
    valid: bool
    manifest: CyberMegatronDatasetManifest | None
    blockers: list[str]


class CyberMegatronResume(StrictModel):
    mcore_model: str = Field(min_length=1, max_length=4096)
    mcore_adapter: str = Field(min_length=1, max_length=4096)


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
    output_available: bool
    static_ready: bool
    requires_explicit_launch_approval: Literal[True] = True
    launch_approval_env: Literal["KOSCHEI_397B_LAUNCH_APPROVED"]
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


def _expected_dataset(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path,
) -> tuple[bytes, bytes, CyberMegatronDatasetManifest]:
    source = _source_config(config)
    rows, examples_sha, manifest_sha, _ = load_cyber_sft_examples(source, root=root)
    explicit = load_cyber_sft_validation_examples(source, root=root)
    validation_examples_sha: str | None = None
    validation_manifest_sha: str | None = None
    if explicit is None:
        training_rows, validation_rows = split_cyber_sft_examples(
            rows,
            validation_ratio=config.validation_ratio,
            seed=config.seed,
        )
    else:
        validation_rows, validation_examples_sha, validation_manifest_sha, _ = explicit
        training_rows = rows
        _assert_disjoint(training_rows, validation_rows)
    if not training_rows:
        raise ValueError("Cyber Megatron dataset contains no training examples")
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
        training_examples=len(training_rows),
        validation_examples=len(validation_rows),
        train_jsonl_sha256=hashlib.sha256(train_payload).hexdigest(),
        validation_jsonl_sha256=hashlib.sha256(validation_payload).hexdigest(),
    )
    return train_payload, validation_payload, manifest


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
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
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
            blockers=["materialized dataset is missing: " + ", ".join(missing)],
        )
    blockers: list[str] = []
    try:
        actual_manifest = CyberMegatronDatasetManifest.model_validate_json(
            manifest_path.read_bytes()
        )
        _, _, expected_manifest = _expected_dataset(config, root=root_path)
    except (OSError, TypeError, ValueError) as exc:
        return CyberMegatronDatasetVerification(
            valid=False,
            manifest=None,
            blockers=[f"materialized dataset cannot be verified: {exc}"],
        )
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    validation_sha = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    if actual_manifest != expected_manifest:
        blockers.append("materialized dataset manifest differs from the current source corpus")
    if train_sha != actual_manifest.train_jsonl_sha256:
        blockers.append("train.jsonl digest differs from materialized dataset manifest")
    if validation_sha != actual_manifest.validation_jsonl_sha256:
        blockers.append("validation.jsonl digest differs from materialized dataset manifest")
    return CyberMegatronDatasetVerification(
        valid=not blockers,
        manifest=actual_manifest,
        blockers=blockers,
    )


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
) -> CyberMegatronPlan:
    root_path = Path(root).resolve()
    dataset = verify_cyber_megatron_dataset(config, root=root_path)
    blockers = list(dataset.blockers)
    output = resolve_under_root(root_path, config.output_dir)
    output_available = not output.exists()
    if not output_available and resume is None:
        blockers.append(f"Cyber Megatron output already exists: {config.output_dir}")
    if dataset.manifest is not None and dataset.manifest.validation_examples == 0:
        blockers.append("Cyber Megatron training requires a non-empty validation dataset")
    warnings = [
        "397B topology is a candidate contract; validate memory, NCCL and storage "
        "on the paid cluster",
        f"execution requires {config.launch_approval_env}={config.run_id}",
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
        output_available=output_available,
        static_ready=not blockers,
        launch_approval_env=config.launch_approval_env,
        command=command,
        blockers=blockers,
        warnings=warnings,
    )


def _assert_runtime_contract(config: CyberMegatronSFTConfig) -> None:
    if os.environ.get(config.launch_approval_env) != config.run_id:
        raise RuntimeError(
            f"paid 397B launch requires {config.launch_approval_env}={config.run_id}"
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
            index.strip(): float(memory.strip()) / 1024.0
            for index, memory in (line.split(",", 1) for line in completed.stdout.splitlines())
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError("nvidia-smi GPU memory preflight failed") from exc
    undersized = [
        index
        for index in visible
        if memory_by_index.get(index, 0.0) + 0.5
        < config.topology.minimum_gpu_memory_gib
    ]
    if undersized:
        raise RuntimeError(
            "visible GPUs below minimum_gpu_memory_gib: " + ", ".join(undersized)
        )


def execute_cyber_megatron_sft(
    config: CyberMegatronSFTConfig,
    *,
    root: str | Path = ".",
    resume: CyberMegatronResume | None = None,
) -> CyberMegatronPlan:
    plan = plan_cyber_megatron_sft(config, root=root, resume=resume)
    if not plan.static_ready:
        raise RuntimeError("Cyber Megatron plan is blocked: " + "; ".join(plan.blockers))
    _assert_runtime_contract(config)
    if resume is not None:
        missing = [
            path
            for path in (resume.mcore_model, resume.mcore_adapter)
            if not Path(path).is_dir()
        ]
        if missing:
            raise RuntimeError("resume checkpoint directories are missing: " + ", ".join(missing))
    subprocess.run(plan.command, cwd=Path(root).resolve(), check=True)
    return plan
