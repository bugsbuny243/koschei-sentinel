from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.causal_defense_corpus import (
    CausalDefenseCorpusManifest,
    CausalDefenseTrainingExample,
)
from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseReflexCorpusManifestV3,
    DefenseReflexTrainingExampleV3,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json, resolve_under_root


SYSTEM_PROMPT = (
    "You are Koschei Sentinel's authorized cyber-defense reasoning model. "
    "Use only the supplied cyber-state and protected-asset scope. Return exactly one "
    "JSON object with keys interpretation and defense_sequence. Never invent evidence, "
    "targets, success claims, or future outcome evidence IDs. Prefer Guard when evidence "
    "is insufficient. Combat or Siege actions must remain inside the supplied protected "
    "environment. Every defense step must cite evidence already present in the supplied "
    "graph and require post-action outcome verification. Do not propose external retaliation."
)


class CyberSFTStage(StrEnum):
    DEFENSE_REFLEX = "DEFENSE_REFLEX"
    CAUSAL_DEFENSE = "CAUSAL_DEFENSE"


class CyberExecutionProfile(StrEnum):
    DENSE_SINGLE_GPU_QLORA = "DENSE_SINGLE_GPU_QLORA"
    MOE_DISTRIBUTED_REQUIRED = "MOE_DISTRIBUTED_REQUIRED"


class CyberQuantizationConfig(StrictModel):
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class CyberLoraConfig(StrictModel):
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_suffixes: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ],
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def suffixes_are_safe(self) -> "CyberLoraConfig":
        if len(self.target_suffixes) != len(set(self.target_suffixes)):
            raise ValueError("Cyber LoRA target suffixes must be unique")
        if any(not row or "/" in row or "\\" in row for row in self.target_suffixes):
            raise ValueError("Cyber LoRA target suffixes must be plain module names")
        return self


class CyberSFTConfig(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-config.v1"] = (
        "sentinel.cyber-sft-config.v1"
    )
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    stage: CyberSFTStage
    execution_profile: CyberExecutionProfile = CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA
    base_model: str = Field(min_length=3, max_length=256)
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    corpus_dir: str = Field(min_length=1, max_length=1024)
    validation_corpus_dir: str | None = Field(default=None, min_length=1, max_length=1024)
    output_dir: str = Field(min_length=1, max_length=1024)
    input_adapter_dir: str | None = Field(default=None, min_length=1, max_length=1024)
    max_sequence_length: int = Field(default=4096, ge=512, le=32768)
    epochs: float = Field(default=1.0, gt=0.0, le=20.0)
    learning_rate: float = Field(default=0.0001, gt=0.0, le=0.01)
    per_device_batch_size: int = Field(default=1, ge=1, le=32)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=1024)
    validation_ratio: float = Field(default=0.10, ge=0.0, le=0.5)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    logging_steps: int = Field(default=5, ge=1, le=10000)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    minimum_cuda_memory_gb: float = Field(default=0.0, ge=0.0, le=512.0)
    gradient_checkpointing: bool = True
    enable_router_aux_loss: bool = False
    quantization: CyberQuantizationConfig = Field(default_factory=CyberQuantizationConfig)
    lora: CyberLoraConfig = Field(default_factory=CyberLoraConfig)

    @model_validator(mode="after")
    def config_is_safe(self) -> "CyberSFTConfig":
        for field_name, value in (
            ("corpus_dir", self.corpus_dir),
            ("validation_corpus_dir", self.validation_corpus_dir),
            ("output_dir", self.output_dir),
            ("input_adapter_dir", self.input_adapter_dir),
        ):
            if value is None:
                continue
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or value.startswith("~"):
                raise ValueError(f"{field_name} must stay inside the repository root")
        if self.validation_corpus_dir is not None:
            if self.validation_corpus_dir == self.corpus_dir:
                raise ValueError("validation_corpus_dir must differ from corpus_dir")
            if self.validation_ratio != 0.0:
                raise ValueError(
                    "explicit validation_corpus_dir requires validation_ratio=0.0; "
                    "preassigned validation must never be re-split"
                )
        if self.base_model.startswith(("http://", "https://")):
            raise ValueError("base_model must be a registry identifier, not a URL")
        if self.base_model.count("/") != 1:
            raise ValueError("base_model must use owner/model format")
        if (
            self.execution_profile is CyberExecutionProfile.MOE_DISTRIBUTED_REQUIRED
            and self.gradient_checkpointing
        ):
            raise ValueError(
                "current Qwen3.5-MoE safety profile forbids gradient checkpointing; "
                "use the future distributed MoE executor"
            )
        if (
            self.execution_profile is CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA
            and self.enable_router_aux_loss
        ):
            raise ValueError("router auxiliary loss is only meaningful for an MoE execution profile")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


class CyberSFTPlan(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-plan.v1"] = (
        "sentinel.cyber-sft-plan.v1"
    )
    run_id: str
    stage: CyberSFTStage
    execution_profile: CyberExecutionProfile
    executable_with_current_trainer: bool
    corpus_promotion_eligible: bool | None
    base_model: str
    base_revision: str
    corpus_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_corpus_examples_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    validation_corpus_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    explicit_validation: bool = False
    example_count: int = Field(gt=0)
    training_examples: int = Field(gt=0)
    validation_examples: int = Field(ge=0)
    effective_batch_size: int = Field(gt=0)
    estimated_optimizer_steps: int = Field(gt=0)
    input_adapter_dir: str | None
    output_dir: str
    warnings: list[str] = Field(default_factory=list)


class _LoadedCorpus(StrictModel):
    rows: list[StrictModel]
    examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    promotion_eligible: bool | None


def load_cyber_sft_config(path: str | Path) -> CyberSFTConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Cyber SFT config is not valid JSON") from exc
    return CyberSFTConfig.model_validate(payload)


def _manifest_type(stage: CyberSFTStage):
    if stage is CyberSFTStage.DEFENSE_REFLEX:
        return DefenseReflexCorpusManifestV3
    return CausalDefenseCorpusManifest


def _example_type(stage: CyberSFTStage):
    if stage is CyberSFTStage.DEFENSE_REFLEX:
        return DefenseReflexTrainingExampleV3
    return CausalDefenseTrainingExample


def _load_cyber_sft_corpus(
    config: CyberSFTConfig,
    corpus_dir: str,
    *,
    root: str | Path = ".",
) -> _LoadedCorpus:
    root_path = Path(root).resolve()
    corpus = resolve_under_root(root_path, corpus_dir)
    examples_path = corpus / "examples.jsonl"
    manifest_path = corpus / "manifest.json"
    if not examples_path.is_file() or not manifest_path.is_file():
        raise ValueError("Cyber SFT corpus requires examples.jsonl and manifest.json")

    raw_examples = examples_path.read_bytes()
    raw_manifest = manifest_path.read_bytes()
    examples_sha = hashlib.sha256(raw_examples).hexdigest()
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    manifest = _manifest_type(config.stage).model_validate_json(raw_manifest)
    if not manifest.ready_for_training_pipeline:
        raise ValueError("Cyber SFT corpus manifest is not training-ready")
    if examples_sha != manifest.examples_sha256:
        raise ValueError("Cyber SFT examples digest differs from corpus manifest")

    model_type = _example_type(config.stage)
    rows: list[StrictModel] = []
    try:
        lines = raw_examples.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Cyber SFT examples.jsonl is not valid UTF-8") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(model_type.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid Cyber SFT example at line {line_number}"
            ) from exc
    if len(rows) != manifest.example_count:
        raise ValueError("Cyber SFT example count differs from corpus manifest")
    if not rows:
        raise ValueError("Cyber SFT corpus is empty")
    return _LoadedCorpus(
        rows=rows,
        examples_sha256=examples_sha,
        manifest_sha256=manifest_sha,
        promotion_eligible=getattr(manifest, "promotion_eligible", None),
    )


def load_cyber_sft_examples(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
) -> tuple[list[StrictModel], str, str, bool | None]:
    loaded = _load_cyber_sft_corpus(config, config.corpus_dir, root=root)
    return (
        loaded.rows,
        loaded.examples_sha256,
        loaded.manifest_sha256,
        loaded.promotion_eligible,
    )


def load_cyber_sft_validation_examples(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
) -> tuple[list[StrictModel], str, str, bool | None] | None:
    if config.validation_corpus_dir is None:
        return None
    loaded = _load_cyber_sft_corpus(config, config.validation_corpus_dir, root=root)
    return (
        loaded.rows,
        loaded.examples_sha256,
        loaded.manifest_sha256,
        loaded.promotion_eligible,
    )


def _bucket(example_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_cyber_sft_examples(
    rows: list[StrictModel],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[StrictModel], list[StrictModel]]:
    ordered = sorted(rows, key=lambda row: (_bucket(row.example_id, seed), row.example_id))
    if len(ordered) == 1 or validation_ratio == 0.0:
        return ordered, []
    validation_count = max(1, int(round(len(ordered) * validation_ratio)))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = ordered[:validation_count]
    training = ordered[validation_count:]
    return training, validation


def _combined_promotion_eligibility(
    training: bool | None,
    validation: bool | None,
) -> bool | None:
    if training is False or validation is False:
        return False
    if training is True and validation is True:
        return True
    return None


def _assert_explicit_split_disjoint(
    training_rows: list[StrictModel],
    validation_rows: list[StrictModel],
) -> None:
    training_example_ids = {str(row.example_id) for row in training_rows}
    validation_example_ids = {str(row.example_id) for row in validation_rows}
    overlap = sorted(training_example_ids & validation_example_ids)
    if overlap:
        raise ValueError(
            "explicit Cyber SFT TRAIN/VALIDATION example IDs overlap: "
            + ", ".join(overlap[:8])
        )

    training_scenario_ids = {
        str(row.scenario_id)
        for row in training_rows
        if hasattr(row, "scenario_id")
    }
    validation_scenario_ids = {
        str(row.scenario_id)
        for row in validation_rows
        if hasattr(row, "scenario_id")
    }
    scenario_overlap = sorted(training_scenario_ids & validation_scenario_ids)
    if scenario_overlap:
        raise ValueError(
            "explicit Cyber SFT TRAIN/VALIDATION scenario IDs overlap: "
            + ", ".join(scenario_overlap[:8])
        )


def plan_cyber_sft(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
) -> CyberSFTPlan:
    root_path = Path(root).resolve()
    rows, examples_sha, manifest_sha, promotion_eligible = load_cyber_sft_examples(
        config,
        root=root_path,
    )

    explicit_validation = config.validation_corpus_dir is not None
    validation_examples_sha: str | None = None
    validation_manifest_sha: str | None = None
    if explicit_validation:
        validation_loaded = load_cyber_sft_validation_examples(config, root=root_path)
        if validation_loaded is None:
            raise RuntimeError("explicit validation was requested but no validation corpus loaded")
        validation_rows, validation_examples_sha, validation_manifest_sha, validation_promotion = (
            validation_loaded
        )
        _assert_explicit_split_disjoint(rows, validation_rows)
        training = rows
        validation = validation_rows
        promotion_eligible = _combined_promotion_eligibility(
            promotion_eligible,
            validation_promotion,
        )
    else:
        training, validation = split_cyber_sft_examples(
            rows,
            validation_ratio=config.validation_ratio,
            seed=config.seed,
        )

    if not training:
        raise ValueError("Cyber SFT split produced no training examples")
    output = resolve_under_root(root_path, config.output_dir)
    if output.exists():
        raise FileExistsError(f"Cyber SFT output already exists: {config.output_dir}")
    if config.input_adapter_dir is not None:
        adapter = resolve_under_root(root_path, config.input_adapter_dir)
        if not adapter.is_dir():
            raise ValueError("Cyber SFT input_adapter_dir does not exist")

    steps_per_epoch = max(
        1,
        (len(training) + config.effective_batch_size - 1)
        // config.effective_batch_size,
    )
    estimated = max(1, int(steps_per_epoch * config.epochs + 0.999999))
    warnings: list[str] = []
    if not validation:
        warnings.append("Cyber SFT run has no validation split")
    if explicit_validation:
        warnings.append("Cyber SFT uses a preassigned explicit validation corpus")
    if len(training) + len(validation) < 100:
        warnings.append("Cyber SFT corpus has fewer than 100 examples; treat as smoke training")
    if promotion_eligible is False:
        warnings.append(
            "Cyber SFT corpus is smoke/training-authorized but not promotion-eligible"
        )
    executable = config.execution_profile is CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA
    if not executable:
        warnings.append(
            "MoE plan is intentionally non-executable with the single-GPU BitsAndBytes trainer; "
            "a distributed/pre-quantized MoE executor is required"
        )
    return CyberSFTPlan(
        run_id=config.run_id,
        stage=config.stage,
        execution_profile=config.execution_profile,
        executable_with_current_trainer=executable,
        corpus_promotion_eligible=promotion_eligible,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256=examples_sha,
        corpus_manifest_sha256=manifest_sha,
        validation_corpus_examples_sha256=validation_examples_sha,
        validation_corpus_manifest_sha256=validation_manifest_sha,
        explicit_validation=explicit_validation,
        example_count=len(training) + len(validation),
        training_examples=len(training),
        validation_examples=len(validation),
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=estimated,
        input_adapter_dir=config.input_adapter_dir,
        output_dir=config.output_dir,
        warnings=warnings,
    )


def _defense_reflex_input(row: DefenseReflexTrainingExampleV3) -> dict[str, object]:
    return {
        "task": "derive an evidence-grounded defensive plan",
        "scenario_id": row.scenario_id,
        "critical_entity_ids": row.critical_entity_ids,
        "graph_snapshots": row.graph_snapshots,
    }


def _causal_input(row: CausalDefenseTrainingExample) -> dict[str, object]:
    return {
        "task": "reason over an evolving attack and choose the reviewed defense sequence",
        "scenario_id": row.scenario_id,
        "temporal_snapshots": row.temporal_snapshots,
        "temporal_transitions": row.temporal_transitions,
        "world_line_observations": row.world_line_observations,
        "world_line_transitions": row.world_line_transitions,
    }


def cyber_sft_messages(row: StrictModel) -> list[dict[str, str]]:
    if isinstance(row, DefenseReflexTrainingExampleV3):
        user_payload = _defense_reflex_input(row)
        assistant_payload = {
            "interpretation": row.expected_interpretation,
            "defense_sequence": row.expected_sequence,
        }
    elif isinstance(row, CausalDefenseTrainingExample):
        user_payload = _causal_input(row)
        assistant_payload = {
            "interpretation": row.corrected_interpretation,
            "defense_sequence": row.expected_defense_sequence,
        }
    else:
        raise TypeError(f"unsupported Cyber SFT example type: {type(row).__name__}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(user_payload)},
        {"role": "assistant", "content": canonical_json(assistant_payload)},
    ]
