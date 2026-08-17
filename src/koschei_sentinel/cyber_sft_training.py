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
from koschei_sentinel.defense_reflex_corpus_v2 import (
    DefenseReflexCorpusManifestV2,
    DefenseReflexTrainingExampleV2,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json, resolve_under_root


SYSTEM_PROMPT = (
    "You are Koschei Sentinel's authorized cyber-defense reasoning model. "
    "Use only the supplied cyber-state, evidence lineage, protected-asset scope and "
    "reviewed observations. Return exactly one JSON object with keys "
    "interpretation and defense_sequence. Never invent evidence or targets. "
    "Prefer Guard when evidence is insufficient. Combat or Siege actions must remain "
    "inside the supplied protected environment. Do not propose external retaliation."
)


class CyberSFTStage(StrEnum):
    DEFENSE_REFLEX = "DEFENSE_REFLEX"
    CAUSAL_DEFENSE = "CAUSAL_DEFENSE"


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
    base_model: str = Field(min_length=3, max_length=256)
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    corpus_dir: str = Field(min_length=1, max_length=1024)
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
    quantization: CyberQuantizationConfig = Field(default_factory=CyberQuantizationConfig)
    lora: CyberLoraConfig = Field(default_factory=CyberLoraConfig)

    @model_validator(mode="after")
    def config_is_safe(self) -> "CyberSFTConfig":
        for field_name, value in (
            ("corpus_dir", self.corpus_dir),
            ("output_dir", self.output_dir),
            ("input_adapter_dir", self.input_adapter_dir),
        ):
            if value is None:
                continue
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or value.startswith("~"):
                raise ValueError(f"{field_name} must stay inside the repository root")
        if self.base_model.startswith(("http://", "https://")):
            raise ValueError("base_model must be a registry identifier, not a URL")
        if self.base_model.count("/") != 1:
            raise ValueError("base_model must use owner/model format")
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
    base_model: str
    base_revision: str
    corpus_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    example_count: int = Field(gt=0)
    training_examples: int = Field(gt=0)
    validation_examples: int = Field(ge=0)
    effective_batch_size: int = Field(gt=0)
    estimated_optimizer_steps: int = Field(gt=0)
    input_adapter_dir: str | None
    output_dir: str
    warnings: list[str] = Field(default_factory=list)


def load_cyber_sft_config(path: str | Path) -> CyberSFTConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Cyber SFT config is not valid JSON") from exc
    return CyberSFTConfig.model_validate(payload)


def _manifest_type(stage: CyberSFTStage):
    if stage is CyberSFTStage.DEFENSE_REFLEX:
        return DefenseReflexCorpusManifestV2
    return CausalDefenseCorpusManifest


def _example_type(stage: CyberSFTStage):
    if stage is CyberSFTStage.DEFENSE_REFLEX:
        return DefenseReflexTrainingExampleV2
    return CausalDefenseTrainingExample


def load_cyber_sft_examples(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
) -> tuple[list[StrictModel], str, str]:
    root_path = Path(root).resolve()
    corpus = resolve_under_root(root_path, config.corpus_dir)
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
    return rows, examples_sha, manifest_sha


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


def plan_cyber_sft(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
) -> CyberSFTPlan:
    root_path = Path(root).resolve()
    rows, examples_sha, manifest_sha = load_cyber_sft_examples(config, root=root_path)
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
    if len(rows) < 100:
        warnings.append("Cyber SFT corpus has fewer than 100 examples; treat as smoke training")
    return CyberSFTPlan(
        run_id=config.run_id,
        stage=config.stage,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256=examples_sha,
        corpus_manifest_sha256=manifest_sha,
        example_count=len(rows),
        training_examples=len(training),
        validation_examples=len(validation),
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=estimated,
        input_adapter_dir=config.input_adapter_dir,
        output_dir=config.output_dir,
        warnings=warnings,
    )


def _defense_reflex_input(row: DefenseReflexTrainingExampleV2) -> dict[str, object]:
    return {
        "task": "derive an evidence-grounded defensive correction",
        "scenario_id": row.scenario_id,
        "failure_type": row.failure_type,
        "critical_entity_ids": row.critical_entity_ids,
        "graph_snapshots": row.graph_snapshots,
        "observed_ticks": row.observed_ticks,
        "review_evidence_ids": row.review_evidence_ids,
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
    if isinstance(row, DefenseReflexTrainingExampleV2):
        user_payload = _defense_reflex_input(row)
        assistant_payload = {
            "interpretation": row.corrected_interpretation,
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
