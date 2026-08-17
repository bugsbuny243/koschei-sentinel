from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.language_foundation import (
    LanguageFoundationDocument,
    LanguageFoundationReleaseManifest,
    verify_language_foundation_release,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import (
    LoraConfig,
    QuantizationConfig,
    atomic_write,
    model_digest,
    resolve_under_root,
)

_DIGEST = r"^[a-f0-9]{64}$"
_COMMIT = r"^[a-f0-9]{40}$"


class LanguageTrainingConfig(StrictModel):
    schema_version: Literal["sentinel.language-training-config.v1"] = (
        "sentinel.language-training-config.v1"
    )
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    base_model: str = Field(min_length=3, max_length=256)
    base_revision: str = Field(pattern=_COMMIT)
    language_release: str = Field(min_length=1, max_length=1024)
    expected_source_commit: str = Field(pattern=_COMMIT)
    expected_source_corpus_sha256: str = Field(pattern=_DIGEST)
    output_dir: str = Field(min_length=1, max_length=1024)
    max_sequence_length: int = Field(default=2048, ge=256, le=32768)
    epochs: float = Field(default=1.0, gt=0.0, le=10.0)
    learning_rate: float = Field(default=0.0001, gt=0.0, le=0.01)
    per_device_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=4096)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    logging_steps: int = Field(default=5, ge=1, le=10000)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    trust_remote_code: Literal[False] = False
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)

    @model_validator(mode="after")
    def paths_and_model_are_safe(self) -> LanguageTrainingConfig:
        _validate_relative_path(self.language_release, "language_release")
        _validate_relative_path(self.output_dir, "output_dir")
        if self.base_model.startswith(("http://", "https://")):
            raise ValueError("base_model must be a registry identifier, not a URL")
        if self.base_model.count("/") != 1:
            raise ValueError("base_model must use owner/model format")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


class LanguageTrainingPlan(StrictModel):
    schema_version: Literal["sentinel.language-training-plan.v1"] = (
        "sentinel.language-training-plan.v1"
    )
    lineage_stage: Literal["stage1_language_foundation"] = "stage1_language_foundation"
    authority: Literal["offline_language_research_only"] = "offline_language_research_only"
    dry_run: Literal[True] = True
    run_id: str
    base_model: str
    base_revision: str = Field(pattern=_COMMIT)
    source_repository: Literal["bugsbuny243/koschei-lang"] = "bugsbuny243/koschei-lang"
    source_commit: str = Field(pattern=_COMMIT)
    source_corpus_sha256: str = Field(pattern=_DIGEST)
    release_manifest_digest: str = Field(pattern=_DIGEST)
    train_split_digest: str = Field(pattern=_DIGEST)
    validation_split_digest: str = Field(pattern=_DIGEST)
    test_split_digest: str = Field(pattern=_DIGEST)
    train_documents: int = Field(ge=1)
    validation_documents: int = Field(ge=1)
    test_documents: int = Field(ge=1)
    train_families: int = Field(ge=1)
    validation_families: int = Field(ge=1)
    test_families: int = Field(ge=1)
    training_config_digest: str = Field(pattern=_DIGEST)
    effective_batch_size: int = Field(ge=1)
    estimated_document_steps: int = Field(ge=1)
    output_dir: str


class LanguageAdapterManifest(StrictModel):
    schema_version: Literal["sentinel.language-adapter-manifest.v1"] = (
        "sentinel.language-adapter-manifest.v1"
    )
    lineage_stage: Literal["stage1_language_foundation"] = "stage1_language_foundation"
    authority: Literal["offline_language_research_only"] = "offline_language_research_only"
    run_id: str
    base_model: str
    base_revision: str = Field(pattern=_COMMIT)
    source_repository: Literal["bugsbuny243/koschei-lang"] = "bugsbuny243/koschei-lang"
    source_commit: str = Field(pattern=_COMMIT)
    source_corpus_sha256: str = Field(pattern=_DIGEST)
    release_manifest_digest: str = Field(pattern=_DIGEST)
    train_split_digest: str = Field(pattern=_DIGEST)
    validation_split_digest: str = Field(pattern=_DIGEST)
    held_out_test_split_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    train_documents: int = Field(ge=1)
    validation_documents: int = Field(ge=1)
    held_out_test_documents: int = Field(ge=1)
    train_chunks: int = Field(ge=1)
    validation_chunks: int = Field(ge=1)
    adapter_digest: str = Field(pattern=_DIGEST)
    adapter_files: list[str] = Field(min_length=1, max_length=256)
    output_dir: str
    train_loss: float | None = None
    eval_loss: float | None = None


def load_language_training_config(path: str | Path) -> LanguageTrainingConfig:
    try:
        return LanguageTrainingConfig.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid language-training config") from exc


def plan_language_training(
    config: LanguageTrainingConfig,
    *,
    root: str | Path = ".",
) -> LanguageTrainingPlan:
    root_path = Path(root).resolve()
    release_path = resolve_under_root(root_path, config.language_release)
    manifest = verify_language_foundation_release(release_path)
    _verify_source_pin(config, manifest)

    output_path = resolve_under_root(root_path, config.output_dir)
    if output_path.exists():
        raise FileExistsError(f"language-training output already exists: {config.output_dir}")

    manifest_path = release_path / "language-foundation-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("language foundation manifest is missing")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    train = manifest.splits["train"]
    validation = manifest.splits["validation"]
    test = manifest.splits["test"]
    document_batches = max(
        1,
        (train.documents + config.effective_batch_size - 1)
        // config.effective_batch_size,
    )
    estimated_steps = max(1, int(document_batches * config.epochs + 0.999999))

    return LanguageTrainingPlan(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        source_commit=manifest.source_commit,
        source_corpus_sha256=manifest.source_corpus_sha256,
        release_manifest_digest=manifest_digest,
        train_split_digest=train.digest,
        validation_split_digest=validation.digest,
        test_split_digest=test.digest,
        train_documents=train.documents,
        validation_documents=validation.documents,
        test_documents=test.documents,
        train_families=train.families,
        validation_families=validation.families,
        test_families=test.families,
        training_config_digest=model_digest(config),
        effective_batch_size=config.effective_batch_size,
        estimated_document_steps=estimated_steps,
        output_dir=config.output_dir,
    )


def write_language_training_plan(plan: LanguageTrainingPlan, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"language-training plan already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def execute_language_training(
    config: LanguageTrainingConfig,
    plan: LanguageTrainingPlan,
    *,
    root: str | Path = ".",
) -> LanguageAdapterManifest:
    if plan.training_config_digest != model_digest(config):
        raise ValueError("language-training plan does not match supplied config")
    recomputed = plan_language_training(config, root=root)
    if recomputed != plan:
        raise ValueError("language-training plan is stale or release lineage changed")

    dependencies = _load_dependencies()
    root_path = Path(root).resolve()
    output_path = resolve_under_root(root_path, config.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        manifest = _train(config, plan, root_path, staging, dependencies)
        os.replace(staging, output_path)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _train(
    config: LanguageTrainingConfig,
    plan: LanguageTrainingPlan,
    root: Path,
    staging: Path,
    dependencies: dict[str, Any],
) -> LanguageAdapterManifest:
    torch = dependencies["torch"]
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if config.quantization.compute_dtype == "bfloat16"
        else torch.float16
    )
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=config.quantization.bits == 4,
        load_in_8bit=config.quantization.bits == 8,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=dtype,
    )
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map="auto",
    )
    model = dependencies["prepare_model_for_kbit_training"](model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = dependencies["get_peft_model"](
        model,
        dependencies["PeftLoraConfig"](
            task_type="CAUSAL_LM",
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        ),
    )

    release = resolve_under_root(root, config.language_release)
    train_documents = _load_split_documents(release, "train")
    validation_documents = _load_split_documents(release, "validation")
    train_rows = _tokenize_documents(
        train_documents,
        tokenizer,
        config.max_sequence_length,
    )
    validation_rows = _tokenize_documents(
        validation_documents,
        tokenizer,
        config.max_sequence_length,
    )
    if not train_rows:
        raise ValueError("language train split produced no token chunks")
    if not validation_rows:
        raise ValueError("language validation split produced no token chunks")

    dataset_type = dependencies["Dataset"]
    train_dataset = dataset_type.from_list(train_rows)
    eval_dataset = dataset_type.from_list(validation_rows)
    arguments_type = dependencies["TrainingArguments"]
    values: dict[str, Any] = {
        "output_dir": str(staging / "checkpoints"),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.seed,
        "bf16": config.quantization.compute_dtype == "bfloat16",
        "fp16": config.quantization.compute_dtype == "float16",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
    }
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(arguments_type.__init__).parameters
        else "evaluation_strategy"
    )
    values[strategy_key] = "epoch"
    trainer = dependencies["Trainer"](
        model=model,
        args=arguments_type(**values),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=dependencies["DataCollatorForLanguageModeling"](
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_path = staging / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    files = sorted(
        str(path.relative_to(staging))
        for path in adapter_path.rglob("*")
        if path.is_file()
    )
    if not files:
        raise ValueError("language training produced no adapter files")
    manifest = LanguageAdapterManifest(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        source_commit=plan.source_commit,
        source_corpus_sha256=plan.source_corpus_sha256,
        release_manifest_digest=plan.release_manifest_digest,
        train_split_digest=plan.train_split_digest,
        validation_split_digest=plan.validation_split_digest,
        held_out_test_split_digest=plan.test_split_digest,
        training_config_digest=plan.training_config_digest,
        train_documents=plan.train_documents,
        validation_documents=plan.validation_documents,
        held_out_test_documents=plan.test_documents,
        train_chunks=len(train_rows),
        validation_chunks=len(validation_rows),
        adapter_digest=_directory_digest(staging, files),
        adapter_files=files,
        output_dir=config.output_dir,
        train_loss=_metric_float(train_result.metrics.get("train_loss")),
        eval_loss=_metric_float(eval_metrics.get("eval_loss")),
    )
    (staging / "language-adapter-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _verify_source_pin(
    config: LanguageTrainingConfig,
    manifest: LanguageFoundationReleaseManifest,
) -> None:
    if manifest.source_commit != config.expected_source_commit:
        raise ValueError("language release source commit does not match training config")
    if manifest.source_corpus_sha256 != config.expected_source_corpus_sha256:
        raise ValueError("language release corpus digest does not match training config")


def _load_split_documents(
    release: Path,
    split: str,
) -> list[LanguageFoundationDocument]:
    path = release / f"{split}.jsonl"
    if not path.is_file():
        raise ValueError(f"language foundation split is missing: {split}")
    documents: list[LanguageFoundationDocument] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank language foundation row at {split}:{line_number}")
        try:
            documents.append(LanguageFoundationDocument.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid language foundation row at {split}:{line_number}"
            ) from exc
    return documents


def _tokenize_documents(
    documents: list[LanguageFoundationDocument],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    for document in documents:
        encoded = tokenizer(document.text, add_special_tokens=False)
        token_ids = list(encoded["input_ids"])
        eos = tokenizer.eos_token_id
        if eos is not None and (not token_ids or token_ids[-1] != eos):
            token_ids.append(eos)
        for start in range(0, len(token_ids), max_length):
            chunk = token_ids[start : start + max_length]
            if len(chunk) < 2:
                continue
            rows.append(
                {
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk),
                }
            )
    return rows


def _directory_digest(root: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metric_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{field} must stay within the repository root")


def _load_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install the project with .[training]"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "PeftLoraConfig": PeftLoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }
