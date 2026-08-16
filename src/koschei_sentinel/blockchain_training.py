from __future__ import annotations

import ctypes
import errno
import hashlib
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityDocument,
    load_blockchain_security_documents,
)
from koschei_sentinel.blockchain_security_release import (
    BlockchainSecurityReleaseManifest,
    verify_blockchain_security_release,
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
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class BlockchainTrainingConfig(StrictModel):
    schema_version: Literal["sentinel.blockchain-training-config.v1"] = (
        "sentinel.blockchain-training-config.v1"
    )
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    base_model: str = Field(min_length=3, max_length=256)
    base_revision: str = Field(pattern=_COMMIT)
    blockchain_release: str = Field(min_length=1, max_length=1024)
    expected_source_corpus_digest: str = Field(pattern=_DIGEST)
    expected_benchmark_suite_digest: str = Field(pattern=_DIGEST)
    output_dir: str = Field(min_length=1, max_length=1024)
    max_sequence_length: int = Field(default=4096, ge=512, le=32768)
    epochs: float = Field(default=1.0, gt=0.0, le=10.0)
    learning_rate: float = Field(default=0.00005, gt=0.0, le=0.01)
    per_device_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=32, ge=1, le=4096)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    logging_steps: int = Field(default=5, ge=1, le=10000)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    trust_remote_code: Literal[False] = False
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)

    @model_validator(mode="after")
    def inputs_are_safe(self) -> BlockchainTrainingConfig:
        _validate_relative_path(self.blockchain_release, "blockchain_release")
        _validate_relative_path(self.output_dir, "output_dir")
        if self.base_model.startswith(("http://", "https://")):
            raise ValueError("base_model must be a registry identifier, not a URL")
        if self.base_model.count("/") != 1:
            raise ValueError("base_model must use owner/model format")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


class BlockchainTrainingPlan(StrictModel):
    schema_version: Literal["sentinel.blockchain-training-plan.v1"] = (
        "sentinel.blockchain-training-plan.v1"
    )
    lineage_stage: Literal["stage2_blockchain_continued_pretraining"] = (
        "stage2_blockchain_continued_pretraining"
    )
    authority: Literal["offline_blockchain_research_only"] = (
        "offline_blockchain_research_only"
    )
    dry_run: Literal[True] = True
    run_id: str
    base_model: str
    base_revision: str = Field(pattern=_COMMIT)
    release_manifest_file_digest: str = Field(pattern=_DIGEST)
    release_manifest_digest: str = Field(pattern=_DIGEST)
    source_corpus_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    holdout_digest: str = Field(pattern=_DIGEST)
    release_policy_digest: str = Field(pattern=_DIGEST)
    train_split_digest: str = Field(pattern=_DIGEST)
    validation_split_digest: str = Field(pattern=_DIGEST)
    held_out_test_split_digest: str = Field(pattern=_DIGEST)
    train_documents: int = Field(ge=1)
    validation_documents: int = Field(ge=1)
    held_out_test_documents: int = Field(ge=1)
    train_sources: int = Field(ge=1)
    validation_sources: int = Field(ge=1)
    held_out_test_sources: int = Field(ge=1)
    training_config_digest: str = Field(pattern=_DIGEST)
    effective_batch_size: int = Field(ge=1)
    estimated_document_steps: int = Field(ge=1)
    output_dir: str


class BlockchainAdapterManifest(StrictModel):
    schema_version: Literal["sentinel.blockchain-adapter-manifest.v1"] = (
        "sentinel.blockchain-adapter-manifest.v1"
    )
    lineage_stage: Literal["stage2_blockchain_continued_pretraining"] = (
        "stage2_blockchain_continued_pretraining"
    )
    authority: Literal["offline_blockchain_research_only"] = (
        "offline_blockchain_research_only"
    )
    run_id: str
    base_model: str
    base_revision: str = Field(pattern=_COMMIT)
    release_manifest_file_digest: str = Field(pattern=_DIGEST)
    release_manifest_digest: str = Field(pattern=_DIGEST)
    source_corpus_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    train_split_digest: str = Field(pattern=_DIGEST)
    validation_split_digest: str = Field(pattern=_DIGEST)
    held_out_test_split_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    train_documents: int = Field(ge=1)
    validation_documents: int = Field(ge=1)
    held_out_test_documents: int = Field(ge=1)
    held_out_test_sources: int = Field(ge=1)
    train_chunks: int = Field(ge=1)
    validation_chunks: int = Field(ge=1)
    adapter_digest: str = Field(pattern=_DIGEST)
    adapter_files: list[str] = Field(min_length=1, max_length=256)
    output_dir: str
    train_loss: float | None = None
    eval_loss: float | None = None
    held_out_test_consumed: Literal[False] = False


def load_blockchain_training_config(path: str | Path) -> BlockchainTrainingConfig:
    try:
        return BlockchainTrainingConfig.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain-training config") from exc


def plan_blockchain_training(
    config: BlockchainTrainingConfig,
    *,
    root: str | Path = ".",
) -> BlockchainTrainingPlan:
    root_path = Path(root).resolve()
    release = resolve_under_root(root_path, config.blockchain_release)
    manifest = verify_blockchain_security_release(release)
    _verify_expected_lineage(config, manifest)

    output = resolve_under_root(root_path, config.output_dir)
    if output.exists():
        raise FileExistsError(f"blockchain-training output already exists: {config.output_dir}")

    manifest_path = release / "blockchain-security-release-manifest.json"
    manifest_file_digest = _hash_file(manifest_path)
    train = manifest.splits["train"]
    validation = manifest.splits["validation"]
    test = manifest.splits["test"]
    document_batches = max(
        1,
        (train.documents + config.effective_batch_size - 1)
        // config.effective_batch_size,
    )
    estimated_steps = max(1, int(document_batches * config.epochs + 0.999999))

    return BlockchainTrainingPlan(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        release_manifest_file_digest=manifest_file_digest,
        release_manifest_digest=_model_digest(manifest),
        source_corpus_digest=manifest.source_corpus_digest,
        benchmark_suite_digest=manifest.benchmark_suite_digest,
        holdout_digest=manifest.holdout_digest,
        release_policy_digest=manifest.release_policy_digest,
        train_split_digest=train.digest,
        validation_split_digest=validation.digest,
        held_out_test_split_digest=test.digest,
        train_documents=train.documents,
        validation_documents=validation.documents,
        held_out_test_documents=test.documents,
        train_sources=train.sources,
        validation_sources=validation.sources,
        held_out_test_sources=test.sources,
        training_config_digest=model_digest(config),
        effective_batch_size=config.effective_batch_size,
        estimated_document_steps=estimated_steps,
        output_dir=config.output_dir,
    )


def write_blockchain_training_plan(plan: BlockchainTrainingPlan, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"blockchain-training plan already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def execute_blockchain_training(
    config: BlockchainTrainingConfig,
    plan: BlockchainTrainingPlan,
    *,
    root: str | Path = ".",
) -> BlockchainAdapterManifest:
    if plan.training_config_digest != model_digest(config):
        raise ValueError("blockchain-training plan does not match supplied config")
    root_path = Path(root).resolve()
    release, manifest = _verify_execution_lineage(config, plan, root=root_path)
    output = resolve_under_root(root_path, config.output_dir)
    if output.exists():
        raise FileExistsError(f"blockchain-training output already exists: {config.output_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)

    dependencies = _load_dependencies()
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        adapter_manifest = _train(
            config,
            plan,
            release,
            manifest,
            staging,
            dependencies,
        )
        _fsync_tree(staging)
        _publish_directory_no_replace(staging, output)
        return adapter_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_execution_lineage(
    config: BlockchainTrainingConfig,
    plan: BlockchainTrainingPlan,
    *,
    root: Path,
) -> tuple[Path, BlockchainSecurityReleaseManifest]:
    release = resolve_under_root(root, config.blockchain_release)
    manifest_path = release / "blockchain-security-release-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("blockchain release manifest is missing")
    if _hash_file(manifest_path) != plan.release_manifest_file_digest:
        raise ValueError("blockchain release manifest bytes changed after planning")
    try:
        manifest = BlockchainSecurityReleaseManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain release manifest") from exc
    if _model_digest(manifest) != plan.release_manifest_digest:
        raise ValueError("blockchain release semantic manifest changed after planning")
    _verify_expected_lineage(config, manifest)

    train = manifest.splits["train"]
    validation = manifest.splits["validation"]
    test = manifest.splits["test"]
    if train.digest != plan.train_split_digest or train.documents != plan.train_documents:
        raise ValueError("planned train split no longer matches release manifest")
    if validation.digest != plan.validation_split_digest or (
        validation.documents != plan.validation_documents
    ):
        raise ValueError("planned validation split no longer matches release manifest")
    if test.digest != plan.held_out_test_split_digest or (
        test.documents != plan.held_out_test_documents
    ):
        raise ValueError("planned held-out test metadata no longer matches release manifest")
    if test.sources != plan.held_out_test_sources:
        raise ValueError("planned held-out test source count no longer matches release manifest")

    for split_name, expected_digest in (
        ("train", plan.train_split_digest),
        ("validation", plan.validation_split_digest),
    ):
        split_path = release / f"{split_name}.jsonl"
        if not split_path.is_file() or split_path.is_symlink():
            raise ValueError(f"blockchain {split_name} split is missing")
        if _hash_file(split_path) != expected_digest:
            raise ValueError(f"blockchain {split_name} split changed after planning")

    # Intentionally do not open/hash/parse test.jsonl here. The executor binds its digest
    # from the already verified release manifest and leaves held-out bytes untouched.
    return release, manifest


def _train(
    config: BlockchainTrainingConfig,
    plan: BlockchainTrainingPlan,
    release: Path,
    manifest: BlockchainSecurityReleaseManifest,
    staging: Path,
    dependencies: dict[str, Any],
) -> BlockchainAdapterManifest:
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

    train_documents = _load_split_documents(release, "train")
    validation_documents = _load_split_documents(release, "validation")
    train_rows = _tokenize_documents(train_documents, tokenizer, config.max_sequence_length)
    validation_rows = _tokenize_documents(
        validation_documents,
        tokenizer,
        config.max_sequence_length,
    )
    if not train_rows:
        raise ValueError("blockchain train split produced no token chunks")
    if not validation_rows:
        raise ValueError("blockchain validation split produced no token chunks")

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
        "save_total_limit": 2,
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
    shutil.rmtree(staging / "checkpoints", ignore_errors=True)
    files = sorted(
        str(path.relative_to(staging))
        for path in adapter_path.rglob("*")
        if path.is_file()
    )
    if not files:
        raise ValueError("blockchain training produced no adapter files")

    adapter_manifest = BlockchainAdapterManifest(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        release_manifest_file_digest=plan.release_manifest_file_digest,
        release_manifest_digest=plan.release_manifest_digest,
        source_corpus_digest=plan.source_corpus_digest,
        benchmark_suite_digest=plan.benchmark_suite_digest,
        train_split_digest=plan.train_split_digest,
        validation_split_digest=plan.validation_split_digest,
        held_out_test_split_digest=plan.held_out_test_split_digest,
        training_config_digest=plan.training_config_digest,
        train_documents=plan.train_documents,
        validation_documents=plan.validation_documents,
        held_out_test_documents=plan.held_out_test_documents,
        held_out_test_sources=plan.held_out_test_sources,
        train_chunks=len(train_rows),
        validation_chunks=len(validation_rows),
        adapter_digest=_directory_digest(staging, files),
        adapter_files=files,
        output_dir=config.output_dir,
        train_loss=_metric_float(train_result.metrics.get("train_loss")),
        eval_loss=_metric_float(eval_metrics.get("eval_loss")),
        held_out_test_consumed=False,
    )
    (staging / "blockchain-adapter-manifest.json").write_text(
        json.dumps(adapter_manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return adapter_manifest


def _verify_expected_lineage(
    config: BlockchainTrainingConfig,
    manifest: BlockchainSecurityReleaseManifest,
) -> None:
    if manifest.source_corpus_digest != config.expected_source_corpus_digest:
        raise ValueError("blockchain release source corpus digest does not match config pin")
    if manifest.benchmark_suite_digest != config.expected_benchmark_suite_digest:
        raise ValueError("blockchain release benchmark suite digest does not match config pin")


def _load_split_documents(release: Path, split: str) -> list[BlockchainSecurityDocument]:
    if split not in {"train", "validation"}:
        raise ValueError("training executor may load only train or validation splits")
    path = release / f"{split}.jsonl"
    return load_blockchain_security_documents(path)


def _render_training_text(document: BlockchainSecurityDocument) -> str:
    chains = ",".join(item.value for item in document.chain_families)
    threats = ",".join(item.value for item in document.threat_domains)
    return (
        "Koschei Sentinel blockchain-security knowledge document.\n"
        f"Chain families: {chains}\n"
        f"Threat domains: {threats}\n"
        f"Source class: {document.source_class.value}\n"
        "Security material:\n"
        f"{document.text}"
    )


def _tokenize_documents(
    documents: list[BlockchainSecurityDocument],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    for document in documents:
        encoded = tokenizer(_render_training_text(document), add_special_tokens=False)
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
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _model_digest(model: StrictModel) -> str:
    return hashlib.sha256(
        json.dumps(
            model.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
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


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name != "posix":
        raise ValueError("atomic blockchain adapter publication requires POSIX")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise ValueError("atomic no-replace publication requires renameat2") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"blockchain-training output already exists: {destination}")
        if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise ValueError("atomic no-replace publication unsupported by filesystem")
        raise OSError(error_number, os.strerror(error_number), destination)
    _fsync_directory(destination.parent)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
