from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_training import (
    CyberExecutionProfile,
    CyberSFTConfig,
    CyberSFTPlan,
    cyber_sft_messages,
    load_cyber_sft_examples,
    split_cyber_sft_examples,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import resolve_under_root


class CyberSFTAdapterManifest(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-adapter-manifest.v1"] = (
        "sentinel.cyber-sft-adapter-manifest.v1"
    )
    run_id: str
    stage: str
    execution_profile: CyberExecutionProfile
    base_model: str
    base_revision: str
    corpus_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_adapter_dir: str | None
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_files: list[str]
    trainable_target_module_count: int = Field(gt=0)
    trainable_target_modules_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_examples: int = Field(gt=0)
    validation_examples: int = Field(ge=0)
    gradient_checkpointing: bool
    output_dir: str


def _load_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cyber SFT dependencies are missing; install the project with .[training]"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForMultimodalLM": AutoModelForMultimodalLM,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _cuda_preflight(config: CyberSFTConfig, torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Cyber SFT execution requires CUDA")
    total = sum(
        torch.cuda.get_device_properties(index).total_memory
        for index in range(torch.cuda.device_count())
    )
    total_gb = total / (1024**3)
    if total_gb + 1e-9 < config.minimum_cuda_memory_gb:
        raise RuntimeError(
            "visible CUDA memory below config minimum: "
            f"{total_gb:.1f} GiB < {config.minimum_cuda_memory_gb:.1f} GiB"
        )
    if (
        config.quantization.compute_dtype == "bfloat16"
        and hasattr(torch.cuda, "is_bf16_supported")
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Cyber SFT config requests bfloat16 on unsupported CUDA hardware")


def _language_lora_targets(model: Any, suffixes: list[str]) -> list[str]:
    suffix_set = set(suffixes)
    targets = sorted(
        name
        for name, _module in model.named_modules()
        if "language_model" in name and name.rsplit(".", 1)[-1] in suffix_set
    )
    if not targets:
        raise RuntimeError(
            "no Qwen3.5 language_model LoRA targets matched the configured suffixes"
        )
    if any("visual" in name or "vision" in name for name in targets):
        raise RuntimeError("Cyber SFT resolved a vision module as a LoRA target")
    return targets


def _render_supervision(
    row: StrictModel,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
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
    encoded = tokenizer(full, add_special_tokens=False, truncation=False)
    full_ids = encoded["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Cyber SFT chat-template prompt is not a prefix of supervision")
    if len(full_ids) > max_length:
        raise ValueError(
            f"Cyber SFT example {row.example_id} requires {len(full_ids)} tokens, "
            f"above max_sequence_length={max_length}; explicit corpus compaction required"
        )
    if len(prompt_ids) >= len(full_ids):
        raise ValueError(f"Cyber SFT example {row.example_id} has no supervised answer tokens")
    labels = list(full_ids)
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": encoded.get("attention_mask", [1] * len(full_ids)),
        "labels": labels,
    }


def _directory_digest(root: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _targets_digest(targets: list[str]) -> str:
    payload = json.dumps(targets, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fast_kernel_warnings() -> list[str]:
    warnings: list[str] = []
    if importlib.util.find_spec("causal_conv1d") is None:
        warnings.append("causal_conv1d is unavailable; Qwen3.5 DeltaNet may use slower kernels")
    if importlib.util.find_spec("fla") is None:
        warnings.append("fla is unavailable; Qwen3.5 DeltaNet may use slower kernels")
    return warnings


def execute_cyber_sft(
    config: CyberSFTConfig,
    plan: CyberSFTPlan,
    *,
    root: str | Path = ".",
) -> CyberSFTAdapterManifest:
    if config.run_id != plan.run_id or config.stage is not plan.stage:
        raise ValueError("Cyber SFT plan does not match the supplied config")
    if config.execution_profile is not plan.execution_profile:
        raise ValueError("Cyber SFT plan execution profile differs from config")
    if config.base_model != plan.base_model or config.base_revision != plan.base_revision:
        raise ValueError("Cyber SFT plan base model pin differs from config")
    if (
        config.execution_profile is not CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA
        or not plan.executable_with_current_trainer
    ):
        raise RuntimeError(
            "this Cyber SFT plan requires the distributed MoE executor and cannot be "
            "started by the dense single-GPU QLoRA trainer"
        )

    dependencies = _load_dependencies()
    torch = dependencies["torch"]
    _cuda_preflight(config, torch)
    root_path = Path(root).resolve()
    output = resolve_under_root(root_path, config.output_dir)
    if output.exists():
        raise FileExistsError(f"Cyber SFT output already exists: {config.output_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest = _train(config, plan, root_path, staging, dependencies)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _train(
    config: CyberSFTConfig,
    plan: CyberSFTPlan,
    root: Path,
    staging: Path,
    dependencies: dict[str, Any],
) -> CyberSFTAdapterManifest:
    torch = dependencies["torch"]
    processor = dependencies["AutoProcessor"].from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
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
    model = dependencies["AutoModelForMultimodalLM"].from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map="auto",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "text_config") and hasattr(
        model.config.text_config,
        "output_router_logits",
    ):
        model.config.text_config.output_router_logits = config.enable_router_aux_loss
    model = dependencies["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )

    if config.input_adapter_dir is not None:
        adapter_path = resolve_under_root(root, config.input_adapter_dir)
        model = dependencies["PeftModel"].from_pretrained(
            model,
            str(adapter_path),
            is_trainable=True,
        )
        targets = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        )
        if not targets:
            raise RuntimeError("input Cyber SFT adapter contains no trainable LoRA parameters")
    else:
        targets = _language_lora_targets(model, config.lora.target_suffixes)
        model = dependencies["get_peft_model"](
            model,
            dependencies["LoraConfig"](
                task_type="CAUSAL_LM",
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=targets,
            ),
        )

    rows, examples_sha, manifest_sha = load_cyber_sft_examples(config, root=root)
    if examples_sha != plan.corpus_examples_sha256:
        raise ValueError("Cyber SFT corpus changed after plan creation")
    if manifest_sha != plan.corpus_manifest_sha256:
        raise ValueError("Cyber SFT corpus manifest changed after plan creation")
    training_rows, validation_rows = split_cyber_sft_examples(
        rows,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    dataset_type = dependencies["Dataset"]
    train_dataset = dataset_type.from_list(
        [
            _render_supervision(row, tokenizer, config.max_sequence_length)
            for row in training_rows
        ]
    )
    eval_dataset = None
    if validation_rows:
        eval_dataset = dataset_type.from_list(
            [
                _render_supervision(row, tokenizer, config.max_sequence_length)
                for row in validation_rows
            ]
        )

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
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(arguments_type.__init__).parameters
        else "evaluation_strategy"
    )
    values[strategy_key] = "epoch" if eval_dataset is not None else "no"
    trainer = dependencies["Trainer"](
        model=model,
        args=arguments_type(**values),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=dependencies["DataCollatorForSeq2Seq"](
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    trainer.train()

    adapter_dir = staging / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    files = sorted(
        str(path.relative_to(staging))
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    manifest = CyberSFTAdapterManifest(
        run_id=config.run_id,
        stage=config.stage.value,
        execution_profile=config.execution_profile,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256=examples_sha,
        corpus_manifest_sha256=manifest_sha,
        input_adapter_dir=config.input_adapter_dir,
        adapter_digest=_directory_digest(staging, files),
        adapter_files=files,
        trainable_target_module_count=len(targets),
        trainable_target_modules_sha256=_targets_digest(targets),
        training_examples=len(training_rows),
        validation_examples=len(validation_rows),
        gradient_checkpointing=config.gradient_checkpointing,
        output_dir=config.output_dir,
    )
    (staging / "adapter-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    warnings = _fast_kernel_warnings()
    if warnings:
        (staging / "runtime-warnings.json").write_text(
            json.dumps({"warnings": warnings}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest
