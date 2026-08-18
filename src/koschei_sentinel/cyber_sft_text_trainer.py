from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from koschei_sentinel.cyber_sft_trainer import (
    _DENSE_QLORA_OPTIMIZER,
    CyberSFTAdapterManifest,
    _build_training_receipt,
    _cuda_preflight,
    _directory_digest,
    _fast_kernel_warnings,
    _metrics_payload,
    _render_supervision,
    _targets_digest,
)
from koschei_sentinel.cyber_sft_training import (
    CyberExecutionProfile,
    CyberSFTConfig,
    CyberSFTPlan,
    _combined_promotion_eligibility,
    load_cyber_sft_examples,
    load_cyber_sft_validation_examples,
    split_cyber_sft_examples,
)
from koschei_sentinel.training import canonical_json, resolve_under_root

_EXPECTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
_RESUME_CHECKPOINT_STEPS = 2
_RESUME_CHECKPOINT_LIMIT = 2
_SUPPORTED_LORA_TARGET_TYPES = {
    "torch.nn.modules.linear.Linear",
    "bitsandbytes.nn.modules.Linear4bit",
    "bitsandbytes.nn.modules.Linear8bitLt",
}


def _load_text_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cyber SFT text dependencies are missing; install the project with .[training]"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _text_lora_targets(model: Any, suffixes: list[str]) -> list[str]:
    suffix_set = set(suffixes)
    targets: list[str] = []
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in suffix_set:
            continue
        lowered = name.lower()
        if "visual" in lowered or "vision" in lowered:
            raise RuntimeError(
                "Cyber SFT text executor resolved a vision module as a LoRA target"
            )
        if name.startswith("model.layers.") or ".language_model.layers." in name:
            targets.append(name)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError(
            "no Qwen3.5 text-backbone LoRA targets matched the configured suffixes"
        )
    return targets


def _module_type_name(module: Any) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__name__}"


def _assert_lora_target_module_types(
    model: Any,
    targets: list[str],
) -> dict[str, int]:
    modules = dict(model.named_modules())
    counts: dict[str, int] = {}
    unsupported: list[str] = []
    missing: list[str] = []
    for target in targets:
        module = modules.get(target)
        if module is None:
            missing.append(target)
            continue
        module_type = _module_type_name(module)
        counts[module_type] = counts.get(module_type, 0) + 1
        if module_type not in _SUPPORTED_LORA_TARGET_TYPES:
            unsupported.append(f"{target}={module_type}")
    if missing:
        raise RuntimeError(
            "resolved LoRA targets disappeared before PEFT wrapping: "
            + ", ".join(missing[:8])
        )
    if unsupported:
        raise RuntimeError(
            "Qwen3.5 LoRA target uses an unsupported projection module type; "
            "expected PyTorch Linear or bitsandbytes Linear4bit/Linear8bitLt. "
            "First mismatches: "
            + ", ".join(unsupported[:8])
        )
    return dict(sorted(counts.items()))


def _assert_text_only_model(model: Any) -> None:
    class_name = model.__class__.__name__
    if class_name != _EXPECTED_MODEL_CLASS:
        raise RuntimeError(
            "Cyber SFT expected the official Qwen3.5 text-only causal-LM class, "
            f"got {class_name}"
        )
    forbidden = sorted(
        name
        for name, _module in model.named_modules()
        if "visual" in name.lower() or "vision" in name.lower()
    )
    if forbidden:
        raise RuntimeError(
            "Cyber SFT text executor loaded vision modules unexpectedly: "
            + ", ".join(forbidden[:8])
        )


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("torch.")


def _assert_requested_model_dtype(
    model: Any,
    expected_dtype: Any,
    torch: Any,
) -> list[str]:
    expected_name = _dtype_name(expected_dtype)
    observed = sorted(
        {
            _dtype_name(parameter.dtype)
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if not observed:
        raise RuntimeError("Qwen3.5 text model exposes no floating parameters after load")
    if expected_name not in observed:
        raise RuntimeError(
            "Qwen3.5 loaded model does not expose the explicitly requested low-precision dtype: "
            f"{expected_name}; observed={observed}"
        )

    competing = torch.bfloat16 if expected_dtype == torch.float16 else torch.float16
    competing_name = _dtype_name(competing)
    mismatched = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and _dtype_name(parameter.dtype) == competing_name
    ]
    if mismatched:
        raise RuntimeError(
            "Qwen3.5 composite-to-text load leaked the competing low-precision dtype; "
            "refusing training. First mismatches: "
            + ", ".join(mismatched[:8])
        )
    return observed


def _config_digest(config: CyberSFTConfig) -> str:
    return hashlib.sha256(
        canonical_json(config.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _resume_directory(root: Path, config: CyberSFTConfig) -> Path:
    output = resolve_under_root(root, config.output_dir)
    return output.parent / ".resume" / output.name


def _resume_binding(config: CyberSFTConfig, plan: CyberSFTPlan) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-sft-resume-binding.v1",
        "run_id": config.run_id,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "corpus_examples_sha256": plan.corpus_examples_sha256,
        "corpus_manifest_sha256": plan.corpus_manifest_sha256,
        "config_sha256": _config_digest(config),
    }
    if plan.explicit_validation:
        payload.update(
            {
                "explicit_validation": True,
                "validation_corpus_examples_sha256": plan.validation_corpus_examples_sha256,
                "validation_corpus_manifest_sha256": plan.validation_corpus_manifest_sha256,
            }
        )
    return payload


def _prepare_resume_directory(
    root: Path,
    config: CyberSFTConfig,
    plan: CyberSFTPlan,
) -> Path:
    resume = _resume_directory(root, config)
    binding_path = resume / "resume-binding.json"
    expected = _resume_binding(config, plan)
    if resume.exists():
        if not binding_path.is_file():
            if any(resume.iterdir()):
                raise RuntimeError(
                    "Cyber SFT resume directory exists without a binding receipt; "
                    "refusing stale checkpoint reuse"
                )
        else:
            try:
                observed = json.loads(binding_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Cyber SFT resume binding is invalid JSON") from exc
            if observed != expected:
                raise RuntimeError(
                    "Cyber SFT resume binding differs from current model/corpus/config; "
                    "refusing checkpoint reuse"
                )
    resume.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resume


def _last_checkpoint(checkpoints: Path) -> Path | None:
    if not checkpoints.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    return max(candidates, default=(0, None), key=lambda row: row[0])[1]


def _assigned_training_rows(
    config: CyberSFTConfig,
    plan: CyberSFTPlan,
    rows: list[Any],
    training_promotion_eligible: bool | None,
    *,
    root: Path,
) -> tuple[list[Any], list[Any]]:
    explicit_requested = config.validation_corpus_dir is not None
    if plan.explicit_validation != explicit_requested:
        raise ValueError("Cyber SFT plan explicit validation mode differs from config")

    if not explicit_requested:
        if training_promotion_eligible != plan.corpus_promotion_eligible:
            raise ValueError("Cyber SFT corpus promotion eligibility changed after plan creation")
        training_rows, validation_rows = split_cyber_sft_examples(
            rows,
            validation_ratio=config.validation_ratio,
            seed=config.seed,
        )
    else:
        loaded = load_cyber_sft_validation_examples(config, root=root)
        if loaded is None:
            raise RuntimeError("explicit validation corpus was not loaded")
        validation_rows, validation_examples_sha, validation_manifest_sha, validation_promotion = (
            loaded
        )
        if validation_examples_sha != plan.validation_corpus_examples_sha256:
            raise ValueError("Cyber SFT validation examples changed after plan creation")
        if validation_manifest_sha != plan.validation_corpus_manifest_sha256:
            raise ValueError("Cyber SFT validation manifest changed after plan creation")
        combined = _combined_promotion_eligibility(
            training_promotion_eligible,
            validation_promotion,
        )
        if combined != plan.corpus_promotion_eligible:
            raise ValueError(
                "Cyber SFT explicit TRAIN/VALIDATION promotion eligibility changed after planning"
            )
        training_ids = {str(row.example_id) for row in rows}
        validation_ids = {str(row.example_id) for row in validation_rows}
        if training_ids & validation_ids:
            raise ValueError("Cyber SFT explicit TRAIN/VALIDATION example IDs overlap at execution")
        training_scenario_ids = {
            str(row.scenario_id) for row in rows if hasattr(row, "scenario_id")
        }
        validation_scenario_ids = {
            str(row.scenario_id)
            for row in validation_rows
            if hasattr(row, "scenario_id")
        }
        if training_scenario_ids & validation_scenario_ids:
            raise ValueError("Cyber SFT explicit TRAIN/VALIDATION scenario IDs overlap at execution")
        training_rows = rows

    if len(training_rows) != plan.training_examples:
        raise ValueError("Cyber SFT training example count changed after plan creation")
    if len(validation_rows) != plan.validation_examples:
        raise ValueError("Cyber SFT validation example count changed after plan creation")
    return training_rows, validation_rows


def execute_cyber_sft_text(
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
            "started by the dense text-only single-GPU QLoRA trainer"
        )

    dependencies = _load_text_dependencies()
    torch = dependencies["torch"]
    device_index = _cuda_preflight(config, torch)
    root_path = Path(root).resolve()
    output = resolve_under_root(root_path, config.output_dir)
    if output.exists():
        raise FileExistsError(f"Cyber SFT output already exists: {config.output_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_dir = _prepare_resume_directory(root_path, config, plan)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest = _train_text(
            config,
            plan,
            root_path,
            staging,
            dependencies,
            device_index=device_index,
            resume_dir=resume_dir,
        )
        os.replace(staging, output)
        shutil.rmtree(resume_dir, ignore_errors=True)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _train_text(
    config: CyberSFTConfig,
    plan: CyberSFTPlan,
    root: Path,
    staging: Path,
    dependencies: dict[str, Any],
    *,
    device_index: int,
    resume_dir: Path,
) -> CyberSFTAdapterManifest:
    torch = dependencies["torch"]
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows, examples_sha, manifest_sha, promotion_eligible = load_cyber_sft_examples(
        config,
        root=root,
    )
    if examples_sha != plan.corpus_examples_sha256:
        raise ValueError("Cyber SFT corpus changed after plan creation")
    if manifest_sha != plan.corpus_manifest_sha256:
        raise ValueError("Cyber SFT corpus manifest changed after plan creation")

    training_rows, validation_rows = _assigned_training_rows(
        config,
        plan,
        rows,
        promotion_eligible,
        root=root,
    )
    train_features = [
        _render_supervision(row, tokenizer, config.max_sequence_length)
        for row in training_rows
    ]
    eval_features = [
        _render_supervision(row, tokenizer, config.max_sequence_length)
        for row in validation_rows
    ]
    dataset_type = dependencies["Dataset"]
    train_dataset = dataset_type.from_list(train_features)
    eval_dataset = dataset_type.from_list(eval_features) if eval_features else None

    torch.cuda.reset_peak_memory_stats(device_index)
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
        dtype=dtype,
        quantization_config=quantization,
        device_map={"": device_index},
    )
    _assert_text_only_model(model)
    observed_floating_dtypes = _assert_requested_model_dtype(model, dtype, torch)
    loaded_model_class = model.__class__.__name__
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = dependencies["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )

    lora_target_module_types: dict[str, int] = {}
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
        targets = _text_lora_targets(model, config.lora.target_suffixes)
        lora_target_module_types = _assert_lora_target_module_types(model, targets)
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

    checkpoint_root = resume_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _last_checkpoint(checkpoint_root)

    arguments_type = dependencies["TrainingArguments"]
    values: dict[str, Any] = {
        "output_dir": str(checkpoint_root),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "steps",
        "save_steps": _RESUME_CHECKPOINT_STEPS,
        "save_total_limit": _RESUME_CHECKPOINT_LIMIT,
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.seed,
        "bf16": config.quantization.compute_dtype == "bfloat16",
        "fp16": config.quantization.compute_dtype == "float16",
        "remove_unused_columns": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "optim": _DENSE_QLORA_OPTIMIZER,
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
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    train_metrics = _metrics_payload(dict(train_result.metrics))
    eval_metrics = (
        _metrics_payload(dict(trainer.evaluate()))
        if eval_dataset is not None
        else {}
    )

    adapter_dir = staging / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
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
        corpus_promotion_eligible=plan.corpus_promotion_eligible,
        input_adapter_dir=config.input_adapter_dir,
        adapter_digest=_directory_digest(staging, files),
        adapter_files=files,
        trainable_target_module_count=len(targets),
        trainable_target_modules_sha256=_targets_digest(targets),
        training_examples=len(training_rows),
        validation_examples=len(validation_rows),
        gradient_checkpointing=config.gradient_checkpointing,
        optimizer=_DENSE_QLORA_OPTIMIZER,
        output_dir=config.output_dir,
    )
    (staging / "adapter-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = _build_training_receipt(
        manifest=manifest,
        trainer=trainer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        torch=torch,
        device_index=device_index,
    )
    (staging / "training-receipt.json").write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "model-runtime.json").write_text(
        json.dumps(
            {
                "loader": "AutoModelForCausalLM",
                "model_class": loaded_model_class,
                "expected_model_class": _EXPECTED_MODEL_CLASS,
                "text_only": True,
                "requested_compute_dtype": config.quantization.compute_dtype,
                "observed_floating_dtypes_before_kbit_prepare": observed_floating_dtypes,
                "lora_target_module_types_before_peft": lora_target_module_types,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (staging / "resume-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "sentinel.cyber-sft-resume-runtime.v1",
                "resumed": resume_checkpoint is not None,
                "resume_checkpoint": (
                    resume_checkpoint.name if resume_checkpoint is not None else None
                ),
                "checkpoint_every_optimizer_steps": _RESUME_CHECKPOINT_STEPS,
                "checkpoint_retention": _RESUME_CHECKPOINT_LIMIT,
                "resume_binding": _resume_binding(config, plan),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    warnings = _fast_kernel_warnings()
    if warnings:
        (staging / "runtime-warnings.json").write_text(
            json.dumps({"warnings": warnings}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest
