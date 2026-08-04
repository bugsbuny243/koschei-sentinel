from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from koschei_sentinel.models import EvidenceConfidence, EvidenceItem, SecurityCase
from koschei_sentinel.training import (
    AdapterManifest,
    TrainingConfig,
    TrainingPlan,
    load_release_examples,
    model_digest,
    resolve_under_root,
    supervised_messages,
)

_CONFIDENCE_ORDER = {
    EvidenceConfidence.VERIFIED: 0,
    EvidenceConfidence.INFERRED: 1,
    EvidenceConfidence.UNVERIFIED: 2,
}
_COMPACTION_STEPS = (
    (3, 1024, 8, 16, 1024),
    (2, 512, 4, 8, 512),
    (1, 256, 2, 4, 256),
    (1, 128, 0, 2, 160),
    (1, 64, 0, 0, 96),
)


def execute_training(
    config: TrainingConfig,
    plan: TrainingPlan,
    *,
    root: str | Path = ".",
) -> AdapterManifest:
    if plan.training_config_digest != model_digest(config):
        raise ValueError("training plan does not match the supplied config")
    dependencies = _load_dependencies()
    root_path = Path(root).resolve()
    output_path = resolve_under_root(root_path, config.output_dir)
    if output_path.exists():
        raise FileExistsError(f"training output already exists: {config.output_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        manifest = _train(config, plan, root_path, staging, dependencies)
        os.replace(staging, output_path)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _train(
    config: TrainingConfig,
    plan: TrainingPlan,
    root: Path,
    staging: Path,
    dependencies: dict[str, Any],
) -> AdapterManifest:
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
    train_rows, _ = load_release_examples(root, plan.splits["train"].path)
    validation_rows, _ = load_release_examples(root, plan.splits["validation"].path)
    dataset_type = dependencies["Dataset"]
    train_dataset = dataset_type.from_list(
        [_tokenize(item, tokenizer, config.max_sequence_length) for item in train_rows]
    )
    eval_dataset = None
    if validation_rows:
        eval_dataset = dataset_type.from_list(
            [
                _tokenize(item, tokenizer, config.max_sequence_length)
                for item in validation_rows
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
    adapter_path = staging / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    files = sorted(
        str(path.relative_to(staging))
        for path in adapter_path.rglob("*")
        if path.is_file()
    )
    manifest = AdapterManifest(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        dataset_manifest_digest=plan.dataset_manifest_digest,
        training_config_digest=plan.training_config_digest,
        adapter_digest=_directory_digest(staging, files),
        adapter_files=files,
        output_dir=config.output_dir,
    )
    (staging / "adapter-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _tokenize(example: Any, tokenizer: Any, max_length: int) -> dict[str, Any]:
    messages = _fit_supervised_messages(example, tokenizer, max_length)
    _, full, prompt_ids, full_ids = _render_supervision(messages, tokenizer)
    if len(full_ids) > max_length:
        raise ValueError("internal sequence budgeting error")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full supervision")

    encoded = tokenizer(
        full,
        add_special_tokens=False,
        truncation=False,
    )
    labels = list(encoded["input_ids"])
    masked = len(prompt_ids)
    if masked >= len(labels):
        raise ValueError("max_sequence_length truncates the entire supervised answer")
    labels[:masked] = [-100] * masked
    encoded["labels"] = labels
    return encoded


def _fit_supervised_messages(
    example: Any,
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, str]]:
    messages = supervised_messages(example)
    if _messages_fit(messages, tokenizer, max_length):
        return messages

    for (
        evidence_per_rule,
        statement_chars,
        attribute_limit,
        limitation_limit,
        summary_chars,
    ) in _COMPACTION_STEPS:
        compact_case = _compact_training_case(
            example.case,
            evidence_per_rule=evidence_per_rule,
            statement_chars=statement_chars,
            attribute_limit=attribute_limit,
            limitation_limit=limitation_limit,
            summary_chars=summary_chars,
        )
        compact_example = example.model_copy(update={"case": compact_case})
        messages = supervised_messages(compact_example)
        if _messages_fit(messages, tokenizer, max_length):
            return messages

    raise ValueError(
        "max_sequence_length cannot fit evidence-grounded compact supervision"
    )


def _messages_fit(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> bool:
    _, _, prompt_ids, full_ids = _render_supervision(messages, tokenizer)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full supervision")
    return len(full_ids) <= max_length and len(prompt_ids) < len(full_ids)


def _render_supervision(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> tuple[str, str, list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    return prompt, full, prompt_ids, full_ids


def _compact_training_case(
    case: SecurityCase,
    *,
    evidence_per_rule: int,
    statement_chars: int,
    attribute_limit: int,
    limitation_limit: int,
    summary_chars: int,
) -> SecurityCase:
    selected = _select_training_evidence(case, evidence_per_rule=evidence_per_rule)
    compact_evidence = [
        item.model_copy(
            update={
                "statement": _shorten(item.statement, statement_chars),
                "attributes": _compact_attributes(item, attribute_limit),
            }
        )
        for item in selected
    ]
    limitations = [
        _shorten(value, statement_chars)
        for value in case.limitations[:limitation_limit]
    ]
    if limitation_limit and len(case.limitations) > limitation_limit:
        limitations.append("Additional limitations omitted from compact training view.")

    signed_verdict = case.signed_verdict.model_copy(
        update={"summary": _shorten(case.signed_verdict.summary, summary_chars)}
    )
    return case.model_copy(
        update={
            "signed_verdict": signed_verdict,
            "evidence": compact_evidence,
            "limitations": limitations,
        }
    )


def _select_training_evidence(
    case: SecurityCase,
    *,
    evidence_per_rule: int,
) -> list[EvidenceItem]:
    indexed = list(enumerate(case.evidence))
    selected_ids: set[str] = set()
    selected: list[EvidenceItem] = []

    for rule_id in case.signed_verdict.triggered_rules:
        candidates = [
            (index, item)
            for index, item in indexed
            if rule_id in item.rule_ids
        ]
        candidates.sort(
            key=lambda pair: (
                _CONFIDENCE_ORDER[pair[1].confidence],
                pair[0],
            )
        )
        for _, item in candidates[:evidence_per_rule]:
            if item.evidence_id not in selected_ids:
                selected_ids.add(item.evidence_id)
                selected.append(item)

    if not selected:
        fallback = sorted(
            indexed,
            key=lambda pair: (
                _CONFIDENCE_ORDER[pair[1].confidence],
                pair[0],
            ),
        )
        selected = [item for _, item in fallback[: max(1, evidence_per_rule)]]

    return selected


def _compact_attributes(
    item: EvidenceItem,
    limit: int,
) -> dict[str, str | int | float | bool | None]:
    if limit == 0:
        return {}
    compact: dict[str, str | int | float | bool | None] = {}
    for key in sorted(item.attributes)[:limit]:
        value = item.attributes[key]
        compact[key] = _shorten(value, 160) if isinstance(value, str) else value
    return compact


def _shorten(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def _directory_digest(root: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
            DataCollatorForSeq2Seq,
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
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }
