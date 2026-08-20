from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_export_verify import (
    CyberSFTExportVerification,
    verify_cyber_sft_export,
)
from koschei_sentinel.cyber_sft_run_attestation import (
    CyberSFTRunAttestation,
    _attestation_digest,
    _config_sha256,
)
from koschei_sentinel.cyber_sft_text_trainer import (
    _assert_requested_model_dtype,
    _assert_text_only_model,
    _cuda_preflight,
)
from koschei_sentinel.cyber_sft_trainer import CyberSFTAdapterManifest
from koschei_sentinel.cyber_sft_training import (
    SYSTEM_PROMPT,
    CyberSFTConfig,
    load_cyber_sft_config,
)
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutInferenceCase,
    GoldHoldoutInferenceManifest,
    GoldHoldoutPredictedStep,
    GoldHoldoutPrediction,
    build_gold_holdout_prediction,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldHoldoutGenerationPolicy(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-generation-policy.v1"] = (
        "sentinel.gold-holdout-generation-policy.v1"
    )
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)
    do_sample: Literal[False] = False
    num_beams: Literal[1] = 1


class GoldHoldoutInferencePlan(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-inference-plan.v2"] = (
        "sentinel.gold-holdout-inference-plan.v2"
    )
    model_ref: str = Field(min_length=3, max_length=512)
    model_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    run_id: str
    training_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_attestation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_export_verification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_count: int = Field(gt=0)
    inputs_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    inference_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    generation_policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_key_isolated: Literal[True] = True
    deterministic_generation: Literal[True] = True
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GoldHoldoutInferenceFailure(StrictModel):
    case_id: str
    scenario_id: str
    input_context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    failure_type: Literal["PROMPT_TOO_LONG", "GENERATION_PARSE_ERROR"]
    detail: str = Field(min_length=1, max_length=4000)


class GoldHoldoutInferenceRunReceipt(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-inference-run-receipt.v2"] = (
        "sentinel.gold-holdout-inference-run-receipt.v2"
    )
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_id: str
    model_ref: str
    model_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    training_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_attestation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_export_verification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failed_case_ids: list[str]
    predictions_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    failures_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cuda_device_index: int = Field(ge=0)
    cuda_device_name: str
    runtime_versions: dict[str, str]
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return _sha256_text(canonical_json(unsigned))


def _policy_sha256(policy: GoldHoldoutGenerationPolicy) -> str:
    return _sha256_text(canonical_json(policy.model_dump(mode="json")))


def _export_verification_sha256(report: CyberSFTExportVerification) -> str:
    return _sha256_text(canonical_json(report.model_dump(mode="json")))


def _allowed_input_context(case: GoldHoldoutInferenceCase) -> None:
    expected_keys = {"scenario_id", "critical_entity_ids", "graph_snapshots"}
    observed_keys = set(case.input_context)
    if observed_keys != expected_keys:
        raise ValueError(
            "Gold HOLDOUT inference input contains fields outside the "
            "answer-key-isolated contract: "
            + ", ".join(sorted(observed_keys - expected_keys))
        )
    if case.input_context.get("scenario_id") != case.scenario_id:
        raise ValueError("Gold HOLDOUT inference case scenario_id differs from input_context")


def _load_inference_pack(
    inference_pack_dir: str | Path,
) -> tuple[list[GoldHoldoutInferenceCase], GoldHoldoutInferenceManifest, bytes]:
    root = Path(inference_pack_dir)
    manifest_path = root / "manifest.json"
    inputs_path = root / "inputs.jsonl"
    if not manifest_path.is_file() or not inputs_path.is_file():
        raise ValueError("Gold HOLDOUT inference pack requires manifest.json and inputs.jsonl")
    manifest_raw = manifest_path.read_bytes()
    manifest = GoldHoldoutInferenceManifest.model_validate_json(manifest_raw)
    if not manifest.answer_key_excluded:
        raise ValueError("Gold HOLDOUT inference manifest does not prove answer-key exclusion")
    raw_inputs = inputs_path.read_bytes()
    if _sha256_bytes(raw_inputs) != manifest.inputs_sha256:
        raise ValueError("Gold HOLDOUT inference inputs SHA does not match manifest")
    try:
        lines = raw_inputs.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Gold HOLDOUT inference inputs are not valid UTF-8") from exc

    rows: list[GoldHoldoutInferenceCase] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = GoldHoldoutInferenceCase.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(
                f"invalid Gold HOLDOUT inference case at line {line_number}"
            ) from exc
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate Gold HOLDOUT inference case: {case.case_id}")
        seen_case_ids.add(case.case_id)
        expected_context_sha = _sha256_text(canonical_json(case.input_context))
        if expected_context_sha != case.input_context_sha256:
            raise ValueError(f"Gold HOLDOUT input-context digest mismatch: {case.case_id}")
        _allowed_input_context(case)
        rows.append(case)

    rows = sorted(rows, key=lambda row: row.case_id)
    if not rows:
        raise ValueError("Gold HOLDOUT inference pack is empty")
    if len(rows) != manifest.case_count:
        raise ValueError("Gold HOLDOUT inference case count differs from manifest")
    if [row.case_id for row in rows] != manifest.case_ids:
        raise ValueError("Gold HOLDOUT inference case IDs differ from manifest")
    return rows, manifest, manifest_raw


def _load_candidate_identity(
    *,
    candidate_export_dir: str | Path,
) -> tuple[
    CyberSFTConfig,
    CyberSFTAdapterManifest,
    Path,
    CyberSFTRunAttestation,
    CyberSFTExportVerification,
]:
    export_root = Path(candidate_export_dir).resolve()
    export_verification = verify_cyber_sft_export(export_root)
    if not export_verification.valid:
        detail = "; ".join(export_verification.violations[:5])
        raise ValueError(
            "Gold HOLDOUT inference requires a valid Cyber SFT candidate export"
            + (f": {detail}" if detail else "")
        )

    config = load_cyber_sft_config(export_root / "training-config.json")
    attestation = CyberSFTRunAttestation.model_validate_json(
        (export_root / "run-attestation.json").read_bytes()
    )
    attestation_payload = attestation.model_dump(mode="json")
    if _attestation_digest(attestation_payload) != attestation.attestation_sha256:
        raise ValueError(
            "Gold HOLDOUT candidate run attestation self-hash does not verify"
        )
    if _config_sha256(config) != attestation.config_sha256:
        raise ValueError("Gold HOLDOUT training config SHA differs from run attestation")

    run_path = export_root / "run"
    manifest = CyberSFTAdapterManifest.model_validate_json(
        (run_path / "adapter-manifest.json").read_bytes()
    )
    checks = (
        ("run_id", manifest.run_id, config.run_id),
        ("base_model", manifest.base_model, config.base_model),
        ("base_revision", manifest.base_revision, config.base_revision),
        ("attestation run_id", attestation.run_id, config.run_id),
        ("attestation base_model", attestation.base_model, config.base_model),
        (
            "attestation base_revision",
            attestation.base_revision,
            config.base_revision,
        ),
        (
            "attestation adapter_digest",
            attestation.adapter_digest,
            manifest.adapter_digest,
        ),
    )
    for label, observed, expected in checks:
        if observed != expected:
            raise ValueError(f"Gold HOLDOUT candidate export mismatch: {label}")
    if manifest.corpus_promotion_eligible is not True:
        raise ValueError(
            "Gold HOLDOUT promotion evaluation requires a promotion-eligible adapter"
        )
    if attestation.promotion_eligible is not True or attestation.smoke_only:
        raise ValueError(
            "Gold HOLDOUT promotion evaluation requires a promotion-eligible "
            "run attestation"
        )
    adapter_path = run_path / "adapter"
    if not adapter_path.is_dir():
        raise ValueError(
            "verified Cyber SFT candidate export is missing its adapter directory"
        )
    return config, manifest, adapter_path, attestation, export_verification


def build_gold_holdout_inference_plan(
    *,
    inference_pack_dir: str | Path,
    candidate_export_dir: str | Path,
    model_ref: str,
    generation_policy: GoldHoldoutGenerationPolicy | None = None,
) -> GoldHoldoutInferencePlan:
    rows, inference_manifest, manifest_raw = _load_inference_pack(inference_pack_dir)
    config, adapter_manifest, _adapter_path, attestation, export_verification = (
        _load_candidate_identity(candidate_export_dir=candidate_export_dir)
    )
    policy = generation_policy or GoldHoldoutGenerationPolicy()
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-inference-plan.v2",
        "model_ref": model_ref,
        "model_revision": adapter_manifest.adapter_digest,
        "adapter_digest": adapter_manifest.adapter_digest,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "run_id": config.run_id,
        "training_config_sha256": attestation.config_sha256,
        "run_attestation_sha256": attestation.attestation_sha256,
        "candidate_export_verification_sha256": _export_verification_sha256(
            export_verification
        ),
        "case_count": len(rows),
        "inputs_sha256": inference_manifest.inputs_sha256,
        "inference_manifest_sha256": _sha256_bytes(manifest_raw),
        "generation_policy_sha256": _policy_sha256(policy),
        "answer_key_isolated": True,
        "deterministic_generation": True,
    }
    payload["plan_sha256"] = _digest_without(payload, "plan_sha256")
    return GoldHoldoutInferencePlan.model_validate(payload)


def _prompt_messages(case: GoldHoldoutInferenceCase) -> list[dict[str, str]]:
    user_payload = {
        "task": "derive an evidence-grounded defensive plan",
        "scenario_id": case.scenario_id,
        "critical_entity_ids": case.input_context["critical_entity_ids"],
        "graph_snapshots": case.input_context["graph_snapshots"],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(user_payload)},
    ]


def _prediction_from_generated_text(
    *,
    case: GoldHoldoutInferenceCase,
    generated_text: str,
    model_ref: str,
    adapter_digest: str,
) -> GoldHoldoutPrediction:
    try:
        payload = json.loads(generated_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("model output is not exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("model output must be one JSON object")
    if set(payload) != {"interpretation", "defense_sequence"}:
        raise ValueError(
            "model output must contain exactly interpretation and defense_sequence"
        )
    interpretation = payload.get("interpretation")
    sequence_payload = payload.get("defense_sequence")
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ValueError("model output interpretation must be non-empty text")
    if not isinstance(sequence_payload, list) or not sequence_payload:
        raise ValueError("model output defense_sequence must be a non-empty list")
    try:
        steps = [GoldHoldoutPredictedStep.model_validate(row) for row in sequence_payload]
    except ValueError as exc:
        raise ValueError(
            "model output defense_sequence does not match the prediction schema"
        ) from exc
    return build_gold_holdout_prediction(
        inference_case=case,
        model_ref=model_ref,
        model_revision=adapter_digest,
        adapter_digest=adapter_digest,
        interpretation=interpretation.strip(),
        defense_sequence=steps,
    )


def _serialize_predictions(rows: list[GoldHoldoutPrediction]) -> str:
    return "".join(
        json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in sorted(rows, key=lambda item: item.case_id)
    )


def _serialize_failures(rows: list[GoldHoldoutInferenceFailure]) -> str:
    return "".join(
        json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in sorted(rows, key=lambda item: item.case_id)
    )


def _runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def _prepare_prompt(
    case: GoldHoldoutInferenceCase,
    tokenizer: Any,
    max_sequence_length: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, int, GoldHoldoutInferenceFailure | None]:
    prompt = tokenizer.apply_chat_template(
        _prompt_messages(case),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    required_tokens = prompt_tokens + max_new_tokens
    if required_tokens <= max_sequence_length:
        return encoded, prompt_tokens, None
    failure = GoldHoldoutInferenceFailure(
        case_id=case.case_id,
        scenario_id=case.scenario_id,
        input_context_sha256=case.input_context_sha256,
        failure_type="PROMPT_TOO_LONG",
        detail=(
            f"prompt requires {prompt_tokens} tokens plus generation reserve "
            f"max_new_tokens={max_new_tokens}, total={required_tokens}, above "
            f"max_sequence_length={max_sequence_length}"
        ),
    )
    return None, prompt_tokens, failure


def execute_gold_holdout_inference(
    *,
    inference_pack_dir: str | Path,
    candidate_export_dir: str | Path,
    model_ref: str,
    output_dir: str | Path,
    generation_policy: GoldHoldoutGenerationPolicy | None = None,
) -> GoldHoldoutInferenceRunReceipt:
    rows, _inference_manifest, _manifest_raw = _load_inference_pack(inference_pack_dir)
    config, adapter_manifest, adapter_path, attestation, export_verification = (
        _load_candidate_identity(candidate_export_dir=candidate_export_dir)
    )
    policy = generation_policy or GoldHoldoutGenerationPolicy()
    plan = build_gold_holdout_inference_plan(
        inference_pack_dir=inference_pack_dir,
        candidate_export_dir=candidate_export_dir,
        model_ref=model_ref,
        generation_policy=policy,
    )

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Gold HOLDOUT inference output already exists: {destination}"
        )

    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Gold HOLDOUT inference requires project training dependencies"
        ) from exc

    device_index = _cuda_preflight(config, torch)
    device = torch.device(f"cuda:{device_index}")
    dtype = (
        torch.bfloat16
        if config.quantization.compute_dtype == "bfloat16"
        else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    failures: list[GoldHoldoutInferenceFailure] = []
    prepared: list[tuple[GoldHoldoutInferenceCase, dict[str, Any], int]] = []
    for case in rows:
        encoded, prompt_tokens, failure = _prepare_prompt(
            case,
            tokenizer,
            config.max_sequence_length,
            policy.max_new_tokens,
        )
        if failure is not None:
            failures.append(failure)
            continue
        if encoded is None:
            raise RuntimeError(
                "Gold HOLDOUT prompt preparation returned no inputs or failure"
            )
        prepared.append((case, encoded, prompt_tokens))

    predictions: list[GoldHoldoutPrediction] = []
    if prepared:
        quantization = BitsAndBytesConfig(
            load_in_4bit=config.quantization.bits == 4,
            load_in_8bit=config.quantization.bits == 8,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            revision=config.base_revision,
            trust_remote_code=False,
            dtype=dtype,
            quantization_config=quantization,
            device_map={"": device_index},
        )
        _assert_text_only_model(base_model)
        _assert_requested_model_dtype(base_model, dtype, torch)
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False,
        )
        model.eval()

        for case, encoded, prompt_tokens in prepared:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=policy.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, prompt_tokens:]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            try:
                prediction = _prediction_from_generated_text(
                    case=case,
                    generated_text=generated_text,
                    model_ref=model_ref,
                    adapter_digest=adapter_manifest.adapter_digest,
                )
            except ValueError as exc:
                failures.append(
                    GoldHoldoutInferenceFailure(
                        case_id=case.case_id,
                        scenario_id=case.scenario_id,
                        input_context_sha256=case.input_context_sha256,
                        failure_type="GENERATION_PARSE_ERROR",
                        detail=str(exc),
                    )
                )
            else:
                predictions.append(prediction)

    prediction_payload = _serialize_predictions(predictions)
    failure_payload = _serialize_failures(failures)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "predictions.jsonl").write_text(
        prediction_payload,
        encoding="utf-8",
    )
    (destination / "failures.jsonl").write_text(
        failure_payload,
        encoding="utf-8",
    )
    (destination / "plan.json").write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "generation-policy.json").write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "training-config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "run-attestation.json").write_text(
        json.dumps(attestation.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (destination / "candidate-export-verification.json").write_text(
        json.dumps(
            export_verification.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    device_name = str(torch.cuda.get_device_properties(device_index).name)
    receipt_payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-inference-run-receipt.v2",
        "plan_sha256": plan.plan_sha256,
        "run_id": config.run_id,
        "model_ref": model_ref,
        "model_revision": adapter_manifest.adapter_digest,
        "adapter_digest": adapter_manifest.adapter_digest,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "training_config_sha256": attestation.config_sha256,
        "run_attestation_sha256": attestation.attestation_sha256,
        "candidate_export_verification_sha256": _export_verification_sha256(
            export_verification
        ),
        "case_count": len(rows),
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "failed_case_ids": sorted(row.case_id for row in failures),
        "predictions_sha256": _sha256_text(prediction_payload),
        "failures_sha256": _sha256_text(failure_payload),
        "cuda_device_index": device_index,
        "cuda_device_name": device_name,
        "runtime_versions": _runtime_versions(),
    }
    receipt_payload["receipt_sha256"] = _digest_without(
        receipt_payload,
        "receipt_sha256",
    )
    receipt = GoldHoldoutInferenceRunReceipt.model_validate(receipt_payload)
    (destination / "receipt.json").write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
