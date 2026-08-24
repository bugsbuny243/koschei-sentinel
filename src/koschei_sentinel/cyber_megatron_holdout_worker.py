from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_megatron_candidate import (
    CyberMegatronCandidateManifest,
    _checkpoint_inventory,
    _resolve_no_symlinks,
)
from koschei_sentinel.cyber_megatron_holdout import (
    CyberMegatronHoldoutPlan,
    verify_cyber_megatron_holdout_plan,
)
from koschei_sentinel.cyber_megatron_training import (
    MS_SWIFT_VERSION,
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
)
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutInferenceCase,
    GoldHoldoutPredictedStep,
    GoldHoldoutPrediction,
    build_gold_holdout_prediction,
)
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    _load_inference_pack,
    _policy_sha256,
    _prompt_messages,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json, resolve_under_root

_DIGEST = r"^[a-f0-9]{64}$"
WORKER_PLAN_FILENAME = "worker-plan.json"
REQUESTS_FILENAME = "requests.jsonl"
REQUEST_BINDINGS_FILENAME = "request-bindings.json"
RAW_RESULTS_FILENAME = "swift-results.jsonl"
EXECUTION_STATE_FILENAME = "execution-state.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
FAILURES_FILENAME = "failures.jsonl"
RECEIPT_FILENAME = "receipt.json"
HOLDOUT_LAUNCH_APPROVAL_ENV = "KOSCHEI_397B_HOLDOUT_APPROVED"
HOLDOUT_LAUNCH_SESSION_ENV = "KOSCHEI_397B_HOLDOUT_SESSION"
HOLDOUT_LAUNCH_APPROVAL_VALUE = "YES_I_ACCEPT_GPU_COST"
RAY_ADDRESS_ENV = "RAY_ADDRESS"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _sha256_canonical(payload: object) -> str:
    return _sha256_text(canonical_json(payload))


def _json_text(model: StrictModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _jsonl_text(rows: list[StrictModel]) -> str:
    return "".join(
        json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )


def _request_jsonl_text(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc


def _load_holdout_plan(path: Path) -> tuple[CyberMegatronHoldoutPlan, bytes]:
    raw = _read_regular_bytes(path, "397B HOLDOUT plan")
    try:
        return CyberMegatronHoldoutPlan.model_validate_json(raw), raw
    except ValueError as exc:
        raise ValueError("397B HOLDOUT plan cannot be parsed") from exc


def _load_candidate(path: Path) -> CyberMegatronCandidateManifest:
    raw = _read_regular_bytes(path, "397B candidate manifest")
    try:
        return CyberMegatronCandidateManifest.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B candidate manifest cannot be parsed") from exc


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"required runtime package is not installed: {name}") from exc


class CyberMegatronHoldoutWorkerProfile(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-worker-profile.v2"] = (
        "sentinel.cyber-megatron-holdout-worker-profile.v2"
    )
    infer_backend: Literal["vllm"] = "vllm"
    distributed_executor_backend: Literal["ray"] = "ray"
    ms_swift_version: Literal["4.5.2"] = MS_SWIFT_VERSION
    vllm_version: str = Field(default="0.17.0", pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    nodes: int = Field(default=4, ge=2, le=64)
    gpus_per_node: int = Field(default=8, ge=1, le=16)
    tensor_parallel_size: int = Field(default=8, ge=1, le=64)
    pipeline_parallel_size: int = Field(default=4, ge=1, le=64)
    enable_expert_parallel: Literal[True] = True
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=0.98)
    max_model_len: int = Field(default=8192, ge=1024, le=262_144)
    max_num_seqs: int = Field(default=8, ge=1, le=1024)
    write_batch_size: Literal[-1] = -1
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def topology_is_explicit_multi_node(self) -> CyberMegatronHoldoutWorkerProfile:
        if self.tensor_parallel_size != self.gpus_per_node:
            raise ValueError("397B HOLDOUT TP must equal GPUs per node")
        if self.pipeline_parallel_size != self.nodes:
            raise ValueError("397B HOLDOUT PP must equal the Ray node count")
        if self.model_parallel_world_size != self.total_gpus:
            raise ValueError("397B HOLDOUT TP * PP must equal total allocated GPUs")
        return self

    @property
    def model_parallel_world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node


class CyberMegatronHoldoutRequestBinding(StrictModel):
    case_id: str
    scenario_id: str
    input_context_sha256: str = Field(pattern=_DIGEST)
    messages_sha256: str = Field(pattern=_DIGEST)


class CyberMegatronHoldoutRequestManifest(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-requests.v1"] = (
        "sentinel.cyber-megatron-holdout-requests.v1"
    )
    case_count: int = Field(gt=0)
    case_ids: list[str] = Field(min_length=1)
    requests_sha256: str = Field(pattern=_DIGEST)
    bindings: list[CyberMegatronHoldoutRequestBinding] = Field(min_length=1)
    bindings_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def manifest_contract_verifies(self) -> CyberMegatronHoldoutRequestManifest:
        if self.case_ids != sorted(self.case_ids):
            raise ValueError("397B worker request case_ids must be sorted")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("397B worker request case_ids must be unique")
        if self.case_count != len(self.case_ids) or self.case_count != len(self.bindings):
            raise ValueError("397B worker request count differs from bindings")
        binding_ids = [row.case_id for row in self.bindings]
        if binding_ids != self.case_ids:
            raise ValueError("397B worker bindings must match sorted case_ids")
        expected = _sha256_canonical(
            [row.model_dump(mode="json") for row in self.bindings]
        )
        if expected != self.bindings_sha256:
            raise ValueError("397B worker bindings_sha256 does not verify")
        return self


class CyberMegatronHoldoutWorkerPlan(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-worker-plan.v2"] = (
        "sentinel.cyber-megatron-holdout-worker-plan.v2"
    )
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    run_id: str
    holdout_plan_file_sha256: str = Field(pattern=_DIGEST)
    holdout_plan_sha256: str = Field(pattern=_DIGEST)
    candidate_sha256: str = Field(pattern=_DIGEST)
    checkpoint_tree_sha256: str = Field(pattern=_DIGEST)
    checkpoint_path: str = Field(min_length=1, max_length=4096)
    checkpoint_format: Literal["swift-merged-hf"] = "swift-merged-hf"
    request_count: int = Field(gt=0)
    case_ids_sha256: str = Field(pattern=_DIGEST)
    requests_sha256: str = Field(pattern=_DIGEST)
    request_bindings_sha256: str = Field(pattern=_DIGEST)
    profile: CyberMegatronHoldoutWorkerProfile
    generation_policy: GoldHoldoutGenerationPolicy
    generation_policy_sha256: str = Field(pattern=_DIGEST)
    answer_key_isolated: Literal[True] = True
    deterministic_generation: Literal[True] = True
    raw_result_format: Literal["ms-swift-infer-jsonl"] = "ms-swift-infer-jsonl"
    launch_approval_env: Literal["KOSCHEI_397B_HOLDOUT_APPROVED"] = (
        HOLDOUT_LAUNCH_APPROVAL_ENV
    )
    launch_session_env: Literal["KOSCHEI_397B_HOLDOUT_SESSION"] = (
        HOLDOUT_LAUNCH_SESSION_ENV
    )
    command: list[str] = Field(min_length=2, max_length=160)
    command_sha256: str = Field(pattern=_DIGEST)
    plan_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def self_hash_verifies(self) -> CyberMegatronHoldoutWorkerPlan:
        if _policy_sha256(self.generation_policy) != self.generation_policy_sha256:
            raise ValueError("397B HOLDOUT worker generation policy SHA does not verify")
        if _sha256_canonical(self.command) != self.command_sha256:
            raise ValueError("397B HOLDOUT worker command SHA does not verify")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("plan_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("397B HOLDOUT worker plan_sha256 does not verify")
        return self


class CyberMegatronHoldoutExecutionState(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-execution-state.v1"] = (
        "sentinel.cyber-megatron-holdout-execution-state.v1"
    )
    worker_plan_sha256: str = Field(pattern=_DIGEST)
    command_sha256: str = Field(pattern=_DIGEST)
    checkpoint_tree_sha256: str = Field(pattern=_DIGEST)
    launch_session_sha256: str = Field(pattern=_DIGEST)
    ray_address_sha256: str = Field(pattern=_DIGEST)
    return_code: int = Field(ge=0, le=255)
    state: Literal["completed", "failed"]
    raw_results_sha256: str | None = Field(default=None, pattern=_DIGEST)
    runtime_versions: dict[str, str]
    state_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def execution_contract_verifies(self) -> CyberMegatronHoldoutExecutionState:
        if self.state == "completed" and (
            self.return_code != 0 or self.raw_results_sha256 is None
        ):
            raise ValueError("completed 397B HOLDOUT execution requires exit 0 and raw results")
        if self.state == "failed" and self.return_code == 0 and self.raw_results_sha256 is not None:
            raise ValueError("failed 397B HOLDOUT execution is inconsistent with exit 0 results")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("state_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("397B HOLDOUT execution state SHA does not verify")
        return self


class CyberMegatronHoldoutFailure(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-failure.v1"] = (
        "sentinel.cyber-megatron-holdout-failure.v1"
    )
    case_id: str
    scenario_id: str
    input_context_sha256: str = Field(pattern=_DIGEST)
    failure_type: Literal[
        "MISSING_RESULT",
        "DUPLICATE_RESULT",
        "GENERATION_PARSE_ERROR",
    ]
    detail: str = Field(min_length=1, max_length=4000)


class CyberMegatronHoldoutRunReceipt(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-run-receipt.v2"] = (
        "sentinel.cyber-megatron-holdout-run-receipt.v2"
    )
    worker_plan_sha256: str = Field(pattern=_DIGEST)
    execution_state_sha256: str = Field(pattern=_DIGEST)
    holdout_plan_sha256: str = Field(pattern=_DIGEST)
    candidate_sha256: str = Field(pattern=_DIGEST)
    checkpoint_tree_sha256: str = Field(pattern=_DIGEST)
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    request_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failed_case_ids: list[str]
    raw_results_sha256: str = Field(pattern=_DIGEST)
    predictions_sha256: str = Field(pattern=_DIGEST)
    failures_sha256: str = Field(pattern=_DIGEST)
    runtime_versions: dict[str, str]
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_self_hash_verifies(self) -> CyberMegatronHoldoutRunReceipt:
        if self.prediction_count + self.failure_count != self.request_count:
            raise ValueError("397B HOLDOUT receipt does not account for every request")
        if self.failed_case_ids != sorted(self.failed_case_ids):
            raise ValueError("397B HOLDOUT failed_case_ids must be sorted")
        if len(self.failed_case_ids) != self.failure_count:
            raise ValueError("397B HOLDOUT failed_case_ids count differs from failure_count")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("receipt_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("397B HOLDOUT receipt_sha256 does not verify")
        return self


class CyberMegatronHoldoutOutputVerification(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-output-verification.v2"] = (
        "sentinel.cyber-megatron-holdout-output-verification.v2"
    )
    valid: bool
    request_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    source_plan_verified: bool
    worker_plan_verified: bool
    request_binding_verified: bool
    execution_verified: bool
    raw_results_replayed: bool
    receipt_verified: bool
    complete_case_accounting: bool
    violations: list[str]
    verification_sha256: str = Field(pattern=_DIGEST)


def _worker_plan_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("plan_sha256", None)
    return _sha256_canonical(unsigned)


def _execution_state_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("state_sha256", None)
    return _sha256_canonical(unsigned)


def _receipt_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return _sha256_canonical(unsigned)


def _verification_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("verification_sha256", None)
    return _sha256_canonical(unsigned)


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    return _sha256_canonical(messages)


def _build_request_material(
    cases: list[GoldHoldoutInferenceCase],
) -> tuple[str, CyberMegatronHoldoutRequestManifest]:
    request_rows: list[dict[str, object]] = []
    bindings: list[CyberMegatronHoldoutRequestBinding] = []
    for case in cases:
        messages = _prompt_messages(case)
        request_rows.append({"messages": messages})
        bindings.append(
            CyberMegatronHoldoutRequestBinding(
                case_id=case.case_id,
                scenario_id=case.scenario_id,
                input_context_sha256=case.input_context_sha256,
                messages_sha256=_messages_sha256(messages),
            )
        )
    request_text = _request_jsonl_text(request_rows)
    bindings_sha = _sha256_canonical(
        [row.model_dump(mode="json") for row in bindings]
    )
    manifest = CyberMegatronHoldoutRequestManifest(
        case_count=len(cases),
        case_ids=[case.case_id for case in cases],
        requests_sha256=_sha256_bytes(request_text.encode("utf-8")),
        bindings=bindings,
        bindings_sha256=bindings_sha,
    )
    return request_text, manifest


def _validate_merged_inference_checkpoint(
    checkpoint_root: Path,
    *,
    expected_tree_sha256: str | None = None,
) -> str:
    if not checkpoint_root.name.startswith("checkpoint-") or not checkpoint_root.name.endswith(
        "-merged"
    ):
        raise ValueError(
            "397B HOLDOUT inference requires SWIFT checkpoint-*-merged output"
        )
    config_path = checkpoint_root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("397B merged inference checkpoint is missing config.json")
    root_safetensors = sorted(checkpoint_root.glob("*.safetensors"))
    index_path = checkpoint_root / "model.safetensors.index.json"
    if not root_safetensors and not index_path.is_file():
        raise ValueError(
            "397B merged inference checkpoint has no root safetensors weights or index"
        )
    _files, _total, tree_sha = _checkpoint_inventory(checkpoint_root)
    if expected_tree_sha256 is not None and tree_sha != expected_tree_sha256:
        raise ValueError("397B merged checkpoint tree SHA differs from approved candidate")
    return tree_sha


def _swift_command(
    *,
    checkpoint_path: str,
    requests_path: str,
    raw_results_path: str,
    generation_policy: GoldHoldoutGenerationPolicy,
    profile: CyberMegatronHoldoutWorkerProfile,
) -> list[str]:
    engine_kwargs = json.dumps(
        {"distributed_executor_backend": profile.distributed_executor_backend},
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        "swift",
        "infer",
        "--model",
        checkpoint_path,
        "--infer_backend",
        "vllm",
        "--val_dataset",
        requests_path,
        "--result_path",
        raw_results_path,
        "--stream",
        "false",
        "--enable_thinking",
        "false",
        "--temperature",
        "0",
        "--do_sample",
        "false",
        "--num_beams",
        str(generation_policy.num_beams),
        "--max_new_tokens",
        str(generation_policy.max_new_tokens),
        "--seed",
        str(profile.seed),
        "--write_batch_size",
        str(profile.write_batch_size),
        "--val_dataset_shuffle",
        "false",
        "--load_data_args",
        "false",
        "--vllm_tensor_parallel_size",
        str(profile.tensor_parallel_size),
        "--vllm_pipeline_parallel_size",
        str(profile.pipeline_parallel_size),
        "--vllm_enable_expert_parallel",
        "true",
        "--vllm_gpu_memory_utilization",
        str(profile.gpu_memory_utilization),
        "--vllm_max_model_len",
        str(profile.max_model_len),
        "--vllm_max_num_seqs",
        str(profile.max_num_seqs),
        "--vllm_disable_custom_all_reduce",
        "true",
        "--vllm_enable_prefix_caching",
        "false",
        "--vllm_use_async_engine",
        "false",
        "--vllm_engine_kwargs",
        engine_kwargs,
    ]


def prepare_cyber_megatron_holdout_worker(
    *,
    holdout_plan_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_config_path: str | Path,
    candidate_training_plan_path: str | Path,
    checkpoint_dir: str | Path,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    output_dir: str | Path,
    profile: CyberMegatronHoldoutWorkerProfile | None = None,
    minimum_case_count: int = 50,
    root: str | Path = ".",
) -> CyberMegatronHoldoutWorkerPlan:
    root_path = Path(root).resolve()
    plan_path = resolve_under_root(root_path, str(holdout_plan_path))
    observed_plan, plan_raw = _load_holdout_plan(plan_path)
    verification = verify_cyber_megatron_holdout_plan(
        root=root_path,
        plan_path=plan_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_config_path=candidate_config_path,
        candidate_training_plan_path=candidate_training_plan_path,
        checkpoint_dir=checkpoint_dir,
        inference_pack=inference_pack,
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
        reviewer_trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
        minimum_case_count=minimum_case_count,
    )
    if not verification.valid or verification.plan is None:
        detail = "; ".join(verification.violations[:5])
        raise ValueError(
            "397B worker requires a freshly verified HOLDOUT plan"
            + (f": {detail}" if detail else "")
        )
    if verification.plan != observed_plan:
        raise ValueError("397B worker HOLDOUT plan differs from fresh verification")

    candidate_path = resolve_under_root(root_path, str(candidate_manifest_path))
    candidate = _load_candidate(candidate_path)
    if candidate.candidate_sha256 != observed_plan.candidate_sha256:
        raise ValueError("397B worker candidate differs from HOLDOUT plan")
    if candidate.checkpoint_tree_sha256 != observed_plan.checkpoint_tree_sha256:
        raise ValueError("397B worker checkpoint identity differs from HOLDOUT plan")

    checkpoint_path = _resolve_no_symlinks(root_path, checkpoint_dir, "397B merged checkpoint")
    _validate_merged_inference_checkpoint(
        checkpoint_path,
        expected_tree_sha256=candidate.checkpoint_tree_sha256,
    )

    cases, _manifest, _manifest_raw = _load_inference_pack(inference_pack)
    if [case.case_id for case in cases] != observed_plan.case_ids:
        raise ValueError("397B worker inference pack case IDs differ from HOLDOUT plan")
    request_text, request_manifest = _build_request_material(cases)

    selected_profile = profile or CyberMegatronHoldoutWorkerProfile()
    destination = resolve_under_root(root_path, str(output_dir))
    if destination.exists():
        raise FileExistsError(f"397B HOLDOUT worker output already exists: {output_dir}")
    destination.mkdir(parents=True)
    requests_path = destination / REQUESTS_FILENAME
    bindings_path = destination / REQUEST_BINDINGS_FILENAME
    worker_plan_path = destination / WORKER_PLAN_FILENAME
    raw_results_path = destination / RAW_RESULTS_FILENAME
    atomic_write(requests_path, request_text)
    atomic_write(bindings_path, _json_text(request_manifest))

    checkpoint_relative = checkpoint_path.relative_to(root_path).as_posix()
    requests_relative = requests_path.relative_to(root_path).as_posix()
    results_relative = raw_results_path.relative_to(root_path).as_posix()
    command = _swift_command(
        checkpoint_path=checkpoint_relative,
        requests_path=requests_relative,
        raw_results_path=results_relative,
        generation_policy=observed_plan.generation_policy,
        profile=selected_profile,
    )
    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-holdout-worker-plan.v2",
        "model": QWEN35_397B_MODEL,
        "model_revision": QWEN35_397B_REVISION,
        "run_id": observed_plan.run_id,
        "holdout_plan_file_sha256": _sha256_bytes(plan_raw),
        "holdout_plan_sha256": observed_plan.plan_sha256,
        "candidate_sha256": candidate.candidate_sha256,
        "checkpoint_tree_sha256": candidate.checkpoint_tree_sha256,
        "checkpoint_path": checkpoint_relative,
        "checkpoint_format": "swift-merged-hf",
        "request_count": len(cases),
        "case_ids_sha256": observed_plan.case_ids_sha256,
        "requests_sha256": request_manifest.requests_sha256,
        "request_bindings_sha256": request_manifest.bindings_sha256,
        "profile": selected_profile.model_dump(mode="json"),
        "generation_policy": observed_plan.generation_policy.model_dump(mode="json"),
        "generation_policy_sha256": observed_plan.generation_policy_sha256,
        "answer_key_isolated": True,
        "deterministic_generation": True,
        "raw_result_format": "ms-swift-infer-jsonl",
        "launch_approval_env": HOLDOUT_LAUNCH_APPROVAL_ENV,
        "launch_session_env": HOLDOUT_LAUNCH_SESSION_ENV,
        "command": command,
        "command_sha256": _sha256_canonical(command),
    }
    worker_plan = CyberMegatronHoldoutWorkerPlan(
        **unsigned,
        plan_sha256=_worker_plan_digest(unsigned),
    )
    atomic_write(worker_plan_path, _json_text(worker_plan))
    return worker_plan


def _load_request_manifest(worker_root: Path) -> CyberMegatronHoldoutRequestManifest:
    raw = _read_regular_bytes(
        worker_root / REQUEST_BINDINGS_FILENAME,
        "397B HOLDOUT request bindings",
    )
    try:
        manifest = CyberMegatronHoldoutRequestManifest.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B HOLDOUT request bindings cannot be parsed") from exc
    if raw != _json_text(manifest).encode("utf-8"):
        raise ValueError("397B HOLDOUT request bindings are not canonical")
    return manifest


def _load_worker_plan(worker_root: Path) -> CyberMegatronHoldoutWorkerPlan:
    raw = _read_regular_bytes(worker_root / WORKER_PLAN_FILENAME, "397B HOLDOUT worker plan")
    try:
        plan = CyberMegatronHoldoutWorkerPlan.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B HOLDOUT worker plan cannot be parsed") from exc
    if raw != _json_text(plan).encode("utf-8"):
        raise ValueError("397B HOLDOUT worker plan is not canonical")
    return plan


def _preexecution_inventory(worker_root: Path) -> None:
    expected = {
        WORKER_PLAN_FILENAME,
        REQUESTS_FILENAME,
        REQUEST_BINDINGS_FILENAME,
    }
    observed: set[str] = set()
    for path in worker_root.rglob("*"):
        relative = path.relative_to(worker_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"397B HOLDOUT pre-execution output contains symlink: {relative}")
        if path.is_file():
            observed.add(relative)
    if observed != expected:
        raise ValueError(
            "397B HOLDOUT pre-execution file set differs from sealed contract: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )


def _verify_worker_preexecution(
    *,
    worker_root: Path,
    root_path: Path,
) -> CyberMegatronHoldoutWorkerPlan:
    if worker_root.is_symlink() or not worker_root.is_dir():
        raise ValueError("397B HOLDOUT worker directory is missing or symlinked")
    _preexecution_inventory(worker_root)
    plan = _load_worker_plan(worker_root)
    request_manifest = _load_request_manifest(worker_root)
    requests_raw = _read_regular_bytes(worker_root / REQUESTS_FILENAME, "397B HOLDOUT requests")
    if _sha256_bytes(requests_raw) != plan.requests_sha256:
        raise ValueError("397B HOLDOUT requests SHA differs from worker plan")
    if request_manifest.requests_sha256 != plan.requests_sha256:
        raise ValueError("397B HOLDOUT request manifest differs from worker plan")
    if request_manifest.bindings_sha256 != plan.request_bindings_sha256:
        raise ValueError("397B HOLDOUT request bindings differ from worker plan")
    if request_manifest.case_count != plan.request_count:
        raise ValueError("397B HOLDOUT request count differs from worker plan")

    checkpoint_path = _resolve_no_symlinks(
        root_path,
        plan.checkpoint_path,
        "397B merged checkpoint",
    )
    _validate_merged_inference_checkpoint(
        checkpoint_path,
        expected_tree_sha256=plan.checkpoint_tree_sha256,
    )
    expected_command = _swift_command(
        checkpoint_path=plan.checkpoint_path,
        requests_path=(worker_root / REQUESTS_FILENAME).relative_to(root_path).as_posix(),
        raw_results_path=(worker_root / RAW_RESULTS_FILENAME).relative_to(root_path).as_posix(),
        generation_policy=plan.generation_policy,
        profile=plan.profile,
    )
    if plan.command != expected_command:
        raise ValueError("397B HOLDOUT worker command differs from allowed command contract")
    if _sha256_canonical(expected_command) != plan.command_sha256:
        raise ValueError("397B HOLDOUT worker command SHA differs from allowed command")
    return plan


def execute_cyber_megatron_holdout_worker(
    *,
    worker_dir: str | Path,
    root: str | Path = ".",
) -> int:
    root_path = Path(root).resolve()
    worker_root = resolve_under_root(root_path, str(worker_dir))
    plan = _verify_worker_preexecution(worker_root=worker_root, root_path=root_path)

    if os.environ.get(HOLDOUT_LAUNCH_APPROVAL_ENV) != HOLDOUT_LAUNCH_APPROVAL_VALUE:
        raise PermissionError(
            f"paid 397B HOLDOUT requires {HOLDOUT_LAUNCH_APPROVAL_ENV}="
            f"{HOLDOUT_LAUNCH_APPROVAL_VALUE}"
        )
    session = os.environ.get(HOLDOUT_LAUNCH_SESSION_ENV, "")
    if session != plan.plan_sha256:
        raise PermissionError(
            f"paid 397B HOLDOUT requires {HOLDOUT_LAUNCH_SESSION_ENV} "
            "to equal the exact worker plan SHA256"
        )
    ray_address = os.environ.get(RAY_ADDRESS_ENV, "").strip()
    if not ray_address:
        raise PermissionError("multi-node 397B HOLDOUT requires RAY_ADDRESS")
    if shutil.which("swift") is None:
        raise ValueError("swift executable is not available on the worker")
    if shutil.which("ray") is None:
        raise ValueError("ray executable is not available on the worker")

    runtime_versions = {
        "ms-swift": _package_version("ms-swift"),
        "vllm": _package_version("vllm"),
        "ray": _package_version("ray"),
    }
    if runtime_versions["ms-swift"] != plan.profile.ms_swift_version:
        raise ValueError("installed ms-swift version differs from worker plan")
    if runtime_versions["vllm"] != plan.profile.vllm_version:
        raise ValueError("installed vLLM version differs from worker plan")

    raw_result_path = worker_root / RAW_RESULTS_FILENAME
    execution_path = worker_root / EXECUTION_STATE_FILENAME
    if raw_result_path.exists():
        raise FileExistsError("397B HOLDOUT raw result already exists; SWIFT would append")
    if execution_path.exists():
        raise FileExistsError("397B HOLDOUT execution state already exists")

    completed = subprocess.run(
        plan.command,
        cwd=root_path,
        check=False,
        env=os.environ.copy(),
    )
    return_code = int(completed.returncode)
    if return_code < 0 or return_code > 255:
        raise ValueError("397B HOLDOUT subprocess returned an invalid exit code")
    raw_sha: str | None = None
    if raw_result_path.is_symlink():
        raise ValueError("397B HOLDOUT raw result must not be a symlink")
    if raw_result_path.is_file():
        raw_sha = _sha256_bytes(raw_result_path.read_bytes())
    state_name: Literal["completed", "failed"] = (
        "completed" if return_code == 0 and raw_sha is not None else "failed"
    )
    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-holdout-execution-state.v1",
        "worker_plan_sha256": plan.plan_sha256,
        "command_sha256": plan.command_sha256,
        "checkpoint_tree_sha256": plan.checkpoint_tree_sha256,
        "launch_session_sha256": _sha256_text(session),
        "ray_address_sha256": _sha256_text(ray_address),
        "return_code": return_code,
        "state": state_name,
        "raw_results_sha256": raw_sha,
        "runtime_versions": runtime_versions,
    }
    state = CyberMegatronHoldoutExecutionState(
        **unsigned,
        state_sha256=_execution_state_digest(unsigned),
    )
    atomic_write(execution_path, _json_text(state))
    return return_code


def _parse_model_response(
    *,
    case: GoldHoldoutInferenceCase,
    generated_text: str,
    checkpoint_tree_sha256: str,
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
        raise ValueError("model output defense_sequence does not match schema") from exc
    return build_gold_holdout_prediction(
        inference_case=case,
        model_ref=QWEN35_397B_MODEL,
        model_revision=QWEN35_397B_REVISION,
        adapter_digest=checkpoint_tree_sha256,
        interpretation=interpretation,
        defense_sequence=steps,
    )


def _replay_raw_results(
    *,
    raw_results: bytes,
    cases: list[GoldHoldoutInferenceCase],
    request_manifest: CyberMegatronHoldoutRequestManifest,
    checkpoint_tree_sha256: str,
) -> tuple[list[GoldHoldoutPrediction], list[CyberMegatronHoldoutFailure]]:
    case_by_id = {case.case_id: case for case in cases}
    binding_by_messages = {
        binding.messages_sha256: binding for binding in request_manifest.bindings
    }
    seen: set[str] = set()
    predictions: dict[str, GoldHoldoutPrediction] = {}
    failures: dict[str, CyberMegatronHoldoutFailure] = {}
    try:
        lines = raw_results.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("397B SWIFT raw results are not valid UTF-8") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SWIFT result JSON at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"SWIFT result at line {line_number} is not an object")
        messages = payload.get("messages")
        response = payload.get("response")
        if not isinstance(messages, list) or not all(isinstance(row, dict) for row in messages):
            raise ValueError(f"SWIFT result at line {line_number} has no messages list")
        if not isinstance(response, str):
            raise ValueError(f"SWIFT result at line {line_number} has no text response")
        message_digest = _sha256_canonical(messages)
        binding = binding_by_messages.get(message_digest)
        if binding is None:
            raise ValueError(f"SWIFT result at line {line_number} is not a planned request")
        case = case_by_id[binding.case_id]
        if binding.case_id in seen:
            failures[binding.case_id] = CyberMegatronHoldoutFailure(
                case_id=case.case_id,
                scenario_id=case.scenario_id,
                input_context_sha256=case.input_context_sha256,
                failure_type="DUPLICATE_RESULT",
                detail="SWIFT returned the same planned request more than once",
            )
            predictions.pop(binding.case_id, None)
            continue
        seen.add(binding.case_id)
        try:
            predictions[binding.case_id] = _parse_model_response(
                case=case,
                generated_text=response,
                checkpoint_tree_sha256=checkpoint_tree_sha256,
            )
        except ValueError as exc:
            failures[binding.case_id] = CyberMegatronHoldoutFailure(
                case_id=case.case_id,
                scenario_id=case.scenario_id,
                input_context_sha256=case.input_context_sha256,
                failure_type="GENERATION_PARSE_ERROR",
                detail=str(exc),
            )

    for case in cases:
        if case.case_id not in seen:
            failures[case.case_id] = CyberMegatronHoldoutFailure(
                case_id=case.case_id,
                scenario_id=case.scenario_id,
                input_context_sha256=case.input_context_sha256,
                failure_type="MISSING_RESULT",
                detail="SWIFT produced no result for this planned HOLDOUT request",
            )
        if case.case_id in failures:
            predictions.pop(case.case_id, None)

    return (
        [predictions[case_id] for case_id in sorted(predictions)],
        [failures[case_id] for case_id in sorted(failures)],
    )


def _load_execution_state(worker_root: Path) -> CyberMegatronHoldoutExecutionState:
    raw = _read_regular_bytes(
        worker_root / EXECUTION_STATE_FILENAME,
        "397B HOLDOUT execution state",
    )
    try:
        state = CyberMegatronHoldoutExecutionState.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B HOLDOUT execution state cannot be parsed") from exc
    if raw != _json_text(state).encode("utf-8"):
        raise ValueError("397B HOLDOUT execution state is not canonical")
    return state


def finalize_cyber_megatron_holdout_worker(
    *,
    worker_dir: str | Path,
    inference_pack: str | Path,
    root: str | Path = ".",
) -> CyberMegatronHoldoutRunReceipt:
    root_path = Path(root).resolve()
    worker_root = resolve_under_root(root_path, str(worker_dir))
    plan = _load_worker_plan(worker_root)
    request_manifest = _load_request_manifest(worker_root)
    execution = _load_execution_state(worker_root)
    if execution.state != "completed" or execution.return_code != 0:
        raise ValueError("397B HOLDOUT cannot finalize a failed SWIFT execution")
    if execution.worker_plan_sha256 != plan.plan_sha256:
        raise ValueError("397B HOLDOUT execution state differs from worker plan")
    if execution.command_sha256 != plan.command_sha256:
        raise ValueError("397B HOLDOUT execution command differs from worker plan")
    if execution.checkpoint_tree_sha256 != plan.checkpoint_tree_sha256:
        raise ValueError("397B HOLDOUT execution checkpoint differs from worker plan")
    if execution.runtime_versions.get("ms-swift") != plan.profile.ms_swift_version:
        raise ValueError("397B HOLDOUT execution ms-swift version differs from worker plan")
    if execution.runtime_versions.get("vllm") != plan.profile.vllm_version:
        raise ValueError("397B HOLDOUT execution vLLM version differs from worker plan")

    requests_raw = _read_regular_bytes(worker_root / REQUESTS_FILENAME, "397B HOLDOUT requests")
    if _sha256_bytes(requests_raw) != plan.requests_sha256:
        raise ValueError("397B HOLDOUT requests SHA differs from worker plan")
    if request_manifest.requests_sha256 != plan.requests_sha256:
        raise ValueError("397B HOLDOUT request manifest differs from worker plan")
    if request_manifest.bindings_sha256 != plan.request_bindings_sha256:
        raise ValueError("397B HOLDOUT request bindings differ from worker plan")

    cases, _manifest, _manifest_raw = _load_inference_pack(inference_pack)
    if [case.case_id for case in cases] != request_manifest.case_ids:
        raise ValueError("397B HOLDOUT inference pack differs from worker requests")
    raw_results = _read_regular_bytes(
        worker_root / RAW_RESULTS_FILENAME,
        "397B SWIFT raw results",
    )
    if execution.raw_results_sha256 != _sha256_bytes(raw_results):
        raise ValueError("397B HOLDOUT raw results differ from execution state")
    predictions, failures = _replay_raw_results(
        raw_results=raw_results,
        cases=cases,
        request_manifest=request_manifest,
        checkpoint_tree_sha256=plan.checkpoint_tree_sha256,
    )
    prediction_text = _jsonl_text(predictions)
    failure_text = _jsonl_text(failures)
    prediction_path = worker_root / PREDICTIONS_FILENAME
    failure_path = worker_root / FAILURES_FILENAME
    receipt_path = worker_root / RECEIPT_FILENAME
    for path in (prediction_path, failure_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"397B HOLDOUT finalize output already exists: {path.name}")
    atomic_write(prediction_path, prediction_text)
    atomic_write(failure_path, failure_text)

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-holdout-run-receipt.v2",
        "worker_plan_sha256": plan.plan_sha256,
        "execution_state_sha256": execution.state_sha256,
        "holdout_plan_sha256": plan.holdout_plan_sha256,
        "candidate_sha256": plan.candidate_sha256,
        "checkpoint_tree_sha256": plan.checkpoint_tree_sha256,
        "model": plan.model,
        "model_revision": plan.model_revision,
        "request_count": plan.request_count,
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "failed_case_ids": [failure.case_id for failure in failures],
        "raw_results_sha256": _sha256_bytes(raw_results),
        "predictions_sha256": _sha256_bytes(prediction_text.encode("utf-8")),
        "failures_sha256": _sha256_bytes(failure_text.encode("utf-8")),
        "runtime_versions": execution.runtime_versions,
    }
    receipt = CyberMegatronHoldoutRunReceipt(
        **unsigned,
        receipt_sha256=_receipt_digest(unsigned),
    )
    atomic_write(receipt_path, _json_text(receipt))
    return receipt


def verify_cyber_megatron_holdout_output(
    *,
    worker_dir: str | Path,
    holdout_plan_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_config_path: str | Path,
    candidate_training_plan_path: str | Path,
    checkpoint_dir: str | Path,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    minimum_case_count: int = 50,
    root: str | Path = ".",
) -> CyberMegatronHoldoutOutputVerification:
    root_path = Path(root).resolve()
    violations: list[str] = []
    request_count = prediction_count = failure_count = 0
    source_plan_verified = False
    worker_plan_verified = False
    request_binding_verified = False
    execution_verified = False
    raw_results_replayed = False
    receipt_verified = False
    complete_case_accounting = False

    try:
        worker_root = resolve_under_root(root_path, str(worker_dir))
        if worker_root.is_symlink() or not worker_root.is_dir():
            raise ValueError("397B HOLDOUT worker output directory is missing or symlinked")
        expected_files = {
            WORKER_PLAN_FILENAME,
            REQUESTS_FILENAME,
            REQUEST_BINDINGS_FILENAME,
            RAW_RESULTS_FILENAME,
            EXECUTION_STATE_FILENAME,
            PREDICTIONS_FILENAME,
            FAILURES_FILENAME,
            RECEIPT_FILENAME,
        }
        observed_files: set[str] = set()
        for path in worker_root.rglob("*"):
            relative = path.relative_to(worker_root).as_posix()
            if path.is_symlink():
                raise ValueError(f"397B HOLDOUT worker output contains symlink: {relative}")
            if path.is_file():
                observed_files.add(relative)
        if observed_files != expected_files:
            missing = sorted(expected_files - observed_files)
            extra = sorted(observed_files - expected_files)
            raise ValueError(
                "397B HOLDOUT worker output file set differs from sealed contract: "
                f"missing={missing}; extra={extra}"
            )

        holdout_verification = verify_cyber_megatron_holdout_plan(
            root=root_path,
            plan_path=holdout_plan_path,
            candidate_manifest_path=candidate_manifest_path,
            candidate_config_path=candidate_config_path,
            candidate_training_plan_path=candidate_training_plan_path,
            checkpoint_dir=checkpoint_dir,
            inference_pack=inference_pack,
            signature_path=signature_path,
            reviewer_public_key_path=reviewer_public_key_path,
            reviewer_trust_policy_path=reviewer_trust_policy_path,
            owner_public_key_path=owner_public_key_path,
            minimum_case_count=minimum_case_count,
        )
        if not holdout_verification.valid or holdout_verification.plan is None:
            raise ValueError(
                "397B source HOLDOUT plan does not freshly verify: "
                + "; ".join(holdout_verification.violations[:5])
            )
        source_plan_verified = True
        holdout_plan = holdout_verification.plan

        worker_plan = _load_worker_plan(worker_root)
        expected_command = _swift_command(
            checkpoint_path=worker_plan.checkpoint_path,
            requests_path=(worker_root / REQUESTS_FILENAME).relative_to(root_path).as_posix(),
            raw_results_path=(worker_root / RAW_RESULTS_FILENAME).relative_to(root_path).as_posix(),
            generation_policy=worker_plan.generation_policy,
            profile=worker_plan.profile,
        )
        worker_plan_checks = (
            worker_plan.holdout_plan_sha256 == holdout_plan.plan_sha256,
            worker_plan.candidate_sha256 == holdout_plan.candidate_sha256,
            worker_plan.checkpoint_tree_sha256 == holdout_plan.checkpoint_tree_sha256,
            worker_plan.request_count == holdout_plan.case_count,
            worker_plan.case_ids_sha256 == holdout_plan.case_ids_sha256,
            worker_plan.generation_policy == holdout_plan.generation_policy,
            worker_plan.generation_policy_sha256 == holdout_plan.generation_policy_sha256,
            worker_plan.command == expected_command,
            worker_plan.command_sha256 == _sha256_canonical(expected_command),
            _worker_plan_digest(worker_plan.model_dump(mode="json")) == worker_plan.plan_sha256,
        )
        if not all(worker_plan_checks):
            raise ValueError("397B HOLDOUT worker plan differs from verified source plan")
        checkpoint_path = _resolve_no_symlinks(
            root_path,
            worker_plan.checkpoint_path,
            "397B merged checkpoint",
        )
        _validate_merged_inference_checkpoint(
            checkpoint_path,
            expected_tree_sha256=worker_plan.checkpoint_tree_sha256,
        )
        worker_plan_verified = True
        request_count = worker_plan.request_count

        request_manifest = _load_request_manifest(worker_root)
        requests_raw = _read_regular_bytes(
            worker_root / REQUESTS_FILENAME,
            "397B HOLDOUT requests",
        )
        cases, _inference_manifest, _manifest_raw = _load_inference_pack(inference_pack)
        expected_request_text, expected_request_manifest = _build_request_material(cases)
        if requests_raw != expected_request_text.encode("utf-8"):
            raise ValueError("397B HOLDOUT requests differ from freshly rebuilt prompts")
        if request_manifest != expected_request_manifest:
            raise ValueError("397B HOLDOUT request bindings differ from freshly rebuilt prompts")
        if _sha256_bytes(requests_raw) != worker_plan.requests_sha256:
            raise ValueError("397B HOLDOUT request SHA differs from worker plan")
        request_binding_verified = True

        execution = _load_execution_state(worker_root)
        raw_results = _read_regular_bytes(
            worker_root / RAW_RESULTS_FILENAME,
            "397B SWIFT raw results",
        )
        execution_checks = (
            execution.worker_plan_sha256 == worker_plan.plan_sha256,
            execution.command_sha256 == worker_plan.command_sha256,
            execution.checkpoint_tree_sha256 == worker_plan.checkpoint_tree_sha256,
            execution.state == "completed",
            execution.return_code == 0,
            execution.raw_results_sha256 == _sha256_bytes(raw_results),
            execution.runtime_versions.get("ms-swift") == worker_plan.profile.ms_swift_version,
            execution.runtime_versions.get("vllm") == worker_plan.profile.vllm_version,
            bool(execution.runtime_versions.get("ray")),
            _execution_state_digest(execution.model_dump(mode="json")) == execution.state_sha256,
        )
        if not all(execution_checks):
            raise ValueError("397B HOLDOUT execution state differs from sealed worker plan/results")
        execution_verified = True

        rebuilt_predictions, rebuilt_failures = _replay_raw_results(
            raw_results=raw_results,
            cases=cases,
            request_manifest=request_manifest,
            checkpoint_tree_sha256=worker_plan.checkpoint_tree_sha256,
        )
        predictions_raw = _read_regular_bytes(
            worker_root / PREDICTIONS_FILENAME,
            "397B HOLDOUT predictions",
        )
        failures_raw = _read_regular_bytes(
            worker_root / FAILURES_FILENAME,
            "397B HOLDOUT failures",
        )
        expected_predictions_raw = _jsonl_text(rebuilt_predictions).encode("utf-8")
        expected_failures_raw = _jsonl_text(rebuilt_failures).encode("utf-8")
        if predictions_raw != expected_predictions_raw:
            raise ValueError("397B normalized predictions differ from raw SWIFT replay")
        if failures_raw != expected_failures_raw:
            raise ValueError("397B normalized failures differ from raw SWIFT replay")
        raw_results_replayed = True
        prediction_count = len(rebuilt_predictions)
        failure_count = len(rebuilt_failures)

        receipt_raw = _read_regular_bytes(
            worker_root / RECEIPT_FILENAME,
            "397B HOLDOUT receipt",
        )
        receipt = CyberMegatronHoldoutRunReceipt.model_validate_json(receipt_raw)
        if receipt_raw != _json_text(receipt).encode("utf-8"):
            raise ValueError("397B HOLDOUT receipt is not canonical")
        expected_receipt_checks = (
            receipt.worker_plan_sha256 == worker_plan.plan_sha256,
            receipt.execution_state_sha256 == execution.state_sha256,
            receipt.holdout_plan_sha256 == holdout_plan.plan_sha256,
            receipt.candidate_sha256 == worker_plan.candidate_sha256,
            receipt.checkpoint_tree_sha256 == worker_plan.checkpoint_tree_sha256,
            receipt.model == worker_plan.model,
            receipt.model_revision == worker_plan.model_revision,
            receipt.request_count == request_count,
            receipt.prediction_count == prediction_count,
            receipt.failure_count == failure_count,
            receipt.failed_case_ids == [row.case_id for row in rebuilt_failures],
            receipt.raw_results_sha256 == _sha256_bytes(raw_results),
            receipt.predictions_sha256 == _sha256_bytes(predictions_raw),
            receipt.failures_sha256 == _sha256_bytes(failures_raw),
            receipt.runtime_versions == execution.runtime_versions,
            _receipt_digest(receipt.model_dump(mode="json")) == receipt.receipt_sha256,
        )
        if not all(expected_receipt_checks):
            raise ValueError("397B HOLDOUT receipt differs from independently verified output")
        receipt_verified = True

        predicted_ids = {row.case_id for row in rebuilt_predictions}
        failed_ids = {row.case_id for row in rebuilt_failures}
        expected_ids = set(holdout_plan.case_ids)
        if predicted_ids & failed_ids:
            raise ValueError("397B HOLDOUT cases appear as both prediction and failure")
        if predicted_ids | failed_ids != expected_ids:
            raise ValueError("397B HOLDOUT output does not account for every planned case")
        complete_case_accounting = True
    except (OSError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-holdout-output-verification.v2",
        "valid": not violations,
        "request_count": request_count,
        "prediction_count": prediction_count,
        "failure_count": failure_count,
        "source_plan_verified": source_plan_verified,
        "worker_plan_verified": worker_plan_verified,
        "request_binding_verified": request_binding_verified,
        "execution_verified": execution_verified,
        "raw_results_replayed": raw_results_replayed,
        "receipt_verified": receipt_verified,
        "complete_case_accounting": complete_case_accounting,
        "violations": violations,
    }
    return CyberMegatronHoldoutOutputVerification(
        **payload,
        verification_sha256=_verification_digest(payload),
    )
