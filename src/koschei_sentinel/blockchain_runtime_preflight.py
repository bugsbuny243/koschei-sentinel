from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_base_candidates import (
    BlockchainBaseCandidate,
    BlockchainBaseCandidateRegistry,
    LicenseReviewStatus,
    RuntimeIntegrationStatus,
    load_base_candidate_registry,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, model_digest

_DIGEST = r"^[a-f0-9]{64}$"
_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class BlockchainHardwareInventory(StrictModel):
    schema_version: Literal["sentinel.blockchain-hardware-inventory.v1"] = (
        "sentinel.blockchain-hardware-inventory.v1"
    )
    inventory_id: str = Field(pattern=_ID)
    cuda_available: bool
    gpu_count: int = Field(ge=0, le=1024)
    gpu_model: str | None = Field(default=None, max_length=256)
    vram_per_gpu_mb: int = Field(ge=0)
    host_ram_mb: int = Field(ge=1)
    free_disk_mb: int = Field(ge=1)
    compute_capability: str | None = Field(default=None, pattern=r"^[0-9]+\.[0-9]+$")
    bfloat16_supported: bool
    inventory_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def inventory_is_consistent(self) -> BlockchainHardwareInventory:
        if self.cuda_available:
            if self.gpu_count < 1 or not self.gpu_model or self.vram_per_gpu_mb < 1:
                raise ValueError("CUDA inventory must describe at least one GPU")
        else:
            if self.gpu_count != 0 or self.vram_per_gpu_mb != 0:
                raise ValueError("non-CUDA inventory may not claim GPU resources")
        payload = self.model_dump(mode="json")
        expected = payload.pop("inventory_digest")
        if _digest(payload) != expected:
            raise ValueError("blockchain hardware inventory digest mismatch")
        return self


class BlockchainRuntimePreflightPolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-runtime-preflight-policy.v1"] = (
        "sentinel.blockchain-runtime-preflight-policy.v1"
    )
    policy_id: str = Field(pattern=_ID)
    quantization_bits: Literal[4, 8] = 4
    checkpoint_bytes_per_parameter: float = Field(default=2.0, ge=0.5, le=8.0)
    disk_safety_bps: int = Field(default=11500, ge=10_000, le=30_000)
    vram_safety_bps: int = Field(default=14000, ge=10_000, le=30_000)
    host_ram_safety_bps: int = Field(default=12000, ge=10_000, le=30_000)
    min_free_disk_margin_mb: int = Field(default=8192, ge=0)
    min_host_ram_margin_mb: int = Field(default=4096, ge=0)
    probe_sequence_tokens: int = Field(default=256, ge=32, le=4096)
    require_cuda: Literal[True] = True
    require_forward: Literal[True] = True
    require_backward: Literal[True] = True
    require_lora_attach: Literal[True] = True
    allow_trust_remote_code: Literal[False] = False
    require_allowlisted_open_license: Literal[True] = True
    require_native_transformers: Literal[True] = True


class BlockchainRuntimePreflightPlan(StrictModel):
    schema_version: Literal["sentinel.blockchain-runtime-preflight-plan.v1"] = (
        "sentinel.blockchain-runtime-preflight-plan.v1"
    )
    candidate_id: str = Field(pattern=_ID)
    model_id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    registry_digest: str = Field(pattern=_DIGEST)
    candidate_digest: str = Field(pattern=_DIGEST)
    policy_digest: str = Field(pattern=_DIGEST)
    inventory_digest: str = Field(pattern=_DIGEST)
    quantization_bits: Literal[4, 8]
    estimated_checkpoint_disk_mb: int = Field(ge=1)
    estimated_quantized_weight_mb: int = Field(ge=1)
    estimated_min_vram_mb: int = Field(ge=1)
    estimated_min_host_ram_mb: int = Field(ge=1)
    available_single_gpu_vram_mb: int = Field(ge=0)
    available_host_ram_mb: int = Field(ge=1)
    available_free_disk_mb: int = Field(ge=1)
    static_hardware_fit: bool
    runtime_probe_authorized: bool
    blockers: list[str] = Field(default_factory=list, max_length=64)
    training_started: Literal[False] = False
    training_authorized: Literal[False] = False
    plan_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def digest_and_authority_are_valid(self) -> BlockchainRuntimePreflightPlan:
        payload = self.model_dump(mode="json")
        expected = payload.pop("plan_digest")
        if _digest(payload) != expected:
            raise ValueError("blockchain runtime preflight plan digest mismatch")
        if self.runtime_probe_authorized and self.blockers:
            raise ValueError("authorized runtime probe may not retain blockers")
        if self.runtime_probe_authorized != self.static_hardware_fit:
            raise ValueError("runtime probe authorization must match fail-closed static fit")
        return self


class BlockchainRuntimeProbeReceipt(StrictModel):
    schema_version: Literal["sentinel.blockchain-runtime-probe-receipt.v1"] = (
        "sentinel.blockchain-runtime-probe-receipt.v1"
    )
    candidate_id: str = Field(pattern=_ID)
    model_id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    registry_digest: str = Field(pattern=_DIGEST)
    candidate_digest: str = Field(pattern=_DIGEST)
    policy_digest: str = Field(pattern=_DIGEST)
    inventory_digest: str = Field(pattern=_DIGEST)
    plan_digest: str = Field(pattern=_DIGEST)
    environment_digest: str = Field(pattern=_DIGEST)
    tokenizer_loaded: bool
    config_loaded: bool
    quantized_model_loaded: bool
    lora_attached: bool
    forward_ok: bool
    backward_ok: bool
    exact_revision_observed: bool
    trust_remote_code_used: Literal[False] = False
    raw_model_outputs_stored: Literal[False] = False
    probe_input_tokens: int = Field(ge=0, le=4096)
    trainable_parameters: int = Field(ge=0)
    total_parameters: int = Field(ge=0)
    peak_gpu_memory_mb: int = Field(ge=0)
    status: Literal["PASSED", "FAILED"]
    error_code: str | None = Field(default=None, max_length=128)
    training_started: Literal[False] = False
    training_authorized: Literal[False] = False
    receipt_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_is_canonical(self) -> BlockchainRuntimeProbeReceipt:
        payload = self.model_dump(mode="json")
        expected = payload.pop("receipt_digest")
        if _digest(payload) != expected:
            raise ValueError("blockchain runtime probe receipt digest mismatch")
        if self.status == "PASSED":
            required = (
                self.tokenizer_loaded,
                self.config_loaded,
                self.quantized_model_loaded,
                self.lora_attached,
                self.forward_ok,
                self.backward_ok,
                self.exact_revision_observed,
            )
            if not all(required) or self.error_code is not None:
                raise ValueError("passed runtime probe receipt is incomplete")
        elif self.error_code is None:
            raise ValueError("failed runtime probe receipt requires an error_code")
        return self


class BlockchainRuntimePreflightAudit(StrictModel):
    schema_version: Literal["sentinel.blockchain-runtime-preflight-audit.v1"] = (
        "sentinel.blockchain-runtime-preflight-audit.v1"
    )
    candidate_id: str = Field(pattern=_ID)
    ready_for_training_authorization_review: bool
    plan_digest: str = Field(pattern=_DIGEST)
    receipt_digest: str = Field(pattern=_DIGEST)
    static_hardware_fit: bool
    runtime_probe_passed: bool
    peak_gpu_memory_mb: int = Field(ge=0)
    available_single_gpu_vram_mb: int = Field(ge=0)
    headroom_mb: int
    violations: list[str]
    training_started: Literal[False] = False
    production_authority: Literal[False] = False


def build_hardware_inventory(
    *,
    inventory_id: str,
    cuda_available: bool,
    gpu_count: int,
    gpu_model: str | None,
    vram_per_gpu_mb: int,
    host_ram_mb: int,
    free_disk_mb: int,
    compute_capability: str | None,
    bfloat16_supported: bool,
) -> BlockchainHardwareInventory:
    payload = {
        "schema_version": "sentinel.blockchain-hardware-inventory.v1",
        "inventory_id": inventory_id,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_model": gpu_model,
        "vram_per_gpu_mb": vram_per_gpu_mb,
        "host_ram_mb": host_ram_mb,
        "free_disk_mb": free_disk_mb,
        "compute_capability": compute_capability,
        "bfloat16_supported": bfloat16_supported,
    }
    return BlockchainHardwareInventory.model_validate(
        {**payload, "inventory_digest": _digest(payload)}
    )


def inspect_local_hardware(inventory_id: str = "local-runtime") -> BlockchainHardwareInventory:
    free_disk_mb = shutil.disk_usage(Path.cwd()).free // (1024 * 1024)
    host_ram_mb = _host_ram_mb()
    cuda_available = False
    gpu_count = 0
    gpu_model: str | None = None
    vram_per_gpu_mb = 0
    compute_capability: str | None = None
    bfloat16_supported = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_count = int(torch.cuda.device_count())
            properties = torch.cuda.get_device_properties(0)
            gpu_model = str(properties.name)
            vram_per_gpu_mb = int(properties.total_memory // (1024 * 1024))
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
            bfloat16_supported = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        pass
    return build_hardware_inventory(
        inventory_id=inventory_id,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        vram_per_gpu_mb=vram_per_gpu_mb,
        host_ram_mb=host_ram_mb,
        free_disk_mb=free_disk_mb,
        compute_capability=compute_capability,
        bfloat16_supported=bfloat16_supported,
    )


def load_hardware_inventory(path: str | Path) -> BlockchainHardwareInventory:
    return _load_model(path, BlockchainHardwareInventory, "blockchain hardware inventory")


def load_runtime_preflight_policy(path: str | Path) -> BlockchainRuntimePreflightPolicy:
    return _load_model(path, BlockchainRuntimePreflightPolicy, "blockchain runtime preflight policy")


def load_runtime_preflight_plan(path: str | Path) -> BlockchainRuntimePreflightPlan:
    return _load_model(path, BlockchainRuntimePreflightPlan, "blockchain runtime preflight plan")


def load_runtime_probe_receipt(path: str | Path) -> BlockchainRuntimeProbeReceipt:
    return _load_model(path, BlockchainRuntimeProbeReceipt, "blockchain runtime probe receipt")


def plan_runtime_preflight(
    registry: BlockchainBaseCandidateRegistry,
    candidate_id: str,
    inventory: BlockchainHardwareInventory,
    policy: BlockchainRuntimePreflightPolicy,
) -> BlockchainRuntimePreflightPlan:
    candidate = _candidate_by_id(registry, candidate_id)
    blockers: list[str] = []
    if policy.require_cuda and not inventory.cuda_available:
        blockers.append("cuda_unavailable")
    if policy.require_allowlisted_open_license and (
        candidate.license_review_status is not LicenseReviewStatus.ALLOWLISTED_OPEN_LICENSE
    ):
        blockers.append("license_not_allowlisted")
    if policy.require_native_transformers and (
        candidate.runtime_integration_status
        is not RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED
    ):
        blockers.append("native_transformers_not_approved")
    if candidate.trust_remote_code_required:
        blockers.append("trust_remote_code_required")
    if candidate.blocked_reasons:
        blockers.extend(f"candidate_block:{item}" for item in candidate.blocked_reasons)

    checkpoint_mb, quantized_mb, min_vram_mb, min_host_ram_mb = estimate_requirements(
        candidate,
        policy,
    )
    if inventory.free_disk_mb < checkpoint_mb + policy.min_free_disk_margin_mb:
        blockers.append("insufficient_free_disk")
    if inventory.vram_per_gpu_mb < min_vram_mb:
        blockers.append("insufficient_single_gpu_vram")
    if inventory.host_ram_mb < min_host_ram_mb + policy.min_host_ram_margin_mb:
        blockers.append("insufficient_host_ram")

    blockers = sorted(set(blockers))
    static_fit = not blockers
    payload = {
        "schema_version": "sentinel.blockchain-runtime-preflight-plan.v1",
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "revision": candidate.revision,
        "registry_digest": registry.registry_digest,
        "candidate_digest": model_digest(candidate),
        "policy_digest": model_digest(policy),
        "inventory_digest": inventory.inventory_digest,
        "quantization_bits": policy.quantization_bits,
        "estimated_checkpoint_disk_mb": checkpoint_mb,
        "estimated_quantized_weight_mb": quantized_mb,
        "estimated_min_vram_mb": min_vram_mb,
        "estimated_min_host_ram_mb": min_host_ram_mb,
        "available_single_gpu_vram_mb": inventory.vram_per_gpu_mb,
        "available_host_ram_mb": inventory.host_ram_mb,
        "available_free_disk_mb": inventory.free_disk_mb,
        "static_hardware_fit": static_fit,
        "runtime_probe_authorized": static_fit,
        "blockers": blockers,
        "training_started": False,
        "training_authorized": False,
    }
    return BlockchainRuntimePreflightPlan.model_validate(
        {**payload, "plan_digest": _digest(payload)}
    )


def estimate_requirements(
    candidate: BlockchainBaseCandidate,
    policy: BlockchainRuntimePreflightPolicy,
) -> tuple[int, int, int, int]:
    parameters = candidate.declared_total_parameters_billion * 1_000_000_000
    mib = 1024 * 1024
    checkpoint = parameters * policy.checkpoint_bytes_per_parameter / mib
    checkpoint *= policy.disk_safety_bps / 10_000
    quantized = parameters * (policy.quantization_bits / 8.0) / mib
    vram = quantized * policy.vram_safety_bps / 10_000
    host_ram = quantized * policy.host_ram_safety_bps / 10_000
    return tuple(math.ceil(value) for value in (checkpoint, quantized, vram, host_ram))


def execute_runtime_probe(
    registry: BlockchainBaseCandidateRegistry,
    inventory: BlockchainHardwareInventory,
    policy: BlockchainRuntimePreflightPolicy,
    plan: BlockchainRuntimePreflightPlan,
) -> BlockchainRuntimeProbeReceipt:
    recomputed = plan_runtime_preflight(registry, plan.candidate_id, inventory, policy)
    if recomputed != plan:
        raise ValueError("runtime preflight plan is stale or no longer matches its inputs")
    if not plan.runtime_probe_authorized:
        raise ValueError("runtime probe is blocked by the preflight plan")

    candidate = _candidate_by_id(registry, plan.candidate_id)
    if candidate.trust_remote_code_required or (
        candidate.runtime_integration_status
        is not RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED
    ):
        raise ValueError("v1 runtime probe refuses custom remote model code")

    state = {
        "tokenizer_loaded": False,
        "config_loaded": False,
        "quantized_model_loaded": False,
        "lora_attached": False,
        "forward_ok": False,
        "backward_ok": False,
        "exact_revision_observed": False,
        "probe_input_tokens": 0,
        "trainable_parameters": 0,
        "total_parameters": 0,
        "peak_gpu_memory_mb": 0,
    }
    error_code: str | None = None
    model: Any | None = None
    try:
        dependencies = _load_runtime_dependencies()
        torch = dependencies["torch"]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = dependencies["AutoConfig"].from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=False,
        )
        state["config_loaded"] = True
        observed_revision = getattr(config, "_commit_hash", None)
        state["exact_revision_observed"] = observed_revision in (None, candidate.revision)

        tokenizer = dependencies["AutoTokenizer"].from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=False,
        )
        state["tokenizer_loaded"] = True
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization = dependencies["BitsAndBytesConfig"](
            load_in_4bit=policy.quantization_bits == 4,
            load_in_8bit=policy.quantization_bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=False,
            quantization_config=quantization,
            device_map="auto",
        )
        state["quantized_model_loaded"] = True
        model = dependencies["prepare_model_for_kbit_training"](model)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        target_modules = _discover_lora_targets(model)
        if not target_modules:
            raise RuntimeError("no_supported_lora_targets")
        model = dependencies["get_peft_model"](
            model,
            dependencies["PeftLoraConfig"](
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=target_modules,
            ),
        )
        state["lora_attached"] = True
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        state["trainable_parameters"] = int(trainable)
        state["total_parameters"] = int(total)

        text = (
            "Defensive blockchain security analysis: identify evidence, preserve authority, "
            "and abstain when the evidence is insufficient."
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=policy.probe_sequence_tokens,
        )
        input_ids = encoded["input_ids"]
        state["probe_input_tokens"] = int(input_ids.shape[-1])
        device = model.get_input_embeddings().weight.device
        input_ids = input_ids.to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        if outputs.loss is None:
            raise RuntimeError("forward_missing_loss")
        state["forward_ok"] = True
        outputs.loss.backward()
        state["backward_ok"] = True
        state["peak_gpu_memory_mb"] = int(
            torch.cuda.max_memory_allocated() // (1024 * 1024)
        )
    except Exception as exc:  # runtime probe must persist only a coarse failure category
        error_code = _safe_error_code(exc)
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            dependencies.get("torch").cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass

    passed = all(
        bool(state[key])
        for key in (
            "tokenizer_loaded",
            "config_loaded",
            "quantized_model_loaded",
            "lora_attached",
            "forward_ok",
            "backward_ok",
            "exact_revision_observed",
        )
    ) and error_code is None
    environment_digest = _runtime_environment_digest(inventory)
    payload = {
        "schema_version": "sentinel.blockchain-runtime-probe-receipt.v1",
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "revision": candidate.revision,
        "registry_digest": registry.registry_digest,
        "candidate_digest": model_digest(candidate),
        "policy_digest": model_digest(policy),
        "inventory_digest": inventory.inventory_digest,
        "plan_digest": plan.plan_digest,
        "environment_digest": environment_digest,
        **state,
        "trust_remote_code_used": False,
        "raw_model_outputs_stored": False,
        "status": "PASSED" if passed else "FAILED",
        "error_code": None if passed else (error_code or "probe_failed"),
        "training_started": False,
        "training_authorized": False,
    }
    return BlockchainRuntimeProbeReceipt.model_validate(
        {**payload, "receipt_digest": _digest(payload)}
    )


def audit_runtime_preflight(
    registry: BlockchainBaseCandidateRegistry,
    inventory: BlockchainHardwareInventory,
    policy: BlockchainRuntimePreflightPolicy,
    plan: BlockchainRuntimePreflightPlan,
    receipt: BlockchainRuntimeProbeReceipt,
) -> BlockchainRuntimePreflightAudit:
    recomputed = plan_runtime_preflight(registry, plan.candidate_id, inventory, policy)
    if recomputed != plan:
        raise ValueError("runtime preflight plan no longer matches registry/hardware/policy")
    candidate = _candidate_by_id(registry, plan.candidate_id)
    expected_pairs = (
        (receipt.candidate_id, candidate.candidate_id, "candidate_id"),
        (receipt.model_id, candidate.model_id, "model_id"),
        (receipt.revision, candidate.revision, "revision"),
        (receipt.registry_digest, registry.registry_digest, "registry_digest"),
        (receipt.candidate_digest, model_digest(candidate), "candidate_digest"),
        (receipt.policy_digest, model_digest(policy), "policy_digest"),
        (receipt.inventory_digest, inventory.inventory_digest, "inventory_digest"),
        (receipt.plan_digest, plan.plan_digest, "plan_digest"),
    )
    for observed, expected, label in expected_pairs:
        if observed != expected:
            raise ValueError(f"runtime probe receipt {label} mismatch")

    violations: list[str] = []
    if not plan.static_hardware_fit:
        violations.append("static hardware fit did not pass")
    if receipt.status != "PASSED":
        violations.append("runtime probe did not pass")
    if receipt.trust_remote_code_used is not False:
        violations.append("runtime probe executed remote model code")
    if receipt.raw_model_outputs_stored is not False:
        violations.append("runtime probe persisted raw model outputs")
    if receipt.peak_gpu_memory_mb > inventory.vram_per_gpu_mb:
        violations.append("observed peak GPU memory exceeds inventory capacity")
    headroom = inventory.vram_per_gpu_mb - receipt.peak_gpu_memory_mb
    return BlockchainRuntimePreflightAudit(
        candidate_id=candidate.candidate_id,
        ready_for_training_authorization_review=not violations,
        plan_digest=plan.plan_digest,
        receipt_digest=receipt.receipt_digest,
        static_hardware_fit=plan.static_hardware_fit,
        runtime_probe_passed=receipt.status == "PASSED",
        peak_gpu_memory_mb=receipt.peak_gpu_memory_mb,
        available_single_gpu_vram_mb=inventory.vram_per_gpu_mb,
        headroom_mb=headroom,
        violations=violations,
    )


def write_model(model: StrictModel, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"preflight artifact already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def registry_and_candidate(
    registry_path: str | Path,
    candidate_id: str,
) -> tuple[BlockchainBaseCandidateRegistry, BlockchainBaseCandidate]:
    registry = load_base_candidate_registry(registry_path)
    return registry, _candidate_by_id(registry, candidate_id)


def _candidate_by_id(
    registry: BlockchainBaseCandidateRegistry,
    candidate_id: str,
) -> BlockchainBaseCandidate:
    matches = [item for item in registry.candidates if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate blockchain base candidate: {candidate_id}")
    return matches[0]


def _discover_lora_targets(model: Any) -> list[str]:
    supported = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return [name for name in supported if name in names]


def _load_runtime_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("training_dependencies_missing") from exc
    return {
        "torch": torch,
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "PeftLoraConfig": PeftLoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


def _runtime_environment_digest(inventory: BlockchainHardwareInventory) -> str:
    versions: dict[str, str] = {}
    for module_name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "missing"
    payload = {
        "inventory_digest": inventory.inventory_digest,
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "versions": versions,
    }
    return _digest(payload)


def _safe_error_code(exc: Exception) -> str:
    text = str(exc).casefold()
    markers = (
        "cuda_unavailable",
        "training_dependencies_missing",
        "no_supported_lora_targets",
        "forward_missing_loss",
    )
    for marker in markers:
        if marker in text:
            return marker
    name = exc.__class__.__name__.lower()
    return f"runtime_{name}"[:128]


def _host_ram_mb() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return max(1, pages * page_size // (1024 * 1024))
    except (AttributeError, OSError, ValueError):
        pass
    return 1


def _load_model(path: str | Path, model_type: type[StrictModel], label: str):
    try:
        return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
