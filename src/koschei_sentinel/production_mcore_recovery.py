from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

RECOVERY_REASON_CODES: tuple[str, ...] = (
    "non_finite_parameter_sample", "non_finite_gradient_sample", "grad_norm_threshold_exceeded",
    "cuda_memory_threshold_exceeded", "non_finite_moe_aux_loss", "non_finite_moe_z_loss",
    "router_telemetry_missing", "expert_utilization_below_threshold", "expert_load_imbalance",
    "non_finite_lm_loss", "non_finite_grad_norm", "non_finite_activation", "lm_loss_spike",
    "grad_norm_spike", "activation_rms_spike", "activation_abs_threshold_exceeded",
    "router_utilization_collapse_trend", "optimizer_step_failed", "non_positive_step_duration",
    "rank_exception",
)


class RecoveryPolicy(StrictModel):
    schema_version: Literal["sentinel.mcore-recovery-policy.v1"] = "sentinel.mcore-recovery-policy.v1"
    max_retries_per_batch: int = Field(ge=0, le=16); learning_rate_backoff: float = Field(gt=0.0, lt=1.0); min_learning_rate: float = Field(gt=0.0); quarantine_after_failures: int = Field(gt=0, le=32); skip_quarantined_batch: bool = True
    @model_validator(mode="after")
    def coherent(self):
        if self.quarantine_after_failures > self.max_retries_per_batch + 1: raise ValueError("quarantine_after_failures cannot exceed max_retries_per_batch + 1")
        return self


class RecoveryDecision(StrictModel):
    schema_version: Literal["sentinel.mcore-recovery-decision.v1"] = "sentinel.mcore-recovery-decision.v1"; globally_unstable: bool; offending_batch_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$"); failure_count: int = Field(ge=0); retry_allowed: bool; quarantine_batch: bool; skip_batch: bool; next_learning_rate: float = Field(gt=0.0); reason_codes: list[str] = Field(default_factory=list)


class RecoveryLedgerState(StrictModel):
    schema_version: Literal["sentinel.mcore-recovery-ledger-state.v1"] = "sentinel.mcore-recovery-ledger-state.v1"; failures_by_fingerprint: dict[str,int] = Field(default_factory=dict); quarantined: list[str] = Field(default_factory=list); total_retries: int = Field(default=0, ge=0)
    @model_validator(mode="after")
    def coherent(self):
        for fingerprint,count in self.failures_by_fingerprint.items():
            if len(fingerprint)!=64 or any(ch not in "0123456789abcdef" for ch in fingerprint): raise ValueError("recovery ledger contains invalid fingerprint")
            if count<=0: raise ValueError("recovery failure counts must be positive")
        if len(set(self.quarantined))!=len(self.quarantined): raise ValueError("quarantined fingerprints must be unique")
        if set(self.quarantined)-set(self.failures_by_fingerprint): raise ValueError("quarantined fingerprint must have recorded failures")
        return self


@dataclass
class RecoveryLedger:
    failures_by_fingerprint: dict[str,int]; quarantined: set[str]; total_retries: int = 0
    @classmethod
    def empty(cls): return cls({},set(),0)
    @classmethod
    def from_state(cls,state): return cls(dict(state.failures_by_fingerprint),set(state.quarantined),state.total_retries)
    def to_state(self): return RecoveryLedgerState(failures_by_fingerprint=dict(sorted(self.failures_by_fingerprint.items())),quarantined=sorted(self.quarantined),total_retries=self.total_retries)
    def record_failure(self,fingerprint):
        count=self.failures_by_fingerprint.get(fingerprint,0)+1; self.failures_by_fingerprint[fingerprint]=count; return count
    def record_retry(self): self.total_retries+=1
    def quarantine(self,fingerprint): self.quarantined.add(fingerprint)


def fingerprint_batch(*,catalog_sha256,global_sequence_offset,global_window):
    raw=json.dumps({"catalog_sha256":catalog_sha256,"global_sequence_offset":int(global_sequence_offset),"global_window":int(global_window)},sort_keys=True,separators=(",",":")).encode(); return hashlib.sha256(raw).hexdigest()


def _canonical_recovery_reason(code: str) -> str:
    # Exception class names are intentionally not transported through collectives;
    # they can differ by rank and would make the recovery decision non-deterministic.
    if code.startswith("rank_exception:") or code == "remote_rank_instability": return "rank_exception"
    return code if code in RECOVERY_REASON_CODES else "rank_exception"


def distributed_recovery_consensus(local_reason_codes: list[str], *, device: Any) -> tuple[bool, list[str]]:
    """Return the identical ordered blocker set on every rank using a fixed-size MAX reduction."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc: raise RuntimeError("PyTorch distributed is required for recovery consensus") from exc
    if not dist.is_initialized(): raise RuntimeError("torch.distributed must be initialized before recovery consensus")
    local={_canonical_recovery_reason(code) for code in local_reason_codes}
    bits=torch.tensor([1 if code in local else 0 for code in RECOVERY_REASON_CODES],device=device,dtype=torch.int32)
    dist.all_reduce(bits,op=dist.ReduceOp.MAX)
    global_codes=[code for index,code in enumerate(RECOVERY_REASON_CODES) if int(bits[index].item())]
    return bool(global_codes),global_codes


def distributed_any_unstable(local_unstable: bool, *, device: Any) -> bool:
    # Compatibility helper for older training loops.
    unstable,_=distributed_recovery_consensus(["rank_exception"] if local_unstable else [],device=device); return unstable


def build_recovery_decision(*,globally_unstable,batch_fingerprint,current_learning_rate,policy,ledger,reason_codes):
    if not globally_unstable: return RecoveryDecision(globally_unstable=False,offending_batch_fingerprint=None,failure_count=0,retry_allowed=False,quarantine_batch=False,skip_batch=False,next_learning_rate=current_learning_rate,reason_codes=[])
    failures=ledger.record_failure(batch_fingerprint); quarantine=failures>=policy.quarantine_after_failures
    if quarantine: ledger.quarantine(batch_fingerprint)
    retry_allowed=failures<=policy.max_retries_per_batch and not quarantine
    if retry_allowed: ledger.record_retry()
    return RecoveryDecision(globally_unstable=True,offending_batch_fingerprint=batch_fingerprint,failure_count=failures,retry_allowed=retry_allowed,quarantine_batch=quarantine,skip_batch=quarantine and policy.skip_quarantined_batch,next_learning_rate=max(policy.min_learning_rate,current_learning_rate*policy.learning_rate_backoff),reason_codes=sorted(set(reason_codes)))
