from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class RecoveryPolicy(StrictModel):
    schema_version: Literal["sentinel.mcore-recovery-policy.v1"] = "sentinel.mcore-recovery-policy.v1"
    max_retries_per_batch: int = Field(ge=0, le=16)
    learning_rate_backoff: float = Field(gt=0.0, lt=1.0)
    min_learning_rate: float = Field(gt=0.0)
    quarantine_after_failures: int = Field(gt=0, le=32)
    skip_quarantined_batch: bool = True

    @model_validator(mode="after")
    def coherent(self) -> "RecoveryPolicy":
        if self.quarantine_after_failures > self.max_retries_per_batch + 1:
            raise ValueError("quarantine_after_failures cannot exceed max_retries_per_batch + 1")
        return self


class RecoveryDecision(StrictModel):
    schema_version: Literal["sentinel.mcore-recovery-decision.v1"] = "sentinel.mcore-recovery-decision.v1"
    globally_unstable: bool
    offending_batch_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    failure_count: int = Field(ge=0)
    retry_allowed: bool
    quarantine_batch: bool
    skip_batch: bool
    next_learning_rate: float = Field(gt=0.0)
    reason_codes: list[str] = Field(default_factory=list)


class RecoveryLedgerState(StrictModel):
    schema_version: Literal["sentinel.mcore-recovery-ledger-state.v1"] = (
        "sentinel.mcore-recovery-ledger-state.v1"
    )
    failures_by_fingerprint: dict[str, int] = Field(default_factory=dict)
    quarantined: list[str] = Field(default_factory=list)
    total_retries: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def coherent(self) -> "RecoveryLedgerState":
        for fingerprint, count in self.failures_by_fingerprint.items():
            if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
                raise ValueError("recovery ledger contains invalid fingerprint")
            if count <= 0:
                raise ValueError("recovery failure counts must be positive")
        if len(set(self.quarantined)) != len(self.quarantined):
            raise ValueError("quarantined fingerprints must be unique")
        unknown = set(self.quarantined) - set(self.failures_by_fingerprint)
        if unknown:
            raise ValueError("quarantined fingerprint must have recorded failures")
        return self


@dataclass
class RecoveryLedger:
    failures_by_fingerprint: dict[str, int]
    quarantined: set[str]
    total_retries: int = 0

    @classmethod
    def empty(cls) -> "RecoveryLedger":
        return cls(failures_by_fingerprint={}, quarantined=set(), total_retries=0)

    @classmethod
    def from_state(cls, state: RecoveryLedgerState) -> "RecoveryLedger":
        return cls(
            failures_by_fingerprint=dict(state.failures_by_fingerprint),
            quarantined=set(state.quarantined),
            total_retries=state.total_retries,
        )

    def to_state(self) -> RecoveryLedgerState:
        return RecoveryLedgerState(
            failures_by_fingerprint=dict(sorted(self.failures_by_fingerprint.items())),
            quarantined=sorted(self.quarantined),
            total_retries=self.total_retries,
        )

    def record_failure(self, fingerprint: str) -> int:
        count = self.failures_by_fingerprint.get(fingerprint, 0) + 1
        self.failures_by_fingerprint[fingerprint] = count
        return count

    def record_retry(self) -> None:
        self.total_retries += 1

    def quarantine(self, fingerprint: str) -> None:
        self.quarantined.add(fingerprint)


def fingerprint_batch(*, catalog_sha256: str, global_sequence_offset: int, global_window: int) -> str:
    payload = {
        "catalog_sha256": catalog_sha256,
        "global_sequence_offset": int(global_sequence_offset),
        "global_window": int(global_window),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def distributed_any_unstable(local_unstable: bool, *, device: Any) -> bool:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch distributed is required for recovery consensus") from exc
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before recovery consensus")
    flag = torch.tensor([1 if local_unstable else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(int(flag.item()))


def build_recovery_decision(
    *,
    globally_unstable: bool,
    batch_fingerprint: str,
    current_learning_rate: float,
    policy: RecoveryPolicy,
    ledger: RecoveryLedger,
    reason_codes: list[str],
) -> RecoveryDecision:
    if not globally_unstable:
        return RecoveryDecision(
            globally_unstable=False,
            offending_batch_fingerprint=None,
            failure_count=0,
            retry_allowed=False,
            quarantine_batch=False,
            skip_batch=False,
            next_learning_rate=current_learning_rate,
            reason_codes=[],
        )

    failures = ledger.record_failure(batch_fingerprint)
    quarantine = failures >= policy.quarantine_after_failures
    if quarantine:
        ledger.quarantine(batch_fingerprint)
    retry_allowed = failures <= policy.max_retries_per_batch and not quarantine
    skip_batch = quarantine and policy.skip_quarantined_batch
    next_lr = max(policy.min_learning_rate, current_learning_rate * policy.learning_rate_backoff)
    return RecoveryDecision(
        globally_unstable=True,
        offending_batch_fingerprint=batch_fingerprint,
        failure_count=failures,
        retry_allowed=retry_allowed,
        quarantine_batch=quarantine,
        skip_batch=skip_batch,
        next_learning_rate=next_lr,
        reason_codes=sorted(set(reason_codes)),
    )
