from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Any


class EffectAgreement(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ObservedEffect:
    action_id: str
    tenant_id: str
    target_id: str
    observer_id: str
    expected_state: str
    observed_state: str | None
    evidence_digest: str
    metadata: Mapping[str, Any]

    @property
    def agreement(self) -> EffectAgreement:
        if self.observed_state is None:
            return EffectAgreement.UNKNOWN
        if self.observed_state == self.expected_state:
            return EffectAgreement.AGREE
        return EffectAgreement.DISAGREE

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def require_verified_effect(effect: ObservedEffect) -> None:
    if effect.agreement is EffectAgreement.UNKNOWN:
        raise RuntimeError("defensive effect is unobserved")
    if effect.agreement is EffectAgreement.DISAGREE:
        raise RuntimeError("observed defensive effect contradicts intended state")
