from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class RiskClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    digest: str
    kind: str
    source: str


@dataclass(frozen=True)
class AuthorityEnvelope:
    principal_id: str
    controller_id: str | None
    delegate_id: str | None
    scopes: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    tenant_id: str
    task_class: str
    risk_class: RiskClass
    prompt: str
    evidence: tuple[EvidenceRef, ...] = ()
    authority: AuthorityEnvelope | None = None
    allowed_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceBinding:
    claim_id: str
    evidence_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    engine_id: str
    engine_version: str
    output: Mapping[str, Any]
    evidence_bindings: tuple[EvidenceBinding, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    refusal: str | None = None


class ReasoningEngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    def infer(self, request: InferenceRequest) -> InferenceResponse: ...


class ResponseValidator(Protocol):
    def validate(self, request: InferenceRequest, response: InferenceResponse) -> None: ...


class ModelRouter:
    """Routes by Sentinel policy; engine names never enter evidence semantics."""

    def __init__(
        self,
        engines: Sequence[ReasoningEngine],
        *,
        validators: Sequence[ResponseValidator] = (),
    ) -> None:
        if not engines:
            raise ValueError("at least one reasoning engine is required")
        self._engines = tuple(engines)
        self._validators = tuple(validators)

    def infer(self, request: InferenceRequest, *, engine_index: int = 0) -> InferenceResponse:
        if engine_index < 0 or engine_index >= len(self._engines):
            raise IndexError("engine_index is out of range")
        engine = self._engines[engine_index]
        response = engine.infer(request)
        if response.request_id != request.request_id:
            raise RuntimeError("reasoning engine returned a mismatched request_id")
        if response.engine_id != engine.engine_id or response.engine_version != engine.engine_version:
            raise RuntimeError("reasoning engine identity/version mismatch")
        for validator in self._validators:
            validator.validate(request, response)
        return response
