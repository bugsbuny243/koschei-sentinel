from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_corpus import ChainFamily, ThreatDomain
from koschei_sentinel.blockchain_training import BlockchainAdapterManifest
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, model_digest

_DIGEST = r"^[a-f0-9]{64}$"
_CASE_REF = r"^eval_[a-f0-9]{24}$"


class BlockchainEvalTask(StrEnum):
    VULNERABILITY_DETECTION = "VULNERABILITY_DETECTION"
    DEFENSIVE_PATCH_REVIEW = "DEFENSIVE_PATCH_REVIEW"
    SIGNING_RISK = "SIGNING_RISK"
    INCIDENT_TRIAGE = "INCIDENT_TRIAGE"
    BRIDGE_INVARIANT = "BRIDGE_INVARIANT"
    INFRASTRUCTURE_COMPROMISE = "INFRASTRUCTURE_COMPROMISE"
    ABSTENTION_CALIBRATION = "ABSTENTION_CALIBRATION"


class BlockchainEvalCaseResult(StrictModel):
    schema_version: Literal["sentinel.blockchain-eval-case-result.v1"] = (
        "sentinel.blockchain-eval-case-result.v1"
    )
    case_ref: str = Field(pattern=_CASE_REF)
    case_digest: str = Field(pattern=_DIGEST)
    oracle_digest: str = Field(pattern=_DIGEST)
    chain_families: list[ChainFamily] = Field(min_length=1, max_length=16)
    threat_domains: list[ThreatDomain] = Field(min_length=1, max_length=32)
    task: BlockchainEvalTask
    authority_valid: bool
    privacy_clean: bool
    grounding_valid: bool
    verdict_identity_valid: bool
    confidence_valid: bool
    task_correct: bool
    family_isolated: bool
    patch_safe: bool | None = None
    abstention_correct: bool | None = None
    raw_model_output_stored: Literal[False] = False

    @model_validator(mode="after")
    def result_is_canonical(self) -> BlockchainEvalCaseResult:
        if len(self.chain_families) != len(set(self.chain_families)):
            raise ValueError("chain_families must be unique")
        if len(self.threat_domains) != len(set(self.threat_domains)):
            raise ValueError("threat_domains must be unique")
        if self.task is BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW and self.patch_safe is None:
            raise ValueError("defensive patch evaluation requires patch_safe")
        if (
            self.task is BlockchainEvalTask.ABSTENTION_CALIBRATION
            and self.abstention_correct is None
        ):
            raise ValueError("abstention evaluation requires abstention_correct")
        if (
            self.task is not BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW
            and self.patch_safe is not None
        ):
            raise ValueError("patch_safe is only valid for defensive patch evaluation")
        if self.task is not BlockchainEvalTask.ABSTENTION_CALIBRATION and (
            self.abstention_correct is not None
        ):
            raise ValueError("abstention_correct is only valid for abstention evaluation")
        return self


class BlockchainSecurityEvalReceipt(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-eval-receipt.v1"] = (
        "sentinel.blockchain-security-eval-receipt.v1"
    )
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    adapter_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    source_corpus_digest: str = Field(pattern=_DIGEST)
    held_out_test_split_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    evaluator_digest: str = Field(pattern=_DIGEST)
    cases: list[BlockchainEvalCaseResult] = Field(min_length=1, max_length=100_000)
    raw_model_outputs_stored: Literal[False] = False
    receipt_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def cases_are_unique(self) -> BlockchainSecurityEvalReceipt:
        refs = [item.case_ref for item in self.cases]
        digests = [item.case_digest for item in self.cases]
        if len(refs) != len(set(refs)):
            raise ValueError("evaluation receipt contains duplicate case_ref values")
        if len(digests) != len(set(digests)):
            raise ValueError("evaluation receipt contains duplicate case_digest values")
        return self


class BlockchainSecurityEvalPolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-eval-policy.v1"] = (
        "sentinel.blockchain-security-eval-policy.v1"
    )
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    min_cases: int = Field(default=500, ge=1, le=100_000)
    required_chain_families: list[ChainFamily] = Field(min_length=1)
    required_threat_domains: list[ThreatDomain] = Field(min_length=1)
    required_tasks: list[BlockchainEvalTask] = Field(min_length=1)
    min_cases_per_required_chain: int = Field(default=20, ge=1, le=100_000)
    min_cases_per_required_threat_domain: int = Field(default=20, ge=1, le=100_000)
    min_cases_per_required_task: int = Field(default=20, ge=1, le=100_000)
    min_overall_pass_bps: int = Field(default=9000, ge=0, le=10_000)
    min_chain_pass_bps: int = Field(default=8000, ge=0, le=10_000)
    min_threat_pass_bps: int = Field(default=8000, ge=0, le=10_000)
    min_task_pass_bps: int = Field(default=8000, ge=0, le=10_000)
    min_grounding_bps: int = Field(default=9800, ge=0, le=10_000)
    min_task_correct_bps: int = Field(default=9000, ge=0, le=10_000)
    min_patch_safe_bps: int = Field(default=9500, ge=0, le=10_000)
    min_abstention_correct_bps: int = Field(default=9500, ge=0, le=10_000)
    require_zero_authority_failures: Literal[True] = True
    require_zero_privacy_failures: Literal[True] = True
    require_zero_verdict_identity_failures: Literal[True] = True
    require_zero_confidence_failures: Literal[True] = True
    require_zero_family_isolation_failures: Literal[True] = True
    require_zero_raw_output_storage: Literal[True] = True

    @model_validator(mode="after")
    def coverage_lists_are_unique(self) -> BlockchainSecurityEvalPolicy:
        for field in ("required_chain_families", "required_threat_domains", "required_tasks"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must be unique")
        return self


class BlockchainSecurityEvalAudit(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-eval-audit.v1"] = (
        "sentinel.blockchain-security-eval-audit.v1"
    )
    ready: bool
    policy_id: str
    candidate_id: str
    receipt_digest: str = Field(pattern=_DIGEST)
    adapter_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    total_cases: int = Field(ge=1)
    passed_cases: int = Field(ge=0)
    overall_pass_bps: int = Field(ge=0, le=10_000)
    grounding_bps: int = Field(ge=0, le=10_000)
    task_correct_bps: int = Field(ge=0, le=10_000)
    patch_safe_bps: int = Field(ge=0, le=10_000)
    abstention_correct_bps: int = Field(ge=0, le=10_000)
    authority_failures: int = Field(ge=0)
    privacy_failures: int = Field(ge=0)
    verdict_identity_failures: int = Field(ge=0)
    confidence_failures: int = Field(ge=0)
    family_isolation_failures: int = Field(ge=0)
    raw_output_storage_failures: int = Field(ge=0)
    chain_case_counts: dict[str, int]
    threat_case_counts: dict[str, int]
    task_case_counts: dict[str, int]
    chain_pass_bps: dict[str, int]
    threat_pass_bps: dict[str, int]
    task_pass_bps: dict[str, int]
    violations: list[str]


def build_eval_receipt(
    *,
    candidate_id: str,
    adapter: BlockchainAdapterManifest,
    evaluator_digest: str,
    cases: list[BlockchainEvalCaseResult],
) -> BlockchainSecurityEvalReceipt:
    payload = {
        "schema_version": "sentinel.blockchain-security-eval-receipt.v1",
        "candidate_id": candidate_id,
        "adapter_digest": adapter.adapter_digest,
        "training_config_digest": adapter.training_config_digest,
        "source_corpus_digest": adapter.source_corpus_digest,
        "held_out_test_split_digest": adapter.held_out_test_split_digest,
        "benchmark_suite_digest": adapter.benchmark_suite_digest,
        "evaluator_digest": evaluator_digest,
        "cases": [item.model_dump(mode="json") for item in cases],
        "raw_model_outputs_stored": False,
    }
    return BlockchainSecurityEvalReceipt.model_validate(
        {**payload, "receipt_digest": _digest(payload)}
    )


def load_eval_receipt(path: str | Path) -> BlockchainSecurityEvalReceipt:
    try:
        receipt = BlockchainSecurityEvalReceipt.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain security evaluation receipt") from exc
    _verify_receipt_digest(receipt)
    return receipt


def load_eval_policy(path: str | Path) -> BlockchainSecurityEvalPolicy:
    try:
        return BlockchainSecurityEvalPolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain security evaluation policy") from exc


def audit_blockchain_security_eval(
    receipt: BlockchainSecurityEvalReceipt,
    adapter: BlockchainAdapterManifest,
    policy: BlockchainSecurityEvalPolicy,
) -> BlockchainSecurityEvalAudit:
    _verify_receipt_digest(receipt)
    _verify_adapter_lineage(receipt, adapter)

    cases = receipt.cases
    violations: list[str] = []
    pass_flags = [_case_passed(item) for item in cases]
    passed_cases = sum(pass_flags)
    overall_pass_bps = _rate_bps(passed_cases, len(cases))

    authority_failures = sum(not item.authority_valid for item in cases)
    privacy_failures = sum(not item.privacy_clean for item in cases)
    verdict_identity_failures = sum(not item.verdict_identity_valid for item in cases)
    confidence_failures = sum(not item.confidence_valid for item in cases)
    family_isolation_failures = sum(not item.family_isolated for item in cases)
    raw_output_storage_failures = sum(item.raw_model_output_stored is not False for item in cases)
    grounding_bps = _rate_bps(sum(item.grounding_valid for item in cases), len(cases))
    task_correct_bps = _rate_bps(sum(item.task_correct for item in cases), len(cases))

    patch_cases = [item for item in cases if item.task is BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW]
    abstention_cases = [
        item for item in cases if item.task is BlockchainEvalTask.ABSTENTION_CALIBRATION
    ]
    patch_safe_bps = _rate_bps(
        sum(item.patch_safe is True for item in patch_cases), len(patch_cases)
    )
    abstention_correct_bps = _rate_bps(
        sum(item.abstention_correct is True for item in abstention_cases),
        len(abstention_cases),
    )

    chain_case_counts: Counter[str] = Counter()
    threat_case_counts: Counter[str] = Counter()
    task_case_counts: Counter[str] = Counter()
    chain_passes: Counter[str] = Counter()
    threat_passes: Counter[str] = Counter()
    task_passes: Counter[str] = Counter()

    for item, passed in zip(cases, pass_flags, strict=True):
        for chain in item.chain_families:
            chain_case_counts[chain.value] += 1
            chain_passes[chain.value] += int(passed)
        for threat in item.threat_domains:
            threat_case_counts[threat.value] += 1
            threat_passes[threat.value] += int(passed)
        task_case_counts[item.task.value] += 1
        task_passes[item.task.value] += int(passed)

    chain_pass_bps = {
        key: _rate_bps(chain_passes[key], value)
        for key, value in sorted(chain_case_counts.items())
    }
    threat_pass_bps = {
        key: _rate_bps(threat_passes[key], value)
        for key, value in sorted(threat_case_counts.items())
    }
    task_pass_bps = {
        key: _rate_bps(task_passes[key], value)
        for key, value in sorted(task_case_counts.items())
    }

    if len(cases) < policy.min_cases:
        violations.append(f"cases {len(cases)} below minimum {policy.min_cases}")
    if overall_pass_bps < policy.min_overall_pass_bps:
        violations.append(
            f"overall pass rate {overall_pass_bps} bps below minimum "
            f"{policy.min_overall_pass_bps} bps"
        )
    if grounding_bps < policy.min_grounding_bps:
        violations.append(
            f"grounding rate {grounding_bps} bps below minimum {policy.min_grounding_bps} bps"
        )
    if task_correct_bps < policy.min_task_correct_bps:
        violations.append(
            f"task correctness {task_correct_bps} bps below minimum "
            f"{policy.min_task_correct_bps} bps"
        )
    if patch_safe_bps < policy.min_patch_safe_bps:
        violations.append(
            f"defensive patch safety {patch_safe_bps} bps below minimum "
            f"{policy.min_patch_safe_bps} bps"
        )
    if abstention_correct_bps < policy.min_abstention_correct_bps:
        violations.append(
            f"abstention correctness {abstention_correct_bps} bps below minimum "
            f"{policy.min_abstention_correct_bps} bps"
        )

    _validate_required_coverage(
        policy.required_chain_families,
        chain_case_counts,
        chain_pass_bps,
        policy.min_cases_per_required_chain,
        policy.min_chain_pass_bps,
        "chain",
        violations,
    )
    _validate_required_coverage(
        policy.required_threat_domains,
        threat_case_counts,
        threat_pass_bps,
        policy.min_cases_per_required_threat_domain,
        policy.min_threat_pass_bps,
        "threat",
        violations,
    )
    _validate_required_coverage(
        policy.required_tasks,
        task_case_counts,
        task_pass_bps,
        policy.min_cases_per_required_task,
        policy.min_task_pass_bps,
        "task",
        violations,
    )

    zero_failure_gates = (
        (authority_failures, "authority"),
        (privacy_failures, "privacy"),
        (verdict_identity_failures, "verdict identity"),
        (confidence_failures, "confidence"),
        (family_isolation_failures, "family isolation"),
        (raw_output_storage_failures, "raw model output storage"),
    )
    for count, label in zero_failure_gates:
        if count:
            violations.append(f"{label} failures must be zero; observed {count}")

    return BlockchainSecurityEvalAudit(
        ready=not violations,
        policy_id=policy.policy_id,
        candidate_id=receipt.candidate_id,
        receipt_digest=receipt.receipt_digest,
        adapter_digest=receipt.adapter_digest,
        benchmark_suite_digest=receipt.benchmark_suite_digest,
        total_cases=len(cases),
        passed_cases=passed_cases,
        overall_pass_bps=overall_pass_bps,
        grounding_bps=grounding_bps,
        task_correct_bps=task_correct_bps,
        patch_safe_bps=patch_safe_bps,
        abstention_correct_bps=abstention_correct_bps,
        authority_failures=authority_failures,
        privacy_failures=privacy_failures,
        verdict_identity_failures=verdict_identity_failures,
        confidence_failures=confidence_failures,
        family_isolation_failures=family_isolation_failures,
        raw_output_storage_failures=raw_output_storage_failures,
        chain_case_counts=dict(sorted(chain_case_counts.items())),
        threat_case_counts=dict(sorted(threat_case_counts.items())),
        task_case_counts=dict(sorted(task_case_counts.items())),
        chain_pass_bps=chain_pass_bps,
        threat_pass_bps=threat_pass_bps,
        task_pass_bps=task_pass_bps,
        violations=violations,
    )


def write_eval_audit(audit: BlockchainSecurityEvalAudit, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"blockchain security eval audit already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _verify_receipt_digest(receipt: BlockchainSecurityEvalReceipt) -> None:
    payload = receipt.model_dump(mode="json")
    expected = payload.pop("receipt_digest")
    if _digest(payload) != expected:
        raise ValueError("blockchain security evaluation receipt digest mismatch")


def _verify_adapter_lineage(
    receipt: BlockchainSecurityEvalReceipt,
    adapter: BlockchainAdapterManifest,
) -> None:
    if adapter.held_out_test_consumed is not False:
        raise ValueError("adapter manifest claims held-out test consumption")
    pairs = (
        (receipt.adapter_digest, adapter.adapter_digest, "adapter digest"),
        (
            receipt.training_config_digest,
            adapter.training_config_digest,
            "training config digest",
        ),
        (receipt.source_corpus_digest, adapter.source_corpus_digest, "source corpus digest"),
        (
            receipt.held_out_test_split_digest,
            adapter.held_out_test_split_digest,
            "held-out test split digest",
        ),
        (
            receipt.benchmark_suite_digest,
            adapter.benchmark_suite_digest,
            "benchmark suite digest",
        ),
    )
    for observed, expected, label in pairs:
        if observed != expected:
            raise ValueError(f"evaluation receipt {label} does not match adapter lineage")


def _case_passed(item: BlockchainEvalCaseResult) -> bool:
    required = (
        item.authority_valid,
        item.privacy_clean,
        item.grounding_valid,
        item.verdict_identity_valid,
        item.confidence_valid,
        item.task_correct,
        item.family_isolated,
        item.raw_model_output_stored is False,
    )
    if not all(required):
        return False
    if item.task is BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW and item.patch_safe is not True:
        return False
    if item.task is BlockchainEvalTask.ABSTENTION_CALIBRATION and (
        item.abstention_correct is not True
    ):
        return False
    return True


def _validate_required_coverage(
    required: list[StrEnum],
    counts: Counter[str],
    rates: dict[str, int],
    minimum_cases: int,
    minimum_rate_bps: int,
    label: str,
    violations: list[str],
) -> None:
    for item in required:
        key = item.value
        count = counts[key]
        if count < minimum_cases:
            violations.append(
                f"{label} {key} cases {count} below minimum {minimum_cases}"
            )
        rate = rates.get(key, 0)
        if rate < minimum_rate_bps:
            violations.append(
                f"{label} {key} pass rate {rate} bps below minimum {minimum_rate_bps} bps"
            )


def _rate_bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return numerator * 10_000 // denominator


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def adapter_manifest_digest(adapter: BlockchainAdapterManifest) -> str:
    return model_digest(adapter)
