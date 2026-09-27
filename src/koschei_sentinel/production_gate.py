from __future__ import annotations

from dataclasses import dataclass

from .conformance import ConformanceReport


@dataclass(frozen=True)
class ProductionThresholds:
    min_cases: int = 50
    min_precision: float = 0.98
    min_recall: float = 0.95
    min_containment_success_rate: float = 0.99
    require_clean_cases: bool = True
    require_adversarial_cases: bool = True
    require_ci_pass: bool = True
    require_tenant_isolation_pass: bool = True
    require_recovery_pass: bool = True
    require_live_inference_pass: bool = True


@dataclass(frozen=True)
class ProductionEvidence:
    clean_cases: int
    adversarial_cases: int
    ci_pass: bool
    tenant_isolation_pass: bool
    recovery_pass: bool
    live_inference_pass: bool


@dataclass(frozen=True)
class ProductionGateResult:
    ready: bool
    failures: tuple[str, ...]


def evaluate_production_readiness(
    report: ConformanceReport,
    evidence: ProductionEvidence,
    thresholds: ProductionThresholds = ProductionThresholds(),
) -> ProductionGateResult:
    failures: list[str] = []
    if report.case_count < thresholds.min_cases:
        failures.append("insufficient conformance corpus")
    if report.precision < thresholds.min_precision:
        failures.append("precision below production threshold")
    if report.recall < thresholds.min_recall:
        failures.append("recall below production threshold")
    if report.containment_success_rate < thresholds.min_containment_success_rate:
        failures.append("containment success below production threshold")
    if thresholds.require_clean_cases and evidence.clean_cases < 1:
        failures.append("clean-case coverage missing")
    if thresholds.require_adversarial_cases and evidence.adversarial_cases < 1:
        failures.append("adversarial coverage missing")
    if thresholds.require_ci_pass and not evidence.ci_pass:
        failures.append("CI evidence missing or failing")
    if thresholds.require_tenant_isolation_pass and not evidence.tenant_isolation_pass:
        failures.append("tenant isolation evidence missing or failing")
    if thresholds.require_recovery_pass and not evidence.recovery_pass:
        failures.append("recovery evidence missing or failing")
    if thresholds.require_live_inference_pass and not evidence.live_inference_pass:
        failures.append("live inference evidence missing or failing")
    return ProductionGateResult(not failures, tuple(failures))
