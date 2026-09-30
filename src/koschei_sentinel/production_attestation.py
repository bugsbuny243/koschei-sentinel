from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .conformance import ConformanceReport
from .production_evidence import ProductionEvidenceManifest, REQUIRED_PRODUCTION_ARTIFACTS
from .production_gate import ProductionEvidence, ProductionGateResult, ProductionThresholds, evaluate_production_readiness


@dataclass(frozen=True)
class ProductionAttestation:
    commit_sha: str
    manifest_digest: str
    conformance_digest: str
    ready: bool
    failures: tuple[str, ...]
    schema_version: str = "sentinel.production-attestation.v1"

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def attest_production_readiness(
    *,
    report: ConformanceReport,
    manifest: ProductionEvidenceManifest,
    evidence: ProductionEvidence,
    thresholds: ProductionThresholds = ProductionThresholds(),
) -> ProductionAttestation:
    failures: list[str] = []
    if manifest.conformance_report_digest != report.digest():
        failures.append("conformance digest mismatch")
    try:
        manifest.require_verified(REQUIRED_PRODUCTION_ARTIFACTS)
    except ValueError as exc:
        failures.append(str(exc))

    gate: ProductionGateResult = evaluate_production_readiness(report, evidence, thresholds)
    failures.extend(gate.failures)
    failures = list(dict.fromkeys(failures))
    return ProductionAttestation(
        commit_sha=manifest.commit_sha,
        manifest_digest=manifest.digest(),
        conformance_digest=report.digest(),
        ready=not failures,
        failures=tuple(failures),
    )
