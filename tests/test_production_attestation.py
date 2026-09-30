from koschei_sentinel.conformance import ConformanceCaseResult, build_report
from koschei_sentinel.production_attestation import attest_production_readiness
from koschei_sentinel.production_evidence import EvidenceArtifact, ProductionEvidenceManifest, REQUIRED_PRODUCTION_ARTIFACTS
from koschei_sentinel.production_gate import ProductionEvidence


def report():
    cases = tuple(ConformanceCaseResult(f"c{i}", frozenset({f"f{i}"}), frozenset({f"f{i}"}), True, True) for i in range(50))
    return build_report(corpus_id="security", corpus_version="1", engine_id="engine", engine_version="1", run_date="2026-09-30", results=cases)


def manifest(r, digest=None):
    artifacts = tuple(EvidenceArtifact(k, f"sha256:{k}", "lab", True) for k in sorted(REQUIRED_PRODUCTION_ARTIFACTS))
    return ProductionEvidenceManifest("commit", "security", "1", "sha256:corpus", "engine", "1", "2026-09-30", digest or r.digest(), artifacts)


def evidence():
    return ProductionEvidence(10, 10, True, True, True, True)


def test_verified_evidence_can_produce_ready_attestation():
    r = report()
    a = attest_production_readiness(report=r, manifest=manifest(r), evidence=evidence())
    assert a.ready is True
    assert a.failures == ()
    assert a.digest().startswith("sha256:")


def test_conformance_digest_mismatch_blocks_attestation():
    r = report()
    a = attest_production_readiness(report=r, manifest=manifest(r, "sha256:wrong"), evidence=evidence())
    assert a.ready is False
    assert "conformance digest mismatch" in a.failures
