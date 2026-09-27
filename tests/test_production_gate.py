from koschei_sentinel.conformance import ConformanceCaseResult, build_report
from koschei_sentinel.production_gate import ProductionEvidence, ProductionThresholds, evaluate_production_readiness


def report(count=50):
    results = tuple(
        ConformanceCaseResult(f"c{i}", frozenset({f"f{i}"}), frozenset({f"f{i}"}), True, True)
        for i in range(count)
    )
    return build_report(corpus_id="security", corpus_version="1", engine_id="engine",
        engine_version="1", run_date="2026-09-27", results=results)


def evidence(**overrides):
    values = dict(clean_cases=10, adversarial_cases=10, ci_pass=True,
        tenant_isolation_pass=True, recovery_pass=True, live_inference_pass=True)
    values.update(overrides)
    return ProductionEvidence(**values)


def test_all_required_evidence_can_open_gate():
    result = evaluate_production_readiness(report(), evidence())
    assert result.ready is True
    assert result.failures == ()


def test_missing_live_inference_keeps_gate_closed():
    result = evaluate_production_readiness(report(), evidence(live_inference_pass=False))
    assert result.ready is False
    assert "live inference evidence missing or failing" in result.failures


def test_small_corpus_keeps_gate_closed():
    result = evaluate_production_readiness(report(10), evidence())
    assert result.ready is False
    assert "insufficient conformance corpus" in result.failures


def test_missing_isolation_or_ci_keeps_gate_closed():
    result = evaluate_production_readiness(report(), evidence(ci_pass=False, tenant_isolation_pass=False))
    assert result.ready is False
    assert "CI evidence missing or failing" in result.failures
    assert "tenant isolation evidence missing or failing" in result.failures
