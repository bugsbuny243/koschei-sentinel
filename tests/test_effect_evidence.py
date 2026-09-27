import pytest

from koschei_sentinel.effect_evidence import EffectAgreement, ObservedEffect, require_verified_effect


def effect(state):
    return ObservedEffect(
        action_id="a1", tenant_id="t1", target_id="workload-1",
        observer_id="observer-1", expected_state="isolated",
        observed_state=state, evidence_digest="sha256:e1", metadata={},
    )


def test_matching_observation_is_verified():
    observed = effect("isolated")
    assert observed.agreement is EffectAgreement.AGREE
    require_verified_effect(observed)
    assert observed.digest().startswith("sha256:")


def test_contradictory_observation_fails_closed():
    observed = effect("reachable")
    assert observed.agreement is EffectAgreement.DISAGREE
    with pytest.raises(RuntimeError, match="contradicts"):
        require_verified_effect(observed)


def test_missing_observation_fails_closed():
    observed = effect(None)
    assert observed.agreement is EffectAgreement.UNKNOWN
    with pytest.raises(RuntimeError, match="unobserved"):
        require_verified_effect(observed)
