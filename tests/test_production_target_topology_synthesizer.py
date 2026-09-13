from __future__ import annotations

import pytest

from koschei_sentinel.production_target_topology_synthesizer import enumerate_candidates, solve_exact


def test_qwen_coder_inspired_shape_hits_exact_parameter_budget() -> None:
    candidate = solve_exact(160, 8)
    assert candidate.exact_total_billion == pytest.approx(397.0)
    assert candidate.exact_active_billion == pytest.approx(35.0)
    assert candidate.expert_parameters_each_billion == pytest.approx(362 / 152)
    assert candidate.dense_shared_parameters_billion == pytest.approx(35 - 8 * (362 / 152))
    assert candidate.status == "research_candidate"


def test_kimi_inspired_shape_hits_exact_parameter_budget() -> None:
    candidate = solve_exact(384, 8)
    assert candidate.exact_total_billion == pytest.approx(397.0)
    assert candidate.exact_active_billion == pytest.approx(35.0)
    assert candidate.expert_parameters_each_billion == pytest.approx(362 / 376)
    assert candidate.dense_shared_parameters_billion == pytest.approx(35 - 8 * (362 / 376))


def test_invalid_router_shape_fails_closed() -> None:
    with pytest.raises(ValueError):
        solve_exact(8, 8)


def test_enumeration_never_claims_verified_or_trainable() -> None:
    candidates = enumerate_candidates()
    assert candidates
    assert all(candidate.status == "research_candidate" for candidate in candidates)
    assert all(candidate.exact_total_billion == pytest.approx(397.0) for candidate in candidates)
    assert all(candidate.exact_active_billion == pytest.approx(35.0) for candidate in candidates)
