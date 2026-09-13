from __future__ import annotations

from dataclasses import dataclass

TARGET_TOTAL_B = 397.0
TARGET_ACTIVE_B = 35.0


@dataclass(frozen=True)
class TopologyCandidate:
    expert_count: int
    experts_per_token: int
    expert_parameters_each_billion: float
    dense_shared_parameters_billion: float
    routing_fraction: float
    exact_total_billion: float
    exact_active_billion: float
    status: str = "research_candidate"


def solve_exact(expert_count: int, experts_per_token: int) -> TopologyCandidate:
    """Solve parameter-budget equations for an exact 397B-total / 35B-active sparse MoE.

    This proves parameter arithmetic only. It does not prove that a transformer with
    these dimensions exists, is numerically stable, or is trainable on available hardware.
    """
    if expert_count <= 1:
        raise ValueError("expert_count must be greater than 1")
    if experts_per_token <= 0 or experts_per_token >= expert_count:
        raise ValueError("experts_per_token must be positive and smaller than expert_count")

    expert_b = (TARGET_TOTAL_B - TARGET_ACTIVE_B) / (expert_count - experts_per_token)
    dense_b = TARGET_ACTIVE_B - experts_per_token * expert_b
    if expert_b <= 0 or dense_b <= 0:
        raise ValueError("topology has no positive exact parameter-budget solution")

    total_b = dense_b + expert_count * expert_b
    active_b = dense_b + experts_per_token * expert_b
    return TopologyCandidate(
        expert_count=expert_count,
        experts_per_token=experts_per_token,
        expert_parameters_each_billion=expert_b,
        dense_shared_parameters_billion=dense_b,
        routing_fraction=experts_per_token / expert_count,
        exact_total_billion=total_b,
        exact_active_billion=active_b,
    )


def enumerate_candidates() -> list[TopologyCandidate]:
    """Enumerate reference-inspired research candidates without promoting any to production."""
    reference_shapes = (
        (128, 8),
        (160, 8),
        (192, 8),
        (256, 8),
        (384, 8),
        (512, 8),
        (160, 16),
        (256, 16),
        (384, 16),
    )
    candidates: list[TopologyCandidate] = []
    for expert_count, top_k in reference_shapes:
        try:
            candidates.append(solve_exact(expert_count, top_k))
        except ValueError:
            continue
    return sorted(candidates, key=lambda c: (abs(c.routing_fraction - 0.05), -c.dense_shared_parameters_billion))
