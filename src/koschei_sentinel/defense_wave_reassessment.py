from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from koschei_sentinel.assured_active_defense import ActiveDefenseAssurancePolicy
from koschei_sentinel.assured_multi_incident_defense import (
    AssuredMultiIncidentDefensePlan,
    build_assured_multi_incident_defense_plan,
)
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourceSchedule,
    assured_multi_incident_plan_sha256,
    verify_defense_resource_schedule,
)
from koschei_sentinel.defense_wave_execution import (
    DefenseWaveExecution,
    verify_defense_wave_execution,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.perception_assurance import PerceptionAssuranceSummary
from koschei_sentinel.perception_graph_binding import BoundPerceptionGraph
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceRegistry,
    perception_registry_sha256,
)

_DIGEST = r"^[a-f0-9]{64}$"


class DefenseWaveReassessmentReceipt(StrictModel):
    schema_version: Literal["sentinel.defense-wave-reassessment-receipt.v1"] = (
        "sentinel.defense-wave-reassessment-receipt.v1"
    )
    previous_wave_id: str
    previous_wave_sha256: str = Field(pattern=_DIGEST)
    previous_schedule_sha256: str = Field(pattern=_DIGEST)
    previous_plan_sha256: str = Field(pattern=_DIGEST)
    previous_source_batch_sha256: str = Field(pattern=_DIGEST)
    previous_graph_sha256: str = Field(pattern=_DIGEST)
    next_source_batch_sha256: str = Field(pattern=_DIGEST)
    next_graph_sha256: str = Field(pattern=_DIGEST)
    next_plan_sha256: str = Field(pattern=_DIGEST)
    fresh_perception: Literal[True] = True
    graph_changed: bool
    plan_changed: bool
    reassessment_sha256: str = Field(pattern=_DIGEST)


class DefenseWaveReassessment(StrictModel):
    schema_version: Literal["sentinel.defense-wave-reassessment.v1"] = (
        "sentinel.defense-wave-reassessment.v1"
    )
    next_plan: AssuredMultiIncidentDefensePlan
    receipt: DefenseWaveReassessmentReceipt


def _previous_binding(plan: AssuredMultiIncidentDefensePlan) -> tuple[str, str]:
    if not plan.component_plans:
        raise ValueError("previous defense plan has no assurance-bound attack components")
    source_batches = {
        row.assured_plan.assurance.source_batch_sha256 for row in plan.component_plans
    }
    graph_hashes = {
        row.assured_plan.assurance.graph_sha256 for row in plan.component_plans
    }
    if len(source_batches) != 1 or len(graph_hashes) != 1:
        raise ValueError("previous multi-incident plan components do not share one perception binding")
    return next(iter(source_batches)), next(iter(graph_hashes))


def _validate_next_perception(
    bound: BoundPerceptionGraph,
    assurance: PerceptionAssuranceSummary,
    registry: PerceptionSourceRegistry,
) -> None:
    if assurance.fused_batch_sha256 != bound.receipt.source_batch_sha256:
        raise ValueError("next perception assurance belongs to a different fused batch")
    if assurance.registry_id != registry.registry_id:
        raise ValueError("next perception assurance belongs to a different source registry")
    if assurance.registry_sha256 != perception_registry_sha256(registry):
        raise ValueError("next perception assurance registry digest is stale or mismatched")


def _receipt_digest(
    *,
    previous_wave_sha256: str,
    previous_schedule_sha256: str,
    previous_plan_sha256: str,
    previous_source_batch_sha256: str,
    previous_graph_sha256: str,
    next_source_batch_sha256: str,
    next_graph_sha256: str,
    next_plan_sha256: str,
) -> str:
    payload = "|".join(
        [
            previous_wave_sha256,
            previous_schedule_sha256,
            previous_plan_sha256,
            previous_source_batch_sha256,
            previous_graph_sha256,
            next_source_batch_sha256,
            next_graph_sha256,
            next_plan_sha256,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reassess_after_defense_wave(
    *,
    completed_wave: DefenseWaveExecution,
    previous_schedule: DefenseResourceSchedule,
    previous_plan: AssuredMultiIncidentDefensePlan,
    next_bound_graph: BoundPerceptionGraph,
    next_perception_assurance: PerceptionAssuranceSummary,
    registry: PerceptionSourceRegistry,
    critical_entity_ids: list[str] | None = None,
    assurance_policy: ActiveDefenseAssurancePolicy | None = None,
) -> DefenseWaveReassessment:
    verify_defense_wave_execution(completed_wave)
    if not completed_wave.complete:
        raise ValueError("defense wave must be complete before reassessment")
    verify_defense_resource_schedule(previous_schedule, previous_plan)
    if completed_wave.schedule_id != previous_schedule.schedule_id:
        raise ValueError("completed wave belongs to a different defense schedule")
    if completed_wave.schedule_sha256 != previous_schedule.schedule_sha256:
        raise ValueError("completed wave schedule digest does not match previous schedule")

    previous_source_batch, previous_graph_sha = _previous_binding(previous_plan)
    _validate_next_perception(
        next_bound_graph,
        next_perception_assurance,
        registry,
    )
    next_source_batch = next_bound_graph.receipt.source_batch_sha256
    if next_source_batch == previous_source_batch:
        raise ValueError(
            "post-wave reassessment requires a fresh fused perception batch; stale telemetry replay rejected"
        )

    next_plan = build_assured_multi_incident_defense_plan(
        next_bound_graph.graph,
        graph_receipt=next_bound_graph.receipt,
        perception_assurance=next_perception_assurance,
        registry=registry,
        critical_entity_ids=critical_entity_ids,
        policy=assurance_policy,
    )
    previous_plan_sha = assured_multi_incident_plan_sha256(previous_plan)
    next_plan_sha = assured_multi_incident_plan_sha256(next_plan)
    digest = _receipt_digest(
        previous_wave_sha256=completed_wave.wave_sha256,
        previous_schedule_sha256=previous_schedule.schedule_sha256,
        previous_plan_sha256=previous_plan_sha,
        previous_source_batch_sha256=previous_source_batch,
        previous_graph_sha256=previous_graph_sha,
        next_source_batch_sha256=next_source_batch,
        next_graph_sha256=next_bound_graph.receipt.graph_sha256,
        next_plan_sha256=next_plan_sha,
    )
    receipt = DefenseWaveReassessmentReceipt(
        previous_wave_id=completed_wave.wave_id,
        previous_wave_sha256=completed_wave.wave_sha256,
        previous_schedule_sha256=previous_schedule.schedule_sha256,
        previous_plan_sha256=previous_plan_sha,
        previous_source_batch_sha256=previous_source_batch,
        previous_graph_sha256=previous_graph_sha,
        next_source_batch_sha256=next_source_batch,
        next_graph_sha256=next_bound_graph.receipt.graph_sha256,
        next_plan_sha256=next_plan_sha,
        graph_changed=next_bound_graph.receipt.graph_sha256 != previous_graph_sha,
        plan_changed=next_plan_sha != previous_plan_sha,
        reassessment_sha256=digest,
    )
    return DefenseWaveReassessment(next_plan=next_plan, receipt=receipt)
