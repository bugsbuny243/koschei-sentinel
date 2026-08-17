from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field

from koschei_sentinel.active_defense_planner import ActiveDefensePlan, build_active_defense_plan
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.interception_execution import InterceptionExecution
from koschei_sentinel.interception_planner import InterceptionPlan, build_interception_plan
from koschei_sentinel.models import StrictModel


class ReassessmentDisposition(StrEnum):
    CONTINUE_CONTAINMENT = "CONTINUE_CONTAINMENT"
    HUNT_ALTERNATE_PATHS = "HUNT_ALTERNATE_PATHS"
    RECOVERY_CANDIDATE = "RECOVERY_CANDIDATE"


class AdaptiveDefenseReassessment(StrictModel):
    schema_version: Literal["sentinel.adaptive-defense-reassessment.v1"] = (
        "sentinel.adaptive-defense-reassessment.v1"
    )
    previous_graph_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    updated_graph_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    graph_changed: bool
    previous_interception_complete: bool
    previous_path_contained: bool
    disposition: ReassessmentDisposition
    active_defense_plan: ActiveDefensePlan
    interception_plan: InterceptionPlan
    rationale: list[str]


def graph_fingerprint(graph: CyberStateGraph) -> str:
    payload = graph.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reassess_after_interception(
    *,
    previous_graph: CyberStateGraph,
    updated_graph: CyberStateGraph,
    previous_execution: InterceptionExecution,
    critical_entity_ids: list[str] | None = None,
) -> AdaptiveDefenseReassessment:
    if previous_graph.graph_id != updated_graph.graph_id:
        raise ValueError("adaptive reassessment requires the same incident graph_id")
    if previous_execution.graph_id != previous_graph.graph_id:
        raise ValueError("previous interception execution belongs to a different graph")

    previous_sha = graph_fingerprint(previous_graph)
    updated_sha = graph_fingerprint(updated_graph)
    changed = previous_sha != updated_sha

    active = build_active_defense_plan(
        updated_graph,
        critical_entity_ids=critical_entity_ids or [],
    )
    interception = build_interception_plan(active)

    credible_stage_present = active.progression.current_stage is not None
    active_relations_present = bool(active.progression.active_relation_ids)
    new_containment_steps = bool(interception.steps)

    rationale = [
        "authorization is recomputed from the updated Cyber State Graph; prior permissions are not inherited",
        "predictions remain predictions and cannot satisfy evidence requirements by themselves",
    ]

    if not changed:
        rationale.append("graph telemetry has not changed since the prior interception cycle")

    if credible_stage_present and new_containment_steps:
        disposition = ReassessmentDisposition.CONTINUE_CONTAINMENT
        rationale.append("credible hostile progression remains and new authorized cut points exist")
    elif credible_stage_present or active_relations_present:
        disposition = ReassessmentDisposition.HUNT_ALTERNATE_PATHS
        rationale.append(
            "hostile evidence remains but no immediate authorized cut point is available; hunt for alternate or displaced paths"
        )
    else:
        disposition = ReassessmentDisposition.RECOVERY_CANDIDATE
        rationale.append(
            "no credible active progression remains in the updated graph; incident may advance toward recovery after continued observation"
        )

    return AdaptiveDefenseReassessment(
        previous_graph_sha256=previous_sha,
        updated_graph_sha256=updated_sha,
        graph_changed=changed,
        previous_interception_complete=previous_execution.complete,
        previous_path_contained=previous_execution.contained,
        disposition=disposition,
        active_defense_plan=active,
        interception_plan=interception,
        rationale=rationale,
    )
