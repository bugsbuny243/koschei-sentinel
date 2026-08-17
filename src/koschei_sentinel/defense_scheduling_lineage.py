from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.attack_world_lines import AttackWorldLineTimeline
from koschei_sentinel.defense_resource_scheduler import (
    DefenseSchedulerState,
    DefenseSchedulingContext,
)
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class SchedulerLineageCarryReceipt(StrictModel):
    schema_version: Literal["sentinel.scheduler-lineage-carry-receipt.v1"] = (
        "sentinel.scheduler-lineage-carry-receipt.v1"
    )
    timeline_sha256: str = Field(pattern=_DIGEST)
    from_tick: int = Field(ge=0)
    to_tick: int = Field(ge=1)
    inherited_predecessors_by_subject: dict[str, list[str]]
    carried_state: DefenseSchedulerState
    carry_sha256: str = Field(pattern=_DIGEST)


def _context_digest(
    *,
    graph_id: str,
    timeline_sha256: str,
    tick: int,
    mapping: dict[str, str],
) -> str:
    payload = {
        "graph_id": graph_id,
        "timeline_sha256": timeline_sha256,
        "tick": tick,
        "mapping": dict(sorted(mapping.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_world_line_scheduling_context(
    plan: AssuredMultiIncidentDefensePlan,
    timeline: AttackWorldLineTimeline,
    *,
    tick: int,
) -> DefenseSchedulingContext:
    if tick < 0 or tick >= timeline.ticks:
        raise ValueError("scheduler world-line context tick is outside the timeline")
    observations = [row for row in timeline.observations if row.tick == tick]
    mapping = {row.component_id: row.world_line_id for row in observations}
    if len(mapping) != len(observations):
        raise ValueError("world-line timeline contains duplicate component observations at tick")
    plan_components = {row.component_id for row in plan.component_plans}
    if set(mapping) != plan_components:
        raise ValueError(
            "world-line observations must map every assured active component exactly once"
        )
    digest = _context_digest(
        graph_id=plan.graph_id,
        timeline_sha256=timeline.timeline_sha256,
        tick=tick,
        mapping=mapping,
    )
    return DefenseSchedulingContext(
        context_id=f"world-line-context:{digest[:24]}",
        scheduling_subject_by_component=dict(sorted(mapping.items())),
        world_line_timeline_sha256=timeline.timeline_sha256,
        tick=tick,
    )


def _carry_digest(
    *,
    timeline_sha256: str,
    from_tick: int,
    to_tick: int,
    inherited: dict[str, list[str]],
    state: DefenseSchedulerState,
) -> str:
    payload = {
        "timeline_sha256": timeline_sha256,
        "from_tick": from_tick,
        "to_tick": to_tick,
        "inherited": dict(sorted(inherited.items())),
        "state": state.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def carry_scheduler_state_across_world_lines(
    previous_state: DefenseSchedulerState,
    timeline: AttackWorldLineTimeline,
    *,
    from_tick: int,
    to_tick: int,
) -> SchedulerLineageCarryReceipt:
    if to_tick != from_tick + 1:
        raise ValueError("scheduler lineage carry currently requires adjacent ticks")
    if from_tick < 0 or to_tick >= timeline.ticks:
        raise ValueError("scheduler lineage carry ticks are outside the timeline")

    current_subjects = {
        row.world_line_id for row in timeline.observations if row.tick == to_tick
    }
    transitions = [row for row in timeline.transitions if row.to_tick == to_tick]
    predecessors_by_successor: dict[str, set[str]] = {
        subject: set() for subject in current_subjects
    }
    for transition in transitions:
        for successor in transition.successor_world_line_ids:
            if successor in predecessors_by_successor:
                predecessors_by_successor[successor].update(
                    transition.predecessor_world_line_ids
                )

    carried: dict[str, int] = {}
    inherited: dict[str, list[str]] = {}
    for subject in sorted(current_subjects):
        if subject in previous_state.wait_cycles_by_subject:
            carried[subject] = previous_state.wait_cycles_by_subject[subject]
            inherited[subject] = [subject]
            continue
        predecessors = sorted(predecessors_by_successor.get(subject, set()))
        inherited[subject] = predecessors
        carried[subject] = max(
            (
                previous_state.wait_cycles_by_subject.get(predecessor, 0)
                for predecessor in predecessors
            ),
            default=0,
        )

    state = DefenseSchedulerState(wait_cycles_by_subject=carried)
    digest = _carry_digest(
        timeline_sha256=timeline.timeline_sha256,
        from_tick=from_tick,
        to_tick=to_tick,
        inherited=inherited,
        state=state,
    )
    return SchedulerLineageCarryReceipt(
        timeline_sha256=timeline.timeline_sha256,
        from_tick=from_tick,
        to_tick=to_tick,
        inherited_predecessors_by_subject={
            key: inherited[key] for key in sorted(inherited)
        },
        carried_state=state,
        carry_sha256=digest,
    )
