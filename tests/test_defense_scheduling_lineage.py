from koschei_sentinel.attack_world_lines import (
    AttackWorldLinePolicy,
    AttackWorldLineTimeline,
    AttackWorldLineTransition,
    WorldLineComponentObservation,
    WorldLineTransitionType,
)
from koschei_sentinel.defense_resource_scheduler import DefenseSchedulerState
from koschei_sentinel.defense_scheduling_lineage import (
    carry_scheduler_state_across_world_lines,
)


def _observation(tick: int, line: str, component: str) -> WorldLineComponentObservation:
    return WorldLineComponentObservation(
        tick=tick,
        world_line_id=line,
        component_id=component,
        entity_ids=[f"entity:{component}"],
        relation_ids=[f"relation:{component}"],
        protected_anchor_ids=[],
        current_stage="LATERAL_MOVEMENT",
        risk_score=0.8,
    )


def _timeline(
    *,
    current_lines: list[tuple[str, str]],
    transition_type: WorldLineTransitionType,
    predecessors: list[str],
) -> AttackWorldLineTimeline:
    observations = [_observation(0, "line:parent", "component:parent")]
    observations.extend(
        _observation(1, line, component) for line, component in current_lines
    )
    successors = [line for line, _ in current_lines]
    successor_components = [component for _, component in current_lines]
    transition = AttackWorldLineTransition(
        from_tick=0,
        to_tick=1,
        transition_type=transition_type,
        predecessor_world_line_ids=predecessors,
        successor_world_line_ids=successors,
        predecessor_component_ids=(
            ["component:parent"] if predecessors else []
        ),
        successor_component_ids=successor_components,
        shared_protected_anchor_ids=[],
        max_overlap_score=0.8 if predecessors else 0.0,
    )
    return AttackWorldLineTimeline(
        stream_id="stream:scheduler-lineage",
        ticks=2,
        protected_anchor_entity_ids=[],
        policy=AttackWorldLinePolicy(),
        observations=observations,
        transitions=[transition],
        timeline_sha256="a" * 64,
    )


def test_split_successors_inherit_parent_wait_age() -> None:
    timeline = _timeline(
        current_lines=[
            ("line:child-a", "component:child-a"),
            ("line:child-b", "component:child-b"),
        ],
        transition_type=WorldLineTransitionType.SPLIT,
        predecessors=["line:parent"],
    )
    receipt = carry_scheduler_state_across_world_lines(
        DefenseSchedulerState(wait_cycles_by_subject={"line:parent": 6}),
        timeline,
        from_tick=0,
        to_tick=1,
    )

    assert receipt.carried_state.wait_cycles_by_subject["line:child-a"] == 6
    assert receipt.carried_state.wait_cycles_by_subject["line:child-b"] == 6
    assert receipt.inherited_predecessors_by_subject["line:child-a"] == ["line:parent"]


def test_new_world_line_starts_with_zero_wait_age() -> None:
    timeline = _timeline(
        current_lines=[("line:new", "component:new")],
        transition_type=WorldLineTransitionType.NEW,
        predecessors=[],
    )
    receipt = carry_scheduler_state_across_world_lines(
        DefenseSchedulerState(wait_cycles_by_subject={"line:parent": 9}),
        timeline,
        from_tick=0,
        to_tick=1,
    )

    assert receipt.carried_state.wait_cycles_by_subject == {"line:new": 0}
    assert receipt.inherited_predecessors_by_subject["line:new"] == []


def test_lineage_carry_is_deterministic() -> None:
    timeline = _timeline(
        current_lines=[("line:child", "component:child")],
        transition_type=WorldLineTransitionType.RECONFIGURED,
        predecessors=["line:parent"],
    )
    state = DefenseSchedulerState(wait_cycles_by_subject={"line:parent": 3})
    first = carry_scheduler_state_across_world_lines(
        state,
        timeline,
        from_tick=0,
        to_tick=1,
    )
    second = carry_scheduler_state_across_world_lines(
        state,
        timeline,
        from_tick=0,
        to_tick=1,
    )

    assert first.model_dump() == second.model_dump()
    assert first.carry_sha256 == second.carry_sha256
