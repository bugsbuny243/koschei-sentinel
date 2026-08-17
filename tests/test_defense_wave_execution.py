from tests.test_defense_resource_scheduler import _plan, _policy

from koschei_sentinel.defense_resource_scheduler import build_defense_resource_schedule
from koschei_sentinel.defense_wave_execution import (
    DefenseWaveComponentStatus,
    authorize_wave_component,
    record_wave_component_execution,
    start_defense_wave_execution,
    verify_wave_component_outcome,
)


def test_parallel_components_can_be_authorized_independently() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"])
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=2, signer=2, reserved=1),
    )
    assert len(schedule.scheduled) == 2

    wave = start_defense_wave_execution(schedule, plan)
    first, second = [row.component_id for row in wave.components]
    wave = authorize_wave_component(
        wave,
        component_id=first,
        precondition_evidence_ids=["evidence:first"],
    )
    wave = authorize_wave_component(
        wave,
        component_id=second,
        precondition_evidence_ids=["evidence:second"],
    )

    by_id = {row.component_id: row for row in wave.components}
    assert by_id[first].status is DefenseWaveComponentStatus.AUTHORIZED
    assert by_id[second].status is DefenseWaveComponentStatus.AUTHORIZED
    assert wave.complete is False


def test_wave_completes_only_after_every_scheduled_component_is_verified() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"])
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=2, signer=2, reserved=1),
    )
    wave = start_defense_wave_execution(schedule, plan)
    components = [row.component_id for row in wave.components]

    for component_id in components:
        wave = authorize_wave_component(
            wave,
            component_id=component_id,
            precondition_evidence_ids=[f"evidence:{component_id}"],
        )
        wave = record_wave_component_execution(
            wave,
            component_id=component_id,
            execution_receipt_ids=[f"receipt:{component_id}"],
        )
        wave = verify_wave_component_outcome(
            wave,
            component_id=component_id,
            succeeded=True,
            outcome_evidence_ids=[f"outcome:{component_id}"],
        )
        if component_id != components[-1]:
            assert wave.complete is False

    assert wave.complete is True
    assert wave.failed_component_ids == []
    assert wave.succeeded_component_ids == sorted(components)


def test_failed_component_does_not_block_other_component_verification() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"])
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=2, signer=2, reserved=1),
    )
    wave = start_defense_wave_execution(schedule, plan)
    first, second = [row.component_id for row in wave.components]

    for component_id, succeeded in ((first, False), (second, True)):
        wave = authorize_wave_component(
            wave,
            component_id=component_id,
            precondition_evidence_ids=[f"evidence:{component_id}"],
        )
        wave = record_wave_component_execution(
            wave,
            component_id=component_id,
            execution_receipt_ids=[f"receipt:{component_id}"],
        )
        wave = verify_wave_component_outcome(
            wave,
            component_id=component_id,
            succeeded=succeeded,
            outcome_evidence_ids=[f"outcome:{component_id}:{int(succeeded)}"],
        )

    assert wave.complete is True
    assert wave.failed_component_ids == [first]
    assert wave.succeeded_component_ids == [second]


def test_wave_exposes_only_scheduled_first_step_per_component() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"])
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=1, signer=1, reserved=1),
    )
    wave = start_defense_wave_execution(schedule, plan)
    assert len(wave.components) == 1
    row = wave.components[0]

    assert row.schedule_item.step_id == row.interception_plan.steps[0].step_id
    assert row.schedule_item.action is row.interception_plan.steps[0].action
    assert row.schedule_item.target_entity_id == row.interception_plan.steps[0].target_entity_id
