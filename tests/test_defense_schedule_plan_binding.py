import pytest

from tests.test_defense_resource_scheduler import _plan, _policy, _scope_for

from koschei_sentinel.defense_resource_scheduler import build_defense_resource_schedule
from koschei_sentinel.defense_wave_execution import start_defense_wave_execution
from koschei_sentinel.scheduled_defense_connector import (
    build_scheduled_assured_connector_envelope,
)


def test_schedule_cannot_be_replayed_against_different_assured_plan() -> None:
    original = _plan(critical=["wallet:a"])
    different = _plan(critical=["wallet:a", "wallet:b"])
    assert original.graph_id == different.graph_id

    schedule = build_defense_resource_schedule(
        original,
        policy=_policy(total=1, signer=1, reserved=1),
    )
    item = schedule.scheduled[0]

    with pytest.raises(ValueError, match="different assured multi-incident plan"):
        start_defense_wave_execution(schedule, different)

    with pytest.raises(ValueError, match="different assured multi-incident plan"):
        build_scheduled_assured_connector_envelope(
            schedule=schedule,
            multi_plan=different,
            component_id=item.component_id,
            scope=_scope_for(item),
            precondition_evidence_ids=["evidence:stale-plan-replay"],
            dry_run=True,
        )
