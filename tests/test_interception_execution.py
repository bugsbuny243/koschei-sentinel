import pytest

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.interception_execution import (
    InterceptionStepStatus,
    authorize_next_step,
    record_step_execution,
    start_interception_execution,
    verify_step_outcome,
)
from koschei_sentinel.interception_planner import (
    InterceptionPlan,
    InterceptionStep,
    InterceptionUrgency,
)


def _plan(*, stop_first: bool = False) -> InterceptionPlan:
    return InterceptionPlan(
        graph_id="graph:test",
        defense_mode=DefenseMode.COMBAT,
        predicted_impact_present=True,
        rationale=["test plan"],
        steps=[
            InterceptionStep(
                step_id="intercept:graph:test:1",
                sequence=1,
                action=DefenseActionType.FREEZE_SIGNER,
                target_entity_id="wallet:signer",
                urgency=InterceptionUrgency.IMMEDIATE,
                expected_effect_score=0.95 if stop_first else 0.8,
                supporting_relation_ids=["rel:signer"],
                verification_required=True,
                stop_if_verified=stop_first,
                objective="remove signing authority from hostile path",
            ),
            InterceptionStep(
                step_id="intercept:graph:test:2",
                sequence=2,
                action=DefenseActionType.PAUSE_PIPELINE,
                target_entity_id="pipeline:release",
                urgency=InterceptionUrgency.HIGH,
                expected_effect_score=0.75,
                supporting_relation_ids=["rel:pipeline"],
                verification_required=True,
                stop_if_verified=False,
                objective="stop compromised release propagation",
            ),
        ],
    )


def test_next_step_cannot_be_authorized_before_current_outcome_is_verified():
    plan = _plan()
    execution = start_interception_execution(plan)
    execution = authorize_next_step(
        execution,
        plan,
        precondition_evidence_ids=["evidence:attack"],
    )
    execution = record_step_execution(
        execution,
        plan,
        step_id=plan.steps[0].step_id,
        execution_receipt_ids=["receipt:freeze"],
    )

    with pytest.raises(ValueError, match="verified before another is authorized"):
        authorize_next_step(
            execution,
            plan,
            precondition_evidence_ids=["evidence:second"],
        )

    execution = verify_step_outcome(
        execution,
        plan,
        step_id=plan.steps[0].step_id,
        succeeded=False,
        outcome_evidence_ids=["evidence:freeze-failed"],
    )
    execution = authorize_next_step(
        execution,
        plan,
        precondition_evidence_ids=["evidence:second"],
    )

    assert execution.steps[1].status is InterceptionStepStatus.AUTHORIZED


def test_verified_stop_cut_point_skips_remaining_steps_and_contains_incident():
    plan = _plan(stop_first=True)
    execution = start_interception_execution(plan)
    execution = authorize_next_step(
        execution,
        plan,
        precondition_evidence_ids=["evidence:signer-compromise"],
    )
    execution = record_step_execution(
        execution,
        plan,
        step_id=plan.steps[0].step_id,
        execution_receipt_ids=["receipt:signer-frozen"],
    )
    execution = verify_step_outcome(
        execution,
        plan,
        step_id=plan.steps[0].step_id,
        succeeded=True,
        outcome_evidence_ids=["evidence:no-signing-authority"],
    )

    assert execution.contained is True
    assert execution.complete is True
    assert execution.steps[0].status is InterceptionStepStatus.VERIFIED_SUCCEEDED
    assert execution.steps[1].status is InterceptionStepStatus.SKIPPED


def test_execution_rejects_missing_evidence_and_receipts():
    plan = _plan()
    execution = start_interception_execution(plan)

    with pytest.raises(ValueError, match="precondition evidence"):
        authorize_next_step(execution, plan, precondition_evidence_ids=[])

    execution = authorize_next_step(
        execution,
        plan,
        precondition_evidence_ids=["evidence:attack"],
    )
    with pytest.raises(ValueError, match="execution receipt"):
        record_step_execution(
            execution,
            plan,
            step_id=plan.steps[0].step_id,
            execution_receipt_ids=[],
        )
