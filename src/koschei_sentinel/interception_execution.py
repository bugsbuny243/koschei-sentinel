from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.interception_planner import InterceptionPlan
from koschei_sentinel.models import StrictModel


class InterceptionStepStatus(StrEnum):
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    EXECUTED = "EXECUTED"
    VERIFIED_SUCCEEDED = "VERIFIED_SUCCEEDED"
    VERIFIED_FAILED = "VERIFIED_FAILED"
    SKIPPED = "SKIPPED"


class InterceptionExecutionStep(StrictModel):
    step_id: str
    sequence: int = Field(ge=1)
    status: InterceptionStepStatus = InterceptionStepStatus.PENDING
    precondition_evidence_ids: list[str] = Field(default_factory=list)
    execution_receipt_ids: list[str] = Field(default_factory=list)
    outcome_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_is_required_for_state(self) -> InterceptionExecutionStep:
        if self.status in {
            InterceptionStepStatus.AUTHORIZED,
            InterceptionStepStatus.EXECUTED,
            InterceptionStepStatus.VERIFIED_SUCCEEDED,
            InterceptionStepStatus.VERIFIED_FAILED,
        } and not self.precondition_evidence_ids:
            raise ValueError("authorized or later interception state requires precondition evidence")
        if self.status in {
            InterceptionStepStatus.EXECUTED,
            InterceptionStepStatus.VERIFIED_SUCCEEDED,
            InterceptionStepStatus.VERIFIED_FAILED,
        } and not self.execution_receipt_ids:
            raise ValueError("executed or later interception state requires execution receipt")
        if self.status in {
            InterceptionStepStatus.VERIFIED_SUCCEEDED,
            InterceptionStepStatus.VERIFIED_FAILED,
        } and not self.outcome_evidence_ids:
            raise ValueError("verified interception state requires outcome evidence")
        return self


class InterceptionExecution(StrictModel):
    schema_version: Literal["sentinel.interception-execution.v1"] = (
        "sentinel.interception-execution.v1"
    )
    graph_id: str
    plan_fingerprint: str
    steps: list[InterceptionExecutionStep]
    contained: bool = False
    complete: bool = False
    rationale: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sequence_integrity(self) -> InterceptionExecution:
        sequences = [step.sequence for step in self.steps]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("interception execution steps must be contiguous and ordered")
        active = [
            step for step in self.steps
            if step.status in {InterceptionStepStatus.AUTHORIZED, InterceptionStepStatus.EXECUTED}
        ]
        if len(active) > 1:
            raise ValueError("only one interception step may be active at a time")
        return self


def _fingerprint(plan: InterceptionPlan) -> str:
    import hashlib

    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def start_interception_execution(plan: InterceptionPlan) -> InterceptionExecution:
    return InterceptionExecution(
        graph_id=plan.graph_id,
        plan_fingerprint=_fingerprint(plan),
        steps=[
            InterceptionExecutionStep(step_id=step.step_id, sequence=step.sequence)
            for step in plan.steps
        ],
        complete=not plan.steps,
        rationale=[
            "interception executes sequentially",
            "post-action verification is mandatory before advancing to the next cut point",
        ],
    )


def _validate_plan(execution: InterceptionExecution, plan: InterceptionPlan) -> None:
    if execution.graph_id != plan.graph_id or execution.plan_fingerprint != _fingerprint(plan):
        raise ValueError("execution does not belong to the supplied interception plan")


def _next_pending(execution: InterceptionExecution) -> InterceptionExecutionStep | None:
    for step in execution.steps:
        if step.status is InterceptionStepStatus.PENDING:
            return step
    return None


def authorize_next_step(
    execution: InterceptionExecution,
    plan: InterceptionPlan,
    *,
    precondition_evidence_ids: list[str],
) -> InterceptionExecution:
    _validate_plan(execution, plan)
    if execution.complete:
        raise ValueError("interception execution is already complete")
    if not precondition_evidence_ids:
        raise ValueError("precondition evidence is required")
    if any(
        step.status in {InterceptionStepStatus.AUTHORIZED, InterceptionStepStatus.EXECUTED}
        for step in execution.steps
    ):
        raise ValueError("current interception step must be verified before another is authorized")

    next_step = _next_pending(execution)
    if next_step is None:
        raise ValueError("no pending interception step remains")
    next_step.status = InterceptionStepStatus.AUTHORIZED
    next_step.precondition_evidence_ids = list(dict.fromkeys(precondition_evidence_ids))
    return InterceptionExecution.model_validate(execution.model_dump())


def record_step_execution(
    execution: InterceptionExecution,
    plan: InterceptionPlan,
    *,
    step_id: str,
    execution_receipt_ids: list[str],
) -> InterceptionExecution:
    _validate_plan(execution, plan)
    if not execution_receipt_ids:
        raise ValueError("execution receipt is required")
    step = next((row for row in execution.steps if row.step_id == step_id), None)
    if step is None:
        raise ValueError(f"unknown interception step: {step_id}")
    if step.status is not InterceptionStepStatus.AUTHORIZED:
        raise ValueError("interception step must be AUTHORIZED before execution is recorded")
    step.status = InterceptionStepStatus.EXECUTED
    step.execution_receipt_ids = list(dict.fromkeys(execution_receipt_ids))
    return InterceptionExecution.model_validate(execution.model_dump())


def verify_step_outcome(
    execution: InterceptionExecution,
    plan: InterceptionPlan,
    *,
    step_id: str,
    succeeded: bool,
    outcome_evidence_ids: list[str],
) -> InterceptionExecution:
    _validate_plan(execution, plan)
    if not outcome_evidence_ids:
        raise ValueError("outcome evidence is required")
    step = next((row for row in execution.steps if row.step_id == step_id), None)
    if step is None:
        raise ValueError(f"unknown interception step: {step_id}")
    if step.status is not InterceptionStepStatus.EXECUTED:
        raise ValueError("interception step must be EXECUTED before verification")

    plan_step = next(row for row in plan.steps if row.step_id == step_id)
    step.status = (
        InterceptionStepStatus.VERIFIED_SUCCEEDED
        if succeeded
        else InterceptionStepStatus.VERIFIED_FAILED
    )
    step.outcome_evidence_ids = list(dict.fromkeys(outcome_evidence_ids))

    if succeeded and plan_step.stop_if_verified:
        for later in execution.steps:
            if later.sequence > step.sequence and later.status is InterceptionStepStatus.PENDING:
                later.status = InterceptionStepStatus.SKIPPED
        execution.contained = True
        execution.complete = True
        execution.rationale.append(
            f"verified cut point {step_id} met the plan stop condition; remaining steps skipped"
        )
    elif not any(row.status is InterceptionStepStatus.PENDING for row in execution.steps):
        execution.complete = True
        execution.contained = any(
            row.status is InterceptionStepStatus.VERIFIED_SUCCEEDED for row in execution.steps
        )

    return InterceptionExecution.model_validate(execution.model_dump())
