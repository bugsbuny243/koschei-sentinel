from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import Field

from . import __version__
from .active_defense_planner import ActiveDefensePlan, build_active_defense_plan
from .assured_active_defense import (
    ActiveDefenseAssurancePolicy,
    AssuredActiveDefensePlan,
    build_assured_active_defense_plan,
)
from .assured_defense_connector import (
    AssuredDefenseConnectorEnvelope,
    build_assured_connector_envelope,
)
from .cyber_state_graph import CyberStateGraph
from .defense_connector_contract import ProtectedScope
from .interception_execution import InterceptionExecution, start_interception_execution
from .interception_planner import InterceptionPlan, build_interception_plan
from .models import CheckedOpinion, SecurityCase, StrictModel
from .perception_assurance import PerceptionAssuranceSummary
from .perception_graph_binding import PerceptionGraphReceipt
from .perception_source_registry import PerceptionSourceRegistry
from .policy import baseline_opinion, validate_opinion

app = FastAPI(
    title="Koschei Sentinel",
    version=__version__,
    description="Evidence-grounded commentary boundary for Koschei security cases.",
)


class ActiveDefensePlanRequest(StrictModel):
    graph: CyberStateGraph
    critical_entity_ids: list[str] = Field(default_factory=list)
    focus_entity_ids: list[str] | None = None


class ActiveDefensePlanResponse(StrictModel):
    active_defense_plan: ActiveDefensePlan
    interception_plan: InterceptionPlan
    interception_execution: InterceptionExecution


class AssuredActiveDefensePlanRequest(StrictModel):
    graph: CyberStateGraph
    graph_receipt: PerceptionGraphReceipt
    perception_assurance: PerceptionAssuranceSummary
    registry: PerceptionSourceRegistry
    critical_entity_ids: list[str] = Field(default_factory=list)
    focus_entity_ids: list[str] | None = None
    policy: ActiveDefenseAssurancePolicy | None = None


class AssuredConnectorEnvelopeRequest(StrictModel):
    scope: ProtectedScope
    assured_plan: AssuredActiveDefensePlan
    interception_plan: InterceptionPlan
    step_id: str
    precondition_evidence_ids: list[str]
    dry_run: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "koschei-sentinel", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready", "engine": "sentinel-baseline-v0.1"}


@app.post("/v1/opinions", response_model=CheckedOpinion)
def create_opinion(case: SecurityCase) -> CheckedOpinion:
    opinion = baseline_opinion(case)
    violations = validate_opinion(case, opinion)
    return CheckedOpinion(opinion=opinion, policy_violations=violations, accepted=not violations)


@app.post("/v1/active-defense/plan", response_model=ActiveDefensePlanResponse)
def create_active_defense_plan(request: ActiveDefensePlanRequest) -> ActiveDefensePlanResponse:
    try:
        active = build_active_defense_plan(
            request.graph,
            critical_entity_ids=request.critical_entity_ids,
            focus_entity_ids=request.focus_entity_ids,
        )
        interception = build_interception_plan(active)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ActiveDefensePlanResponse(
        active_defense_plan=active,
        interception_plan=interception,
        interception_execution=start_interception_execution(interception),
    )


@app.post("/v1/active-defense/assured-plan", response_model=AssuredActiveDefensePlan)
def create_assured_active_defense_plan(
    request: AssuredActiveDefensePlanRequest,
) -> AssuredActiveDefensePlan:
    try:
        return build_assured_active_defense_plan(
            request.graph,
            graph_receipt=request.graph_receipt,
            perception_assurance=request.perception_assurance,
            registry=request.registry,
            critical_entity_ids=request.critical_entity_ids,
            focus_entity_ids=request.focus_entity_ids,
            policy=request.policy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/active-defense/connector-envelope", response_model=AssuredDefenseConnectorEnvelope)
def create_assured_connector_envelope(
    request: AssuredConnectorEnvelopeRequest,
) -> AssuredDefenseConnectorEnvelope:
    # This unauthenticated API only creates dry-run artifacts; production commands
    # require a separate, authenticated authority and approval boundary.
    if not request.dry_run:
        raise HTTPException(
            status_code=403,
            detail="production connector commands require an authenticated authorization boundary",
        )
    try:
        return build_assured_connector_envelope(
            scope=request.scope,
            assured_plan=request.assured_plan,
            interception_plan=request.interception_plan,
            step_id=request.step_id,
            precondition_evidence_ids=request.precondition_evidence_ids,
            dry_run=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
