from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from pydantic import Field, model_validator

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
from .request_body_limit import ActiveDefenseRequestBodyLimit

app = FastAPI(
    title="Koschei Sentinel",
    version=__version__,
    description="Evidence-grounded commentary boundary for Koschei security cases.",
)

app.add_middleware(ActiveDefenseRequestBodyLimit)

# Limits apply to the public simulation endpoints, not the internal planning engine.
_MAX_API_GRAPH_ENTITIES = 4096
_MAX_API_GRAPH_RELATIONS = 16384
_MAX_API_SOURCE_ENROLLMENTS = 4096
_MAX_API_INTERCEPTION_STEPS = 128


def _check_graph_budget(graph: CyberStateGraph) -> None:
    if len(graph.entities) > _MAX_API_GRAPH_ENTITIES:
        raise ValueError("active-defense API graph entity limit exceeded")
    if len(graph.relations) > _MAX_API_GRAPH_RELATIONS:
        raise ValueError("active-defense API graph relation limit exceeded")


def _mark_simulation_response(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Sentinel-Execution-Mode"] = "simulation-only"
    response.headers["X-Sentinel-Evidence-Trust"] = "caller-supplied"


class ActiveDefensePlanRequest(StrictModel):
    graph: CyberStateGraph
    critical_entity_ids: list[str] = Field(default_factory=list, max_length=512)
    focus_entity_ids: list[str] | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def enforce_request_budget(self) -> ActiveDefensePlanRequest:
        _check_graph_budget(self.graph)
        return self


class ActiveDefensePlanResponse(StrictModel):
    active_defense_plan: ActiveDefensePlan
    interception_plan: InterceptionPlan
    interception_execution: InterceptionExecution


class AssuredActiveDefensePlanRequest(StrictModel):
    graph: CyberStateGraph
    graph_receipt: PerceptionGraphReceipt
    perception_assurance: PerceptionAssuranceSummary
    registry: PerceptionSourceRegistry
    critical_entity_ids: list[str] = Field(default_factory=list, max_length=512)
    focus_entity_ids: list[str] | None = Field(default=None, max_length=512)
    policy: ActiveDefenseAssurancePolicy | None = None

    @model_validator(mode="after")
    def enforce_request_budget(self) -> AssuredActiveDefensePlanRequest:
        _check_graph_budget(self.graph)
        if len(self.registry.enrollments) > _MAX_API_SOURCE_ENROLLMENTS:
            raise ValueError("active-defense API source enrollment limit exceeded")
        return self


class AssuredConnectorEnvelopeRequest(StrictModel):
    scope: ProtectedScope
    assured_plan: AssuredActiveDefensePlan
    interception_plan: InterceptionPlan
    step_id: str
    precondition_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    dry_run: bool = True

    @model_validator(mode="after")
    def enforce_request_budget(self) -> AssuredConnectorEnvelopeRequest:
        if len(self.interception_plan.steps) > _MAX_API_INTERCEPTION_STEPS:
            raise ValueError("active-defense API interception step limit exceeded")
        return self


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


@app.post(
    "/v1/active-defense/plan",
    response_model=ActiveDefensePlanResponse,
    description="Pure simulation using caller-supplied cyber graph; no live defense execution.",
)
def create_active_defense_plan(
    request: ActiveDefensePlanRequest, response: Response
) -> ActiveDefensePlanResponse:
    _mark_simulation_response(response)
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


@app.post(
    "/v1/active-defense/assured-plan",
    response_model=AssuredActiveDefensePlan,
    description=(
        "Validate caller-supplied receipt consistency and evidence domains. "
        "Does not attest to source ownership or authenticate telemetry."
    ),
)
def create_assured_active_defense_plan(
    request: AssuredActiveDefensePlanRequest, response: Response
) -> AssuredActiveDefensePlan:
    _mark_simulation_response(response)
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


@app.post(
    "/v1/active-defense/connector-envelope",
    response_model=AssuredDefenseConnectorEnvelope,
    description=(
        "Generate a dry-run command envelope only. The legacy production_authorized "
        "field is not a server-issued authorization or proof of source identity. "
        "Live commands are always rejected."
    ),
)
def create_assured_connector_envelope(
    request: AssuredConnectorEnvelopeRequest, response: Response
) -> AssuredDefenseConnectorEnvelope:
    # This unauthenticated API only creates dry-run artifacts; production commands
    # require a separate, authenticated authority and approval boundary.
    _mark_simulation_response(response)
    response.headers["X-Sentinel-Production-Authorization"] = "not-granted"
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
