from datetime import datetime, timezone

from koschei_sentinel.authority_policy import CapabilityGrant, DelegationChain
from koschei_sentinel.cybersecurity_core import AnalysisMode, SecurityArtifact, SecurityCase, SecurityDomain, SecurityFinding
from koschei_sentinel.defensive_recovery import DefensiveRecoveryCoordinator
from koschei_sentinel.defensive_response import DefensiveAction, DefensiveActionRequest, DefensivePolicy, DefensiveResponsePlane
from koschei_sentinel.defensive_runtime import DefensiveSentinelRuntime
from koschei_sentinel.inference_contract import AuthorityEnvelope, EvidenceBinding, InferenceResponse, RiskClass
from koschei_sentinel.inference_validation import FailClosedResponseValidator
from koschei_sentinel.sentinel_runtime import SentinelRuntime

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)

class Engine:
    engine_id = "fixture"
    engine_version = "1"
    def infer(self, request):
        return InferenceResponse(request.request_id, self.engine_id, self.engine_version,
            {"finding": "suspicious workload behavior"},
            (EvidenceBinding("f1", ("log-1",), "bound to observed log"),))

class Executor:
    def execute(self, request):
        return {"verified": True, "isolated": request.target_id}

class Recovery:
    def rollback(self, request, execution):
        return {"verified": True}


def test_analysis_to_verified_containment_receipt():
    case = SecurityCase("case-1", "t1", AnalysisMode.INVESTIGATE, (
        SecurityArtifact("log-1", SecurityDomain.LOG, "text/plain", "fixture", "lab", "sha256:log"),
    ), "contain verified threat")
    authority = AuthorityEnvelope("operator", "controller", "sentinel", ("analyze",))
    analysis = SentinelRuntime(engine=Engine(), validators=(FailClosedResponseValidator(),))

    runtime = DefensiveSentinelRuntime(
        analysis_runtime=analysis,
        response_plane=DefensiveResponsePlane(executor=Executor()),
        recovery=DefensiveRecoveryCoordinator(recovery_executor=Recovery()),
        finding_parser=lambda result: (SecurityFinding("f1", "suspicious behavior", "high", .95, ("log-1",), "isolate workload", "verified"),),
        action_planner=lambda findings: (DefensiveActionRequest("a1", "t1", DefensiveAction.ISOLATE_WORKLOAD, "workload-1", ("f1",), ("log-1",), "contain verified threat"),),
        effect_verifier=lambda action, execution: execution.get("isolated") == action.target_id,
    )
    result = runtime.defend(
        case=case,
        authority=authority,
        delegation_chain=DelegationChain((CapabilityGrant("g1", "operator", "sentinel", ("analyze",)),)),
        defensive_policy=DefensivePolicy("t1", frozenset({DefensiveAction.ISOLATE_WORKLOAD}), frozenset({"workload-1"})),
        risk_class=RiskClass.HIGH,
        required_scopes=("analyze",),
        audit_id="audit-1",
        now=NOW,
    )
    assert result.analysis.policy_decision.allowed
    assert result.findings[0].evidence_ids == ("log-1",)
    assert result.action_receipts[0].effect_verified
    assert result.action_receipts[0].digest().startswith("sha256:")
