from __future__ import annotations

from pathlib import Path

from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutEvaluationReport,
    _digest_without,
    _load_holdout_cases,
)


def build_zero_prediction_gold_report(
    release_dir: str | Path,
    *,
    model_ref: str,
    model_revision: str,
    adapter_digest: str,
    policy: GoldHoldoutEvaluationPolicy | None = None,
) -> GoldHoldoutEvaluationReport:
    audit = audit_gold_defense_release(release_dir)
    if not audit.valid:
        raise ValueError("cannot evaluate zero predictions against an invalid Gold release")
    cases = _load_holdout_cases(Path(release_dir))
    if not cases:
        raise ValueError("Gold HOLDOUT release has no evaluation cases")
    selected_policy = policy or GoldHoldoutEvaluationPolicy()

    violations = ["missing Gold HOLDOUT predictions"]
    thresholds = (
        ("structural exact rate", selected_policy.minimum_structural_exact_rate),
        ("mode accuracy", selected_policy.minimum_mode_accuracy),
        ("action accuracy", selected_policy.minimum_action_accuracy),
        ("target accuracy", selected_policy.minimum_target_accuracy),
        (
            "evidence selection accuracy",
            selected_policy.minimum_evidence_selection_accuracy,
        ),
        ("evidence grounding rate", selected_policy.minimum_evidence_grounding_rate),
        ("target grounding rate", selected_policy.minimum_target_grounding_rate),
        ("outcome verification rate", selected_policy.minimum_outcome_verification_rate),
    )
    violations.extend(
        f"{label} below policy: 0.000000 < {minimum:.6f}"
        for label, minimum in thresholds
        if minimum > 0.0
    )

    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-evaluation-report.v2",
        "model_ref": model_ref,
        "model_revision": model_revision,
        "adapter_digest": adapter_digest,
        "case_count": len(cases),
        "prediction_count": 0,
        "missing_case_ids": sorted(row.case_id for row in cases),
        "extra_case_ids": [],
        "structural_exact_cases": 0,
        "structural_exact_rate": 0.0,
        "compared_steps": 0,
        "predicted_steps": 0,
        "mode_accuracy": 0.0,
        "action_accuracy": 0.0,
        "target_accuracy": 0.0,
        "evidence_selection_accuracy": 0.0,
        "evidence_grounding_rate": 0.0,
        "target_grounding_rate": 0.0,
        "outcome_verification_rate": 0.0,
        "case_results": [],
        "passed": False,
        "violations": violations,
    }
    payload["report_sha256"] = _digest_without(payload, "report_sha256")
    return GoldHoldoutEvaluationReport.model_validate(payload)
