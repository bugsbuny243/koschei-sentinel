from __future__ import annotations

from koschei_sentinel.agentic_security_eval_generator import GeneratedCase
from koschei_sentinel.agentic_security_review import ReviewRecord, review_record_digest


def build_review_record(
    *,
    case: GeneratedCase,
    decision: str,
    reviewer_id: str,
    reviewer_evidence_ref: str,
    rationale: str,
    rewritten_case_ref: str | None = None,
) -> ReviewRecord:
    payload = {
        "case_id": case.case_id,
        "source_lineage_sha256": case.lineage_sha256,
        "decision": decision,
        "reviewer_id": reviewer_id,
        "reviewer_evidence_ref": reviewer_evidence_ref,
        "rationale": rationale,
        "rewritten_case_ref": rewritten_case_ref,
        "record_sha256": "0" * 64,
    }
    provisional = ReviewRecord.model_validate(payload)
    payload["record_sha256"] = review_record_digest(provisional)
    return ReviewRecord.model_validate(payload)
