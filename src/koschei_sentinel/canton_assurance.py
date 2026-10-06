"""Minimal Canton-specific evidence and deterministic assurance proof-of-work.

This module is deliberately read-only. It normalizes a bounded fixture into a
canonical evidence contract and evaluates deterministic assurance conditions.
It does not connect to a validator, sign transactions, or claim production
Canton integration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA = "canton.security.evidence.v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(value: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def normalize_canton_fixture(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize a public/test Canton-like artifact into bounded evidence."""
    evidence_body = {
        "schema": SCHEMA,
        "source_class": source.get("source_class", "test_fixture"),
        "subject_ref": source.get("subject_ref", "unknown"),
        "artifact_version": source.get("artifact_version"),
        "provenance": source.get("provenance"),
        "integrity": source.get("integrity"),
        "expected_evidence": bool(source.get("expected_evidence", False)),
        "observed_evidence": bool(source.get("observed_evidence", False)),
    }
    return {**evidence_body, "evidence_id": _stable_id(evidence_body)}


def assess_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic finding that always cites its evidence ID."""
    evidence_id = evidence["evidence_id"]
    missing = [
        field
        for field in ("provenance", "integrity", "artifact_version")
        if not evidence.get(field)
    ]

    if missing:
        status = "insufficient_evidence"
        severity = "unknown"
        reason = "required evidence fields are unavailable: " + ", ".join(sorted(missing))
    elif evidence["expected_evidence"] and not evidence["observed_evidence"]:
        status = "fail"
        severity = "high"
        reason = "expected security evidence was not observed"
    else:
        status = "pass"
        severity = "none"
        reason = "bounded evidence satisfies the prototype assurance condition"

    finding_body = {
        "contract": "canton.security.assurance.v1",
        "rule_id": "CANTON-EVIDENCE-COVERAGE-001",
        "status": status,
        "severity": severity,
        "reason": reason,
        "evidence_ids": [evidence_id],
    }
    return {**finding_body, "finding_id": _stable_id(finding_body)}


def run_proof(source: dict[str, Any]) -> dict[str, Any]:
    evidence = normalize_canton_fixture(source)
    return {"evidence": evidence, "finding": assess_evidence(evidence)}
