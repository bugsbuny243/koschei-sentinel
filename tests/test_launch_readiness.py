from __future__ import annotations

import json
from pathlib import Path

from koschei_sentinel.launch_readiness import audit_launch_readiness


DIGEST = "a" * 64


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _finalization() -> dict:
    return {
        "schema_version": "sentinel.candidate-finalization.v1",
        "candidate_id": "candidate-1",
        "state": "finalized_incubation",
        "authority": "explanation_only",
        "offline_receipt_digest": DIGEST,
        "job_digest": DIGEST,
        "adapter_manifest_digest": DIGEST,
        "adapter_digest": DIGEST,
        "dataset_manifest_digest": DIGEST,
        "training_config_digest": DIGEST,
        "comparison_digest": DIGEST,
        "benchmark_suite_digest": DIGEST,
        "benchmark_report_digest": DIGEST,
        "candidate_record_digest": DIGEST,
        "previous_registry_digest": DIGEST,
        "updated_registry_digest": DIGEST,
        "automatic_registry_replacement_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "finalization_digest": DIGEST,
    }


def test_launch_readiness_blocks_current_incubation_contract(tmp_path: Path) -> None:
    finalization = _write(tmp_path / "candidate-finalization.json", _finalization())
    holdout = _write(
        tmp_path / "holdout.json",
        {"schema_version": "sentinel.gold-holdout-evaluation.v1", "ok": True},
    )
    authority = _write(
        tmp_path / "authority.json",
        {
            "candidate_id": "candidate-1",
            "production_deployment_allowed": True,
        },
    )

    report = audit_launch_readiness(
        finalization_path=finalization,
        holdout_evidence_paths=[holdout],
        production_authority_path=authority,
    )

    assert report.market_release_ready is False
    assert report.production_authority_present is True
    assert any("forbids production deployment" in item for item in report.blockers)


def test_launch_readiness_blocks_missing_holdout_and_authority(tmp_path: Path) -> None:
    finalization = _write(tmp_path / "candidate-finalization.json", _finalization())

    report = audit_launch_readiness(
        finalization_path=finalization,
        holdout_evidence_paths=[],
    )

    assert report.market_release_ready is False
    assert report.holdout_evidence_count == 0
    assert any("no independent holdout evidence" in item for item in report.blockers)
    assert any("no explicit production deployment authority" in item for item in report.blockers)
