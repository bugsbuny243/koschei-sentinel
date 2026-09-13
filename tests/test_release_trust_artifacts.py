from __future__ import annotations

import hashlib
import json

import pytest

from koschei_sentinel.candidate_finalization import (
    CandidateFinalization,
    CandidateFinalizationBlocked,
    verify_candidate_finalization,
)
from koschei_sentinel.production_authority import (
    ProductionAuthorityBlocked,
    load_verified_production_holdout,
)


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _finalization() -> CandidateFinalization:
    d = "a" * 64
    payload = {
        "schema_version": "sentinel.candidate-finalization.v1",
        "candidate_id": "candidate-1",
        "state": "finalized_incubation",
        "authority": "explanation_only",
        "offline_receipt_digest": d,
        "job_digest": d,
        "adapter_manifest_digest": d,
        "adapter_digest": d,
        "dataset_manifest_digest": d,
        "training_config_digest": d,
        "comparison_digest": d,
        "benchmark_suite_digest": d,
        "benchmark_report_digest": d,
        "candidate_record_digest": d,
        "previous_registry_digest": d,
        "updated_registry_digest": d,
        "automatic_registry_replacement_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    return CandidateFinalization.model_validate(
        {**payload, "finalization_digest": _digest(payload)}
    )


def test_candidate_finalization_self_hash_is_verified() -> None:
    finalization = _finalization()
    assert verify_candidate_finalization(finalization) == finalization


def test_candidate_finalization_rejects_schema_valid_tampering() -> None:
    finalization = _finalization().model_copy(update={"adapter_digest": "b" * 64})
    with pytest.raises(CandidateFinalizationBlocked, match="digest"):
        verify_candidate_finalization(finalization)


def test_generic_holdout_named_json_is_not_release_evidence(tmp_path) -> None:
    artifact = tmp_path / "fake-holdout.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "sentinel.some-holdout-looking-artifact.v1",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProductionAuthorityBlocked, match="Gold HOLDOUT"):
        load_verified_production_holdout(artifact)


def test_failed_gold_holdout_shape_cannot_be_used_as_release_evidence(tmp_path) -> None:
    artifact = tmp_path / "failed-gold-holdout.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "sentinel.gold-holdout-evaluation-evidence.v1",
                "passed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProductionAuthorityBlocked):
        load_verified_production_holdout(artifact)
