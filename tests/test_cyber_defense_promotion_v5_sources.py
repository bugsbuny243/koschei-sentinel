import pytest

import koschei_sentinel.cyber_defense_promotion_v5 as promotion_v5
from koschei_sentinel.cyber_defense_promotion_v5 import (
    build_cyber_defense_promotion_v5_from_sources,
)
from koschei_sentinel.cyber_megatron_gold_evidence import (
    CyberMegatronGoldEvaluationEvidence,
)
from koschei_sentinel.cyber_megatron_training import (
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
)
from tests.test_cyber_defense_promotion import _load, _multi, _single
from tests.test_cyber_defense_promotion_v5 import (
    _CHECKPOINT_SHA,
    _bundle,
    _gold_evidence,
    _policy,
    _sha,
)


def _source_kwargs(supplied):
    return {
        "promotion_id": "promotion:397b-v5-sources",
        "candidate_model_ref": QWEN35_397B_MODEL,
        "candidate_model_revision": QWEN35_397B_REVISION,
        "candidate_checkpoint_sha256": _CHECKPOINT_SHA,
        "training_bundle": _bundle(),
        "cyber_range_report": _single(),
        "multi_incident_range_report": _multi(),
        "defense_load_range_report": _load(),
        "supplied_gold_evidence": supplied,
        "gold_policy": _policy(),
        "gold_release_dir": "release",
        "gold_worker_dir": "worker",
        "gold_holdout_plan_path": "holdout.json",
        "gold_candidate_manifest_path": "candidate.json",
        "gold_candidate_config_path": "config.json",
        "gold_candidate_training_plan_path": "training-plan.json",
        "gold_checkpoint_dir": "checkpoint-100-merged",
        "gold_inference_pack": "pack",
        "gold_pack_signature": "pack-signature.json",
        "gold_reviewer_public_key": "reviewer.pem",
        "gold_reviewer_trust_policy": "reviewer-trust.json",
        "gold_owner_public_key": "owner.pem",
        "minimum_case_count": 50,
        "root": ".",
    }


def test_promotion_v5_source_path_accepts_only_freshly_rebuilt_evidence(monkeypatch) -> None:
    fresh = _gold_evidence()
    monkeypatch.setattr(
        promotion_v5,
        "build_cyber_megatron_gold_evidence",
        lambda **_kwargs: fresh,
    )

    promotion = build_cyber_defense_promotion_v5_from_sources(
        **_source_kwargs(fresh)
    )

    assert promotion.ready_for_promotion is True
    assert promotion.gold_evaluation_evidence_sha256 == fresh.evidence_sha256


def test_promotion_v5_source_path_rejects_supplied_evidence_drift(monkeypatch) -> None:
    fresh = _gold_evidence()
    payload = fresh.model_dump(mode="json")
    payload.pop("evidence_sha256")
    payload["run_id"] = "forged-but-self-consistent-run"
    forged = CyberMegatronGoldEvaluationEvidence(
        **payload,
        evidence_sha256=_sha(payload),
    )
    monkeypatch.setattr(
        promotion_v5,
        "build_cyber_megatron_gold_evidence",
        lambda **_kwargs: fresh,
    )

    with pytest.raises(ValueError, match="differs from fresh source-artifact rebuild"):
        build_cyber_defense_promotion_v5_from_sources(
            **_source_kwargs(forged)
        )
