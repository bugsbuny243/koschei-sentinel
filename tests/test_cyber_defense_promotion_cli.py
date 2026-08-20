import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import koschei_sentinel.cyber_defense_promotion as promotion_module
import koschei_sentinel.cyber_defense_promotion_cli as promotion_cli
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.training import canonical_json
from tests.test_cyber_defense_promotion import (
    ADAPTER_DIGEST,
    _bundle,
    _gold_evidence,
    _load,
    _multi,
    _single,
)


def _required_args() -> list[str]:
    return [
        "--promotion-id",
        "promotion:test",
        "--candidate-model",
        "sentinel:candidate",
        "--candidate-revision",
        ADAPTER_DIGEST,
        "--training-bundle",
        "bundle.json",
        "--cyber-range-report",
        "single.json",
        "--multi-incident-range-report",
        "multi.json",
        "--defense-load-range-report",
        "load.json",
        "--gold-holdout-evidence",
        "gold.json",
        "--gold-holdout-policy",
        "policy.json",
        "--output",
        "promotion.json",
    ]


def test_promotion_cli_requires_gold_source_artifacts() -> None:
    with pytest.raises(SystemExit) as exc:
        promotion_cli.build_parser().parse_args(_required_args())

    assert exc.value.code == 2


def test_promotion_cli_accepts_all_gold_source_artifacts() -> None:
    args = promotion_cli.build_parser().parse_args(
        _required_args()
        + [
            "--gold-release-dir",
            "release",
            "--gold-inference-pack",
            "pack",
            "--gold-inference-output",
            "inference-output",
            "--gold-candidate-export",
            "candidate-export",
            "--gold-reviewer-public-key",
            "reviewer-public.pem",
        ]
    )

    assert args.gold_release_dir == "release"
    assert args.gold_inference_pack == "pack"
    assert args.gold_inference_output == "inference-output"
    assert args.gold_candidate_export == "candidate-export"
    assert args.gold_reviewer_public_key == "reviewer-public.pem"


def _signed_evidence(evidence):
    payload = evidence.model_dump(mode="json")
    payload.pop("evidence_sha256", None)
    payload["review_signature_audit_sha256"] = "a" * 64
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return evidence.__class__.model_validate({**payload, "evidence_sha256": digest})


def _source_builder_kwargs(policy, supplied):
    reviewer_public_key = Ed25519PrivateKey.generate().public_key()
    return {
        "promotion_id": "promotion:test",
        "candidate_model_ref": "sentinel:candidate",
        "candidate_model_revision": ADAPTER_DIGEST,
        "training_bundle": _bundle(),
        "cyber_range_report": _single(),
        "multi_incident_range_report": _multi(),
        "defense_load_range_report": _load(),
        "supplied_gold_holdout_evidence": supplied,
        "gold_holdout_policy": policy,
        "gold_release_dir": "release",
        "gold_inference_pack_dir": "pack",
        "gold_inference_output_dir": "inference-output",
        "gold_candidate_export_dir": "candidate-export",
        "gold_reviewer_public_key": reviewer_public_key,
    }


def test_promotion_rejects_supplied_gold_evidence_that_differs_from_fresh_rebuild(
    monkeypatch,
) -> None:
    policy = GoldHoldoutEvaluationPolicy()
    supplied = _signed_evidence(_gold_evidence(policy=policy))
    rebuilt = _signed_evidence(_gold_evidence(revision="8" * 64, policy=policy))
    monkeypatch.setattr(
        promotion_module,
        "build_gold_holdout_evaluation_evidence",
        lambda **_kwargs: rebuilt,
    )

    with pytest.raises(ValueError, match="differs from fresh source-artifact rebuild"):
        promotion_module.build_cyber_defense_promotion_evidence_from_sources(
            **_source_builder_kwargs(policy, supplied)
        )


def test_promotion_accepts_supplied_gold_evidence_only_when_fresh_rebuild_matches(
    monkeypatch,
) -> None:
    policy = GoldHoldoutEvaluationPolicy()
    supplied = _signed_evidence(_gold_evidence(policy=policy))
    monkeypatch.setattr(
        promotion_module,
        "build_gold_holdout_evaluation_evidence",
        lambda **_kwargs: supplied,
    )

    promotion = promotion_module.build_cyber_defense_promotion_evidence_from_sources(
        **_source_builder_kwargs(policy, supplied)
    )

    assert promotion.ready_for_promotion is True
    assert promotion.gold_review_signature_audit_sha256 == "a" * 64
    assert promotion.gold_holdout_evaluation_evidence_sha256 == supplied.evidence_sha256
