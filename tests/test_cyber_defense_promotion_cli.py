import json
from types import SimpleNamespace

import pytest

import koschei_sentinel.cyber_defense_promotion_cli as promotion_cli
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from tests.test_cyber_defense_promotion import ADAPTER_DIGEST, _gold_evidence


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
        ]
    )

    assert args.gold_release_dir == "release"
    assert args.gold_inference_pack == "pack"
    assert args.gold_inference_output == "inference-output"
    assert args.gold_candidate_export == "candidate-export"


def test_promotion_rejects_supplied_gold_evidence_that_differs_from_fresh_rebuild(
    tmp_path,
    monkeypatch,
) -> None:
    policy = GoldHoldoutEvaluationPolicy()
    supplied = _gold_evidence(policy=policy)
    rebuilt = _gold_evidence(revision="8" * 64, policy=policy)
    supplied_path = tmp_path / "gold.json"
    supplied_path.write_text(
        json.dumps(supplied.model_dump(mode="json")),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        gold_holdout_evidence=str(supplied_path),
        gold_release_dir="release",
        gold_inference_pack="pack",
        gold_inference_output="inference-output",
        gold_candidate_export="candidate-export",
    )
    monkeypatch.setattr(
        promotion_cli,
        "build_gold_holdout_evaluation_evidence",
        lambda **_kwargs: rebuilt,
    )

    with pytest.raises(ValueError, match="differs from fresh source-artifact rebuild"):
        promotion_cli._rebuild_and_match_gold_evidence(args, policy)


def test_promotion_accepts_supplied_gold_evidence_only_when_fresh_rebuild_matches(
    tmp_path,
    monkeypatch,
) -> None:
    policy = GoldHoldoutEvaluationPolicy()
    supplied = _gold_evidence(policy=policy)
    supplied_path = tmp_path / "gold.json"
    supplied_path.write_text(
        json.dumps(supplied.model_dump(mode="json")),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        gold_holdout_evidence=str(supplied_path),
        gold_release_dir="release",
        gold_inference_pack="pack",
        gold_inference_output="inference-output",
        gold_candidate_export="candidate-export",
    )
    monkeypatch.setattr(
        promotion_cli,
        "build_gold_holdout_evaluation_evidence",
        lambda **_kwargs: supplied,
    )

    rebuilt = promotion_cli._rebuild_and_match_gold_evidence(args, policy)

    assert rebuilt.evidence_sha256 == supplied.evidence_sha256
