import pytest

from koschei_sentinel.gold_holdout_evaluation_evidence_cli import build_parser


def _base_args():
    return [
        "--release-dir",
        "release",
        "--inference-pack",
        "pack",
        "--inference-pack-signature",
        "pack-signature.json",
        "--inference-output",
        "output",
        "--candidate-export",
        "candidate-export",
        "--reviewer-public-key",
        "reviewer-public.pem",
        "--policy",
        "configs/training/gold-holdout-evaluation-policy.v1.json",
        "--output",
        "evidence.json",
    ]


def test_gold_holdout_evidence_cli_requires_explicit_policy() -> None:
    argv = _base_args()
    index = argv.index("--policy")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_candidate_export() -> None:
    argv = _base_args()
    index = argv.index("--candidate-export")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_reviewer_public_key() -> None:
    argv = _base_args()
    index = argv.index("--reviewer-public-key")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_pack_signature() -> None:
    argv = _base_args()
    index = argv.index("--inference-pack-signature")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_accepts_explicit_policy_export_and_reviewer_key() -> None:
    args = build_parser().parse_args(_base_args())

    assert args.candidate_export == "candidate-export"
    assert args.inference_pack_signature == "pack-signature.json"
    assert args.reviewer_public_key == "reviewer-public.pem"
    assert args.policy == "configs/training/gold-holdout-evaluation-policy.v1.json"
