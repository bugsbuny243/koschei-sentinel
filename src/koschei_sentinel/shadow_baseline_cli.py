from __future__ import annotations

import argparse
import json

from koschei_sentinel.promotion import (
    load_owner_private_key,
    load_owner_public_key,
)
from koschei_sentinel.shadow_baseline import (
    ShadowBaselineBlocked,
    apply_shadow_baseline_advance,
    approve_shadow_baseline_proposal,
    build_shadow_baseline_proposal,
    load_shadow_baseline_approval,
    load_shadow_baseline_lineage,
    load_shadow_baseline_proposal,
    verify_shadow_baseline_lineage,
    write_shadow_baseline_artifact,
)
from koschei_sentinel.shadow_receipt import load_shadow_replay_receipt
from koschei_sentinel.shadow_regression import load_shadow_regression_report
from koschei_sentinel.shadow_review import load_shadow_review_scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-shadow-baseline",
        description="Maintain an owner-signed best-known shadow baseline lineage",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose", help="Create a signed-baseline advance challenge")
    _add_evidence_inputs(propose)
    propose.add_argument("--owner-public-key", required=True)
    propose.add_argument("--lineage")
    propose.add_argument("--output", required=True)

    approve = commands.add_parser("approve", help="Owner-sign a baseline advance proposal")
    approve.add_argument("--proposal", required=True)
    approve.add_argument("--owner-private-key", required=True)
    approve.add_argument("--approver-id", required=True)
    approve.add_argument("--output", required=True)

    apply = commands.add_parser("apply", help="Verify evidence and append the approved baseline")
    _add_evidence_inputs(apply)
    apply.add_argument("--proposal", required=True)
    apply.add_argument("--approval", required=True)
    apply.add_argument("--owner-public-key", required=True)
    apply.add_argument("--lineage")
    apply.add_argument("--output", required=True)

    verify = commands.add_parser(
        "verify",
        help="Verify lineage digest, parent chain, proposal digests, and all owner signatures",
    )
    verify.add_argument("--lineage", required=True)
    verify.add_argument("--owner-public-key", required=True)
    verify.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "propose":
            artifact = build_shadow_baseline_proposal(
                *_load_evidence(args),
                load_owner_public_key(args.owner_public_key),
                lineage=_load_optional_lineage(args.lineage),
            )
            write_shadow_baseline_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "approve":
            artifact = approve_shadow_baseline_proposal(
                load_shadow_baseline_proposal(args.proposal),
                load_owner_private_key(args.owner_private_key),
                approver_id=args.approver_id,
            )
            write_shadow_baseline_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "apply":
            artifact = apply_shadow_baseline_advance(
                load_shadow_baseline_proposal(args.proposal),
                load_shadow_baseline_approval(args.approval),
                *_load_evidence(args),
                load_owner_public_key(args.owner_public_key),
                lineage=_load_optional_lineage(args.lineage),
            )
            write_shadow_baseline_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        else:
            artifact = verify_shadow_baseline_lineage(
                load_shadow_baseline_lineage(args.lineage),
                load_owner_public_key(args.owner_public_key),
            )
            result = {
                "ok": True,
                "head_candidate_id": artifact.head_candidate_id,
                "entries": len(artifact.entries),
                "replay_sha256": artifact.replay_sha256,
                "lineage_digest": artifact.lineage_digest,
                "historical_signatures_verified": True,
                "automatic_baseline_selection_allowed": False,
                "production_deployment_allowed": False,
                "web3_runtime_integration_allowed": False,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ShadowBaselineBlocked as exc:
        print(f"sentinel-shadow-baseline: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-shadow-baseline: {exc}")
        return 2


def _add_evidence_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--regression", required=True)
    parser.add_argument("--baseline-scorecard", required=True)
    parser.add_argument("--baseline-receipt", required=True)
    parser.add_argument("--candidate-scorecard", required=True)
    parser.add_argument("--candidate-receipt", required=True)


def _load_evidence(args: argparse.Namespace) -> tuple[object, object, object, object, object]:
    return (
        load_shadow_regression_report(args.regression),
        load_shadow_review_scorecard(args.baseline_scorecard),
        load_shadow_replay_receipt(args.baseline_receipt),
        load_shadow_review_scorecard(args.candidate_scorecard),
        load_shadow_replay_receipt(args.candidate_receipt),
    )


def _load_optional_lineage(path: str | None):
    return load_shadow_baseline_lineage(path) if path else None


if __name__ == "__main__":
    raise SystemExit(main())
