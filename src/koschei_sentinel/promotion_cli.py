from __future__ import annotations

import argparse
import json

from koschei_sentinel.promotion import (
    PromotionBlocked,
    approve_promotion_proposal,
    build_promotion_proposal,
    load_candidate_finalization,
    load_owner_private_key,
    load_owner_public_key,
    load_promotion_approval,
    load_promotion_proposal,
    verify_promotion_approval,
    write_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create, owner-sign, or verify a shadow-research-only Sentinel "
            "promotion artifact"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser(
        "propose",
        help="Create an owner-signature challenge from a finalized incubation candidate",
    )
    propose.add_argument("--finalization", required=True)
    propose.add_argument("--owner-public-key", required=True)
    propose.add_argument("--output", required=True)

    approve = commands.add_parser(
        "approve",
        help="Sign a shadow-research proposal with the owner's Ed25519 private key",
    )
    approve.add_argument("--proposal", required=True)
    approve.add_argument("--owner-private-key", required=True)
    approve.add_argument("--approver-id", required=True)
    approve.add_argument("--output", required=True)

    verify = commands.add_parser(
        "verify",
        help="Verify an owner approval against its proposal and public key",
    )
    verify.add_argument("--proposal", required=True)
    verify.add_argument("--approval", required=True)
    verify.add_argument("--owner-public-key", required=True)
    verify.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "propose":
            artifact = build_promotion_proposal(
                load_candidate_finalization(args.finalization),
                load_owner_public_key(args.owner_public_key),
            )
            write_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "approve":
            artifact = approve_promotion_proposal(
                load_promotion_proposal(args.proposal),
                load_owner_private_key(args.owner_private_key),
                approver_id=args.approver_id,
            )
            write_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        else:
            artifact = verify_promotion_approval(
                load_promotion_proposal(args.proposal),
                load_promotion_approval(args.approval),
                load_owner_public_key(args.owner_public_key),
            )
            result = {
                "ok": True,
                "candidate_id": artifact.candidate_id,
                "state": artifact.state,
                "approved_stage": artifact.approved_stage,
                "authority": artifact.authority,
                "approver_id": artifact.approver_id,
                "owner_key_fingerprint": artifact.owner_key_fingerprint,
                "proposal_digest": artifact.proposal_digest,
                "approval_digest": artifact.approval_digest,
                "production_deployment_allowed": False,
                "web3_runtime_integration_allowed": False,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except PromotionBlocked as exc:
        print(f"sentinel-promotion: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-promotion: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
