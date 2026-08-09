from __future__ import annotations

import argparse
import json

from koschei_sentinel.owner_rotation import (
    OwnerRotationBlocked,
    accept_owner_rotation,
    approve_owner_rotation_proposal,
    build_owner_rotation_checkpoint,
    build_owner_rotation_proposal,
    claim_owner_rotation,
    load_current_owner_rotation_approval,
    load_next_owner_rotation_acceptance,
    load_owner_rotation_checkpoint,
    load_owner_rotation_proposal,
    verify_owner_rotation_checkpoint,
    write_owner_rotation_artifact,
)
from koschei_sentinel.promotion import (
    load_owner_private_key,
    load_owner_public_key,
)
from koschei_sentinel.shadow_baseline import load_shadow_baseline_lineage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-owner-rotation",
        description="Create dual-signed, non-activating owner-key governance checkpoints",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    _add_lineage_and_public_keys(propose)
    propose.add_argument("--output", required=True)

    approve = commands.add_parser("approve-current")
    approve.add_argument("--proposal", required=True)
    approve.add_argument("--current-owner-private-key", required=True)
    approve.add_argument("--approver-id", required=True)
    approve.add_argument("--output", required=True)

    accept = commands.add_parser("accept-next")
    accept.add_argument("--proposal", required=True)
    accept.add_argument("--current-approval", required=True)
    accept.add_argument("--current-owner-public-key", required=True)
    accept.add_argument("--next-owner-private-key", required=True)
    accept.add_argument("--accepter-id", required=True)
    accept.add_argument("--output", required=True)

    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("--proposal", required=True)
    checkpoint.add_argument("--current-approval", required=True)
    checkpoint.add_argument("--next-acceptance", required=True)
    _add_lineage_and_public_keys(checkpoint)
    checkpoint.add_argument("--claim-dir", required=True)
    checkpoint.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    _add_lineage_and_public_keys(verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "propose":
            artifact = build_owner_rotation_proposal(
                load_shadow_baseline_lineage(args.lineage),
                load_owner_public_key(args.current_owner_public_key),
                load_owner_public_key(args.next_owner_public_key),
            )
            write_owner_rotation_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "approve-current":
            artifact = approve_owner_rotation_proposal(
                load_owner_rotation_proposal(args.proposal),
                load_owner_private_key(args.current_owner_private_key),
                approver_id=args.approver_id,
            )
            write_owner_rotation_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "accept-next":
            artifact = accept_owner_rotation(
                load_owner_rotation_proposal(args.proposal),
                load_current_owner_rotation_approval(args.current_approval),
                load_owner_public_key(args.current_owner_public_key),
                load_owner_private_key(args.next_owner_private_key),
                accepter_id=args.accepter_id,
            )
            write_owner_rotation_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "checkpoint":
            lineage = load_shadow_baseline_lineage(args.lineage)
            current_key = load_owner_public_key(args.current_owner_public_key)
            next_key = load_owner_public_key(args.next_owner_public_key)
            artifact = build_owner_rotation_checkpoint(
                load_owner_rotation_proposal(args.proposal),
                load_current_owner_rotation_approval(args.current_approval),
                load_next_owner_rotation_acceptance(args.next_acceptance),
                lineage,
                current_key,
                next_key,
            )
            claim_path = claim_owner_rotation(
                artifact,
                lineage,
                current_key,
                next_key,
                args.claim_dir,
            )
            write_owner_rotation_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
            result["rotation_claim_path"] = str(claim_path)
        else:
            artifact = verify_owner_rotation_checkpoint(
                load_owner_rotation_checkpoint(args.checkpoint),
                load_shadow_baseline_lineage(args.lineage),
                load_owner_public_key(args.current_owner_public_key),
                load_owner_public_key(args.next_owner_public_key),
            )
            result = {
                "ok": True,
                "checkpoint_digest": artifact.checkpoint_digest,
                "current_owner_key_fingerprint": artifact.current_owner_key_fingerprint,
                "next_owner_key_fingerprint": artifact.next_owner_key_fingerprint,
                "dual_signature_verified": True,
                "baseline_rekey_required": True,
                "automatic_key_activation_allowed": False,
                "production_deployment_allowed": False,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except OwnerRotationBlocked as exc:
        print(f"sentinel-owner-rotation: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-owner-rotation: {exc}")
        return 2


def _add_lineage_and_public_keys(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--current-owner-public-key", required=True)
    parser.add_argument("--next-owner-public-key", required=True)


if __name__ == "__main__":
    raise SystemExit(main())
