from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_training import load_blockchain_training_config
from koschei_sentinel.blockchain_training_authorization import (
    approve_training_authorization_proposal,
    build_training_authorization_proposal,
    load_training_authorization_approval,
    load_training_authorization_policy,
    load_training_authorization_proposal,
    verify_training_authorization_bundle,
    write_training_authorization_artifact,
)
from koschei_sentinel.promotion import (
    load_owner_private_key,
    load_owner_public_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, owner-sign, and verify fail-closed blockchain training authorization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    proposal = subparsers.add_parser("proposal")
    _proposal_inputs(proposal)
    proposal.add_argument("--authorization-id", required=True)
    proposal.add_argument("--owner-public-key", required=True)
    proposal.add_argument("--output", required=True)

    approve = subparsers.add_parser("approve")
    approve.add_argument("--proposal", required=True)
    approve.add_argument("--policy", required=True)
    approve.add_argument("--owner-private-key", required=True)
    approve.add_argument("--approver-id", required=True)
    approve.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    _proposal_inputs(verify)
    verify.add_argument("--proposal", required=True)
    verify.add_argument("--approval", required=True)
    verify.add_argument("--owner-public-key", required=True)
    return parser


def _proposal_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seal", required=True)
    parser.add_argument("--preflight-run", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--root", default=".")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = load_training_authorization_policy(args.policy)
        if args.command == "approve":
            proposal = load_training_authorization_proposal(args.proposal)
            approval = approve_training_authorization_proposal(
                proposal,
                load_owner_private_key(args.owner_private_key),
                policy,
                approver_id=args.approver_id,
            )
            write_training_authorization_artifact(approval, args.output)
            _print(
                {
                    "ok": True,
                    "state": approval.state,
                    "authorization_id": approval.authorization_id,
                    "candidate_id": approval.candidate_id,
                    "training_authorized": approval.training_authorized,
                    "training_started": approval.training_started,
                    "production_authority": approval.production_authority,
                    "approval_digest": approval.approval_digest,
                    "output": args.output,
                }
            )
            return 0

        config = load_blockchain_training_config(args.training_config)
        public_key = load_owner_public_key(args.owner_public_key)
        if args.command == "proposal":
            proposal = build_training_authorization_proposal(
                authorization_id=args.authorization_id,
                seal_path=args.seal,
                preflight_run_dir=args.preflight_run,
                registry_path=args.registry,
                release_dir=args.release,
                training_config=config,
                owner_public_key=public_key,
                policy=policy,
                root=args.root,
            )
            write_training_authorization_artifact(proposal, args.output)
            _print(
                {
                    "ok": True,
                    "state": proposal.state,
                    "authorization_id": proposal.authorization_id,
                    "candidate_id": proposal.candidate_id,
                    "model_id": proposal.model_id,
                    "model_revision": proposal.model_revision,
                    "source_corpus_digest": proposal.source_corpus_digest,
                    "benchmark_suite_digest": proposal.benchmark_suite_digest,
                    "training_authorized": proposal.training_authorized,
                    "training_started": proposal.training_started,
                    "production_authority": proposal.production_authority,
                    "proposal_digest": proposal.proposal_digest,
                    "output": args.output,
                }
            )
            return 0

        proposal = load_training_authorization_proposal(args.proposal)
        approval = load_training_authorization_approval(args.approval)
        verified = verify_training_authorization_bundle(
            proposal=proposal,
            approval=approval,
            owner_public_key=public_key,
            policy=policy,
            seal_path=args.seal,
            preflight_run_dir=args.preflight_run,
            registry_path=args.registry,
            release_dir=args.release,
            training_config=config,
            root=args.root,
        )
        _print(
            {
                "ok": True,
                "state": verified.state,
                "authorization_id": verified.authorization_id,
                "candidate_id": verified.candidate_id,
                "training_authorized": verified.training_authorized,
                "training_started": verified.training_started,
                "production_authority": verified.production_authority,
                "approval_digest": verified.approval_digest,
            }
        )
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-authorize: {exc}")
        return 2


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
