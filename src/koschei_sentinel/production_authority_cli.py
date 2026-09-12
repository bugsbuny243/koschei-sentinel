from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.production_authority import (
    ProductionAuthority,
    ProductionAuthorityBlocked,
    ProductionAuthorityProposal,
    approve_production_authority,
    build_production_authority_proposal,
    verify_production_authority,
)
from koschei_sentinel.promotion import (
    load_owner_private_key,
    load_owner_public_key,
    write_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, owner-sign, or verify Sentinel production canary authority"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    propose.add_argument("--finalization", required=True)
    propose.add_argument("--holdout-evidence", action="append", required=True)
    propose.add_argument("--owner-public-key", required=True)
    propose.add_argument("--max-initial-traffic-percent", type=int, default=5)
    propose.add_argument("--output", required=True)

    approve = commands.add_parser("approve")
    approve.add_argument("--proposal", required=True)
    approve.add_argument("--owner-private-key", required=True)
    approve.add_argument("--approver-id", required=True)
    approve.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--proposal", required=True)
    verify.add_argument("--authority", required=True)
    verify.add_argument("--owner-public-key", required=True)
    return parser


def _load(path: str, model):
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "propose":
            finalization = _load(args.finalization, CandidateFinalization)
            artifact = build_production_authority_proposal(
                finalization,
                args.holdout_evidence,
                load_owner_public_key(args.owner_public_key),
                max_initial_traffic_percent=args.max_initial_traffic_percent,
            )
            write_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        elif args.command == "approve":
            proposal = _load(args.proposal, ProductionAuthorityProposal)
            artifact = approve_production_authority(
                proposal,
                load_owner_private_key(args.owner_private_key),
                approver_id=args.approver_id,
            )
            write_artifact(artifact, args.output)
            result = artifact.model_dump(mode="json")
        else:
            proposal = _load(args.proposal, ProductionAuthorityProposal)
            authority = _load(args.authority, ProductionAuthority)
            verified = verify_production_authority(
                proposal, authority, load_owner_public_key(args.owner_public_key)
            )
            result = {
                "ok": True,
                "candidate_id": verified.candidate_id,
                "deployment_scope": verified.deployment_scope,
                "max_initial_traffic_percent": verified.max_initial_traffic_percent,
                "rollback_required": verified.rollback_required,
                "emergency_disable_required": verified.emergency_disable_required,
                "automatic_expansion_allowed": verified.automatic_expansion_allowed,
                "production_deployment_allowed": verified.production_deployment_allowed,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, ProductionAuthorityBlocked, TypeError, ValueError) as exc:
        print(f"sentinel-production-authority: blocked: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
