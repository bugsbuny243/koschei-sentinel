from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.full_release import (
    FullReleaseAuthority,
    FullReleaseProposal,
    approve_full_release,
    build_full_release_proposal,
    load_canary_evidence,
    verify_full_release,
)
from koschei_sentinel.production_authority import ProductionAuthority
from koschei_sentinel.promotion import load_owner_private_key, load_owner_public_key


def _load_model(path: str, model_type):
    return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koschei Sentinel signed full-release gate")
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    propose.add_argument("--canary-authority", required=True)
    propose.add_argument("--canary-evidence", required=True)
    propose.add_argument("--owner-public-key", required=True)
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


def _write(path: str, model) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "propose":
            artifact = build_full_release_proposal(
                _load_model(args.canary_authority, ProductionAuthority),
                load_canary_evidence(args.canary_evidence),
                load_owner_public_key(args.owner_public_key),
            )
            _write(args.output, artifact)
        elif args.command == "approve":
            artifact = approve_full_release(
                _load_model(args.proposal, FullReleaseProposal),
                load_owner_private_key(args.owner_private_key),
                approver_id=args.approver_id,
            )
            _write(args.output, artifact)
        else:
            artifact = verify_full_release(
                _load_model(args.proposal, FullReleaseProposal),
                _load_model(args.authority, FullReleaseAuthority),
                load_owner_public_key(args.owner_public_key),
            )
        print(json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-full-release: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
