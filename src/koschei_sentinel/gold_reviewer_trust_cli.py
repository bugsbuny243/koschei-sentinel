from __future__ import annotations

import argparse
import json

from koschei_sentinel.gold_reviewer_trust import (
    build_gold_reviewer_trust_policy,
    load_gold_reviewer_trust_policy,
    verify_gold_reviewer_trust_policy,
    write_gold_reviewer_trust_policy,
)
from koschei_sentinel.gold_review_signing import load_reviewer_public_key
from koschei_sentinel.promotion import load_owner_private_key, load_owner_public_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue or verify an owner-signed Gold reviewer trust policy"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    issue = commands.add_parser("issue")
    issue.add_argument("--policy-id", required=True)
    issue.add_argument("--reviewer-public-key", required=True)
    issue.add_argument("--owner-private-key", required=True)
    issue.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--policy", required=True)
    verify.add_argument("--reviewer-public-key", required=True)
    verify.add_argument("--owner-public-key", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "issue":
            policy = build_gold_reviewer_trust_policy(
                load_reviewer_public_key(args.reviewer_public_key),
                load_owner_private_key(args.owner_private_key),
                policy_id=args.policy_id,
            )
            write_gold_reviewer_trust_policy(policy, args.output)
        else:
            policy = verify_gold_reviewer_trust_policy(
                load_gold_reviewer_trust_policy(args.policy),
                load_reviewer_public_key(args.reviewer_public_key),
                load_owner_public_key(args.owner_public_key),
            )
        print(json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-reviewer-trust: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
