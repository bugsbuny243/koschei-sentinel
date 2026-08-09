from __future__ import annotations

import argparse
import json

from koschei_sentinel.shadow_replay import (
    ShadowReplayBlocked,
    build_shadow_replay_plan,
    load_promotion_approval,
    load_promotion_policy,
    load_promotion_proposal,
    write_shadow_replay_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sealed, manually dispatched Sentinel shadow-research plan "
            "for an immutable offline JSONL replay dataset"
        )
    )
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_shadow_replay_plan(
            load_promotion_proposal(args.proposal),
            load_promotion_approval(args.approval),
            load_promotion_policy(args.policy),
            owner_public_key_path=args.owner_public_key,
            replay_path=args.replay,
            output_dir=args.output_dir,
            root=args.root,
        )
        write_shadow_replay_plan(plan, args.output)
        print(
            json.dumps(
                {
                    "ok": True,
                    "candidate_id": plan.candidate_id,
                    "state": plan.state,
                    "replay_cases": plan.replay_cases,
                    "replay_sha256": plan.replay_sha256,
                    "promotion_policy_digest": plan.promotion_policy_digest,
                    "plan_digest": plan.plan_digest,
                    "manual_dispatch_required": True,
                    "network_access_allowed": False,
                    "live_chain_reads_allowed": False,
                    "live_customer_traffic_allowed": False,
                    "verdict_mutation_allowed": False,
                    "production_deployment_allowed": False,
                    "web3_runtime_integration_allowed": False,
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ShadowReplayBlocked as exc:
        print(f"sentinel-shadow-plan: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"sentinel-shadow-plan: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
