from __future__ import annotations

import argparse
import json

from koschei_sentinel.shadow_receipt import (
    ShadowReceiptBlocked,
    build_shadow_replay_receipt,
    write_shadow_replay_receipt,
)
from koschei_sentinel.shadow_replay import load_shadow_replay_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal one completed offline shadow replay against its immutable plan "
            "and exact result bytes"
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_shadow_replay_receipt(
            load_shadow_replay_plan(args.plan),
            results_path=args.results,
            root=args.root,
        )
        write_shadow_replay_receipt(receipt, args.output)
    except (OSError, ValueError, ShadowReceiptBlocked) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "state": "blocked",
                    "message": str(exc),
                    "production_deployment_allowed": False,
                    "web3_runtime_integration_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "candidate_id": receipt.candidate_id,
                "state": receipt.state,
                "plan_digest": receipt.plan_digest,
                "results_sha256": receipt.results_sha256,
                "results_cases": receipt.results_cases,
                "receipt_digest": receipt.receipt_digest,
                "manual_review_required": True,
                "production_deployment_allowed": False,
                "web3_runtime_integration_allowed": False,
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
