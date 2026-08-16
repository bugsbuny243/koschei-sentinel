from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_model_tournament import (
    load_tournament_policy,
    load_tournament_spec,
    run_blockchain_model_tournament,
    write_tournament_result,
)
from koschei_sentinel.blockchain_security_eval import load_eval_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare already-gated blockchain-security candidates on identical held-out "
            "lineage and a common runtime profile; security dominates efficiency"
        )
    )
    parser.add_argument("--tournament", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--eval-policy", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_blockchain_model_tournament(
            load_tournament_spec(args.tournament),
            load_tournament_policy(args.policy),
            load_eval_policy(args.eval_policy),
            root=args.root,
        )
        if args.output:
            write_tournament_result(result, args.output)
        print(
            json.dumps(
                {
                    "ready": result.ready,
                    "tournament_id": result.tournament_id,
                    "total_candidates": result.total_candidates,
                    "eligible_candidates": result.eligible_candidates,
                    "winner_candidate_id": result.winner_candidate_id,
                    "ranking": result.ranking,
                    "violations": result.violations,
                    "training_started": False,
                    "production_authority": False,
                    "automatic_production_promotion_allowed": False,
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.ready else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-tournament: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
