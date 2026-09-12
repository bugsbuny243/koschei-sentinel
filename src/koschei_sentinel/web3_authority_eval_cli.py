from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.web3_authority_eval_contract import load_and_validate_eval_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for Sentinel Web3 authority/evidence eval bundles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/web3-authority-evals.v1.json"),
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("configs/training/web3-authority-eval-seeds.v1.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config, seeds = load_and_validate_eval_bundle(args.config, args.seeds)
    result = {
        "status": "PASS",
        "schema_version": config.schema_version,
        "families": len(config.families),
        "seed_cases": len(seeds.cases),
        "target_architecture": {
            "total_parameters": config.target_architecture.total_parameters,
            "active_parameters": config.target_architecture.active_parameters,
        },
        "safety": "defensive_authorized_use_only",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
