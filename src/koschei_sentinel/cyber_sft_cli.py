from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_sft_trainer import execute_cyber_sft
from koschei_sentinel.cyber_sft_training import load_cyber_sft_config, plan_cyber_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute evidence-grounded Koschei Sentinel Cyber SFT"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-output")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run QLoRA after local corpus planning; default is a network-free dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_cyber_sft_config(args.config)
        plan = plan_cyber_sft(config)
        if args.plan_output:
            destination = Path(args.plan_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        result = execute_cyber_sft(config, plan) if args.execute else plan
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
