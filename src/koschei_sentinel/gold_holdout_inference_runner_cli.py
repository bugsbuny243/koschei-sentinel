from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    build_gold_holdout_inference_plan,
    execute_gold_holdout_inference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute answer-key-isolated Gold HOLDOUT inference with a verified "
            "Cyber SFT adapter"
        )
    )
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--generation-policy", required=True)
    parser.add_argument("--plan-output")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load the pinned base model + verified adapter and generate HOLDOUT predictions",
    )
    return parser


def _load_policy(path: str) -> GoldHoldoutGenerationPolicy:
    try:
        return GoldHoldoutGenerationPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Gold HOLDOUT generation policy: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_policy = _load_policy(args.generation_policy)
        plan = build_gold_holdout_inference_plan(
            inference_pack_dir=args.inference_pack,
            run_dir=args.run_dir,
            training_config_path=args.training_config,
            model_ref=args.model_ref,
            generation_policy=selected_policy,
        )
        if args.plan_output:
            destination = Path(args.plan_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if not args.execute:
            print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required with --execute")
        receipt = execute_gold_holdout_inference(
            inference_pack_dir=args.inference_pack,
            run_dir=args.run_dir,
            training_config_path=args.training_config,
            model_ref=args.model_ref,
            output_dir=args.output_dir,
            generation_policy=selected_policy,
        )
        print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-infer: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
