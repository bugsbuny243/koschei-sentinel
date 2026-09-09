from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_megatron_holdout import (
    build_cyber_megatron_holdout_plan,
    verify_cyber_megatron_holdout_plan,
)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-plan", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--pack-signature", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--minimum-case-count", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a verified Qwen3.5-397B-A17B Megatron candidate to an "
            "owner-trusted answer-key-isolated Gold HOLDOUT pack"
        )
    )
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    _add_sources(plan)
    plan.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    _add_sources(verify)
    verify.add_argument("--plan", required=True)
    return parser


def _source_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "candidate_manifest_path": args.candidate,
        "candidate_config_path": args.config,
        "candidate_training_plan_path": args.training_plan,
        "checkpoint_dir": args.checkpoint,
        "inference_pack": args.inference_pack,
        "signature_path": args.pack_signature,
        "reviewer_public_key_path": args.reviewer_public_key,
        "reviewer_trust_policy_path": args.reviewer_trust_policy,
        "owner_public_key_path": args.owner_public_key,
        "minimum_case_count": args.minimum_case_count,
        "root": args.root,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kwargs = _source_kwargs(args)
        if args.command == "plan":
            result = build_cyber_megatron_holdout_plan(
                **kwargs,
                output_path=args.output,
            )
            payload = result.model_dump(mode="json")
            code = 0
        else:
            result = verify_cyber_megatron_holdout_plan(
                **kwargs,
                plan_path=args.plan,
            )
            payload = result.model_dump(mode="json")
            code = 0 if result.valid else 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return code
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-megatron-holdout: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
