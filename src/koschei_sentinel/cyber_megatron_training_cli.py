from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_megatron_training import (
    CyberMegatronResume,
    execute_cyber_megatron_sft,
    load_cyber_megatron_config,
    materialize_cyber_megatron_dataset,
    plan_cyber_megatron_sft,
)
from koschei_sentinel.training import atomic_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the single-model Qwen3.5-397B-A17B Megatron SFT path"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-output")
    parser.add_argument(
        "--materialize-dataset",
        action="store_true",
        help="Render the sealed Cyber corpus into deterministic ms-swift messages JSONL",
    )
    parser.add_argument("--resume-mcore-model")
    parser.add_argument("--resume-mcore-adapter")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch Megatron-SWIFT after dataset, runtime and paid-run approval checks",
    )
    return parser


def _resume(args: argparse.Namespace) -> CyberMegatronResume | None:
    if bool(args.resume_mcore_model) != bool(args.resume_mcore_adapter):
        raise ValueError("resume requires both --resume-mcore-model and --resume-mcore-adapter")
    if not args.resume_mcore_model:
        return None
    return CyberMegatronResume(
        mcore_model=args.resume_mcore_model,
        mcore_adapter=args.resume_mcore_adapter,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_cyber_megatron_config(args.config)
        resume = _resume(args)
        if args.materialize_dataset:
            materialize_cyber_megatron_dataset(config)
        if args.execute:
            result = execute_cyber_megatron_sft(config, resume=resume)
        else:
            result = plan_cyber_megatron_sft(config, resume=resume)
        payload = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.plan_output:
            atomic_write(Path(args.plan_output), payload)
        print(payload, end="")
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-megatron-sft: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
