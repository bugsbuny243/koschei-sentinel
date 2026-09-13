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
from koschei_sentinel.production_target_training_gate import verify_production_training_gate
from koschei_sentinel.training import atomic_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the Megatron SFT validation lane; production 397B/35B requires a verified binding"
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
        "--resume-binding-manifest",
        help="Bound koschei-run-identity.json from the existing output directory",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch Megatron-SWIFT after dataset, runtime and paid-run approval checks",
    )
    parser.add_argument("--production-target-binding")
    parser.add_argument("--production-architecture-manifest")
    parser.add_argument("--production-router-manifest")
    parser.add_argument("--production-expert-topology-manifest")
    return parser


def _resume(args: argparse.Namespace) -> CyberMegatronResume | None:
    values = (
        args.resume_mcore_model,
        args.resume_mcore_adapter,
        args.resume_binding_manifest,
    )
    if any(values) and not all(values):
        raise ValueError(
            "resume requires --resume-mcore-model, --resume-mcore-adapter and "
            "--resume-binding-manifest"
        )
    if not any(values):
        return None
    return CyberMegatronResume(
        mcore_model=args.resume_mcore_model,
        mcore_adapter=args.resume_mcore_adapter,
        binding_manifest=args.resume_binding_manifest,
    )


def _verify_production_lane(args: argparse.Namespace, config) -> None:
    values = (
        args.production_target_binding,
        args.production_architecture_manifest,
        args.production_router_manifest,
        args.production_expert_topology_manifest,
    )
    if any(values) and not all(values):
        raise ValueError(
            "production lane requires binding, architecture, router and expert-topology manifests"
        )
    if not any(values):
        return
    verify_production_training_gate(
        binding_path=args.production_target_binding,
        architecture_manifest_path=args.production_architecture_manifest,
        router_manifest_path=args.production_router_manifest,
        expert_topology_manifest_path=args.production_expert_topology_manifest,
        trainer_model_ref=config.model,
        trainer_model_revision=config.model_revision,
        trainer_adapter=config.backend,
        trainer_adapter_version=config.ms_swift_version,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_cyber_megatron_config(args.config)
        _verify_production_lane(args, config)
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
