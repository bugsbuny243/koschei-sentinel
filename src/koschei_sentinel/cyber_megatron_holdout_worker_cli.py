from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_megatron_holdout_worker import (
    CyberMegatronHoldoutWorkerProfile,
    execute_cyber_megatron_holdout_worker,
    finalize_cyber_megatron_holdout_worker,
    prepare_cyber_megatron_holdout_worker,
    verify_cyber_megatron_holdout_output,
)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--holdout-plan", required=True)
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


def _source_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "holdout_plan_path": args.holdout_plan,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, execute, finalize, or independently verify the paid "
            "Qwen3.5-397B-A17B Gold HOLDOUT worker"
        )
    )
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_sources(prepare)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--nodes", type=int, default=4)
    prepare.add_argument("--gpus-per-node", type=int, default=8)
    prepare.add_argument("--tensor-parallel-size", type=int, default=8)
    prepare.add_argument("--pipeline-parallel-size", type=int, default=4)
    prepare.add_argument("--vllm-version", default="0.17.0")
    prepare.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    prepare.add_argument("--max-model-len", type=int, default=8192)
    prepare.add_argument("--max-num-seqs", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=1701)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--worker-dir", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--worker-dir", required=True)
    finalize.add_argument("--inference-pack", required=True)

    verify = subparsers.add_parser("verify")
    _add_sources(verify)
    verify.add_argument("--worker-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            profile = CyberMegatronHoldoutWorkerProfile(
                vllm_version=args.vllm_version,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                seed=args.seed,
            )
            result = prepare_cyber_megatron_holdout_worker(
                **_source_kwargs(args),
                output_dir=args.output_dir,
                profile=profile,
            )
            payload = result.model_dump(mode="json")
            code = 0
        elif args.command == "execute":
            return execute_cyber_megatron_holdout_worker(
                worker_dir=args.worker_dir,
                root=args.root,
            )
        elif args.command == "finalize":
            result = finalize_cyber_megatron_holdout_worker(
                worker_dir=args.worker_dir,
                inference_pack=args.inference_pack,
                root=args.root,
            )
            payload = result.model_dump(mode="json")
            code = 0
        else:
            result = verify_cyber_megatron_holdout_output(
                **_source_kwargs(args),
                worker_dir=args.worker_dir,
            )
            payload = result.model_dump(mode="json")
            code = 0 if result.valid else 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return code
    except (FileExistsError, OSError, PermissionError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-megatron-holdout-worker: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
