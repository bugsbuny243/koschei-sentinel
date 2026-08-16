from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_base_candidates import load_base_candidate_registry
from koschei_sentinel.blockchain_runtime_preflight import (
    audit_runtime_preflight,
    execute_runtime_probe,
    inspect_local_hardware,
    load_hardware_inventory,
    load_runtime_preflight_plan,
    load_runtime_preflight_policy,
    load_runtime_probe_receipt,
    plan_runtime_preflight,
    write_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and execute fail-closed hardware/runtime probes for pinned base models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inspect local hardware without model I/O")
    inventory.add_argument("--inventory-id", default="local-runtime")
    inventory.add_argument("--output", required=True)

    plan = subparsers.add_parser("plan", help="Build an offline hardware-fit plan")
    _common_plan_arguments(plan)
    plan.add_argument("--output", required=True)

    execute = subparsers.add_parser("execute", help="Run one quantized LoRA forward/backward probe")
    _common_plan_arguments(execute)
    execute.add_argument("--plan", required=True)
    execute.add_argument("--output", required=True)

    audit = subparsers.add_parser("audit", help="Verify a completed runtime probe receipt")
    _common_plan_arguments(audit)
    audit.add_argument("--plan", required=True)
    audit.add_argument("--receipt", required=True)
    audit.add_argument("--output")
    return parser


def _common_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--policy", required=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            inventory = inspect_local_hardware(args.inventory_id)
            write_model(inventory, args.output)
            _print(
                {
                    "inventory_id": inventory.inventory_id,
                    "cuda_available": inventory.cuda_available,
                    "gpu_count": inventory.gpu_count,
                    "gpu_model": inventory.gpu_model,
                    "vram_per_gpu_mb": inventory.vram_per_gpu_mb,
                    "host_ram_mb": inventory.host_ram_mb,
                    "free_disk_mb": inventory.free_disk_mb,
                    "bfloat16_supported": inventory.bfloat16_supported,
                    "inventory_digest": inventory.inventory_digest,
                    "model_download_started": False,
                    "training_started": False,
                    "output": args.output,
                }
            )
            return 0

        registry = load_base_candidate_registry(args.registry)
        hardware = load_hardware_inventory(args.hardware)
        policy = load_runtime_preflight_policy(args.policy)

        if args.command == "plan":
            plan = plan_runtime_preflight(registry, args.candidate_id, hardware, policy)
            write_model(plan, args.output)
            _print(
                {
                    "candidate_id": plan.candidate_id,
                    "model_id": plan.model_id,
                    "revision": plan.revision,
                    "estimated_checkpoint_disk_mb": plan.estimated_checkpoint_disk_mb,
                    "estimated_quantized_weight_mb": plan.estimated_quantized_weight_mb,
                    "estimated_min_vram_mb": plan.estimated_min_vram_mb,
                    "estimated_min_host_ram_mb": plan.estimated_min_host_ram_mb,
                    "available_single_gpu_vram_mb": plan.available_single_gpu_vram_mb,
                    "available_host_ram_mb": plan.available_host_ram_mb,
                    "available_free_disk_mb": plan.available_free_disk_mb,
                    "static_hardware_fit": plan.static_hardware_fit,
                    "runtime_probe_authorized": plan.runtime_probe_authorized,
                    "blockers": plan.blockers,
                    "model_download_started": False,
                    "training_started": False,
                    "training_authorized": False,
                    "output": args.output,
                }
            )
            return 0 if plan.runtime_probe_authorized else 3

        plan = load_runtime_preflight_plan(args.plan)
        if plan.candidate_id != args.candidate_id:
            raise ValueError("candidate-id does not match runtime preflight plan")

        if args.command == "execute":
            receipt = execute_runtime_probe(registry, hardware, policy, plan)
            write_model(receipt, args.output)
            _print(
                {
                    "candidate_id": receipt.candidate_id,
                    "status": receipt.status,
                    "tokenizer_loaded": receipt.tokenizer_loaded,
                    "config_loaded": receipt.config_loaded,
                    "quantized_model_loaded": receipt.quantized_model_loaded,
                    "lora_attached": receipt.lora_attached,
                    "forward_ok": receipt.forward_ok,
                    "backward_ok": receipt.backward_ok,
                    "exact_revision_observed": receipt.exact_revision_observed,
                    "probe_input_tokens": receipt.probe_input_tokens,
                    "trainable_parameters": receipt.trainable_parameters,
                    "total_parameters": receipt.total_parameters,
                    "peak_gpu_memory_mb": receipt.peak_gpu_memory_mb,
                    "error_code": receipt.error_code,
                    "trust_remote_code_used": False,
                    "raw_model_outputs_stored": False,
                    "training_started": False,
                    "training_authorized": False,
                    "output": args.output,
                }
            )
            return 0 if receipt.status == "PASSED" else 4

        receipt = load_runtime_probe_receipt(args.receipt)
        audit = audit_runtime_preflight(registry, hardware, policy, plan, receipt)
        if args.output:
            write_model(audit, args.output)
        _print(
            {
                "candidate_id": audit.candidate_id,
                "ready_for_training_authorization_review": (
                    audit.ready_for_training_authorization_review
                ),
                "static_hardware_fit": audit.static_hardware_fit,
                "runtime_probe_passed": audit.runtime_probe_passed,
                "peak_gpu_memory_mb": audit.peak_gpu_memory_mb,
                "available_single_gpu_vram_mb": audit.available_single_gpu_vram_mb,
                "headroom_mb": audit.headroom_mb,
                "violations": audit.violations,
                "training_started": False,
                "production_authority": False,
                "output": args.output,
            }
        )
        return 0 if audit.ready_for_training_authorization_review else 5
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-preflight: {exc}")
        return 2


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
