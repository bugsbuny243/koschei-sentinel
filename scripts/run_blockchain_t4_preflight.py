from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.blockchain_base_candidates import load_base_candidate_registry
from koschei_sentinel.blockchain_runtime_preflight import (
    audit_runtime_preflight,
    execute_runtime_probe,
    inspect_local_hardware,
    load_runtime_preflight_policy,
    plan_runtime_preflight,
    write_model,
)

DEFAULT_CANDIDATE = "qwen2.5-coder-7b-base"
DEFAULT_REGISTRY = "configs/models/blockchain-base-candidates.v1.json"
DEFAULT_POLICY = "configs/models/blockchain-runtime-preflight-policy.v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete fail-closed T4 feasibility preflight for one pinned "
            "blockchain base-model candidate. This script never starts training."
        )
    )
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inventory-id", default="colab-t4-preflight")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"preflight output directory must be new or empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_base_candidate_registry(args.registry)
    policy = load_runtime_preflight_policy(args.policy)
    hardware = inspect_local_hardware(args.inventory_id)

    inventory_path = output_dir / "hardware-inventory.json"
    plan_path = output_dir / "preflight-plan.json"
    receipt_path = output_dir / "runtime-probe-receipt.json"
    audit_path = output_dir / "preflight-audit.json"
    summary_path = output_dir / "summary.json"

    write_model(hardware, inventory_path)
    plan = plan_runtime_preflight(registry, args.candidate_id, hardware, policy)
    write_model(plan, plan_path)

    summary: dict[str, object] = {
        "candidate_id": plan.candidate_id,
        "model_id": plan.model_id,
        "revision": plan.revision,
        "gpu_model": hardware.gpu_model,
        "available_single_gpu_vram_mb": hardware.vram_per_gpu_mb,
        "available_host_ram_mb": hardware.host_ram_mb,
        "available_free_disk_mb": hardware.free_disk_mb,
        "estimated_checkpoint_disk_mb": plan.estimated_checkpoint_disk_mb,
        "estimated_quantized_weight_mb": plan.estimated_quantized_weight_mb,
        "estimated_min_vram_mb": plan.estimated_min_vram_mb,
        "estimated_min_host_ram_mb": plan.estimated_min_host_ram_mb,
        "static_hardware_fit": plan.static_hardware_fit,
        "runtime_probe_authorized": plan.runtime_probe_authorized,
        "blockers": plan.blockers,
        "runtime_probe_status": "NOT_RUN",
        "ready_for_training_authorization_review": False,
        "training_started": False,
        "training_authorized": False,
        "production_authority": False,
        "raw_model_outputs_stored": False,
    }

    if not plan.runtime_probe_authorized:
        _write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 3

    receipt = execute_runtime_probe(registry, hardware, policy, plan)
    write_model(receipt, receipt_path)
    audit = audit_runtime_preflight(registry, hardware, policy, plan, receipt)
    write_model(audit, audit_path)

    summary.update(
        {
            "runtime_probe_status": receipt.status,
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
            "headroom_mb": audit.headroom_mb,
            "error_code": receipt.error_code,
            "violations": audit.violations,
            "ready_for_training_authorization_review": (
                audit.ready_for_training_authorization_review
            ),
        }
    )
    _write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if audit.ready_for_training_authorization_review else 4


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"preflight summary already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
