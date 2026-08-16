from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.blockchain_security_eval import (
    audit_blockchain_security_eval,
    load_eval_policy,
    load_eval_receipt,
    write_eval_audit,
)
from koschei_sentinel.blockchain_training import BlockchainAdapterManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate a blockchain-security adapter using an aggregate held-out evaluation "
            "receipt without persisting raw model outputs"
        )
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--adapter-manifest", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = load_eval_receipt(args.receipt)
        adapter = BlockchainAdapterManifest.model_validate_json(
            Path(args.adapter_manifest).read_text(encoding="utf-8")
        )
        policy = load_eval_policy(args.policy)
        audit = audit_blockchain_security_eval(receipt, adapter, policy)
        if args.output:
            write_eval_audit(audit, args.output)
        print(
            json.dumps(
                {
                    "ready": audit.ready,
                    "candidate_id": audit.candidate_id,
                    "total_cases": audit.total_cases,
                    "passed_cases": audit.passed_cases,
                    "overall_pass_bps": audit.overall_pass_bps,
                    "grounding_bps": audit.grounding_bps,
                    "task_correct_bps": audit.task_correct_bps,
                    "patch_safe_bps": audit.patch_safe_bps,
                    "abstention_correct_bps": audit.abstention_correct_bps,
                    "authority_failures": audit.authority_failures,
                    "privacy_failures": audit.privacy_failures,
                    "verdict_identity_failures": audit.verdict_identity_failures,
                    "confidence_failures": audit.confidence_failures,
                    "family_isolation_failures": audit.family_isolation_failures,
                    "raw_output_storage_failures": audit.raw_output_storage_failures,
                    "violations": audit.violations,
                    "training_started": False,
                    "production_authority": False,
                    "raw_model_outputs_stored": False,
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if audit.ready else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-eval-gate: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
