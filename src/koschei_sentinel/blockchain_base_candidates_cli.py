from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_base_candidates import (
    audit_base_candidate_registry,
    load_base_candidate_policy,
    load_base_candidate_registry,
    write_base_candidate_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact-pinned blockchain base-model candidates before any model download "
            "or GPU preflight is authorized"
        )
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_base_candidate_registry(
            load_base_candidate_registry(args.registry),
            load_base_candidate_policy(args.policy),
        )
        if args.output:
            write_base_candidate_audit(audit, args.output)
        print(
            json.dumps(
                {
                    "ready_for_preflight": audit.ready_for_preflight,
                    "candidates": audit.candidates,
                    "admitted_for_preflight": audit.admitted_for_preflight,
                    "ready_for_training_authorization_review": (
                        audit.ready_for_training_authorization_review
                    ),
                    "blocked_candidates": audit.blocked_candidates,
                    "violations": audit.violations,
                    "model_download_started": False,
                    "training_started": False,
                    "production_authority": False,
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if audit.ready_for_preflight else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-base-candidates: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
