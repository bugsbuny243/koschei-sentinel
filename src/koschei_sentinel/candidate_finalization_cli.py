from __future__ import annotations

import argparse
import json

from koschei_sentinel.candidate_finalization import (
    CandidateFinalizationBlocked,
    finalize_candidate,
    load_offline_training_receipt,
    write_finalization_bundle,
)
from koschei_sentinel.incubation_registry import (
    load_adapter_manifest,
    load_autotrain_plan,
    load_comparison_matrix,
    load_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a verified offline receipt and passing benchmark into one "
            "sealed incubation-registry finalization bundle"
        )
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--autotrain-plan", required=True)
    parser.add_argument("--adapter-manifest", required=True)
    parser.add_argument("--comparison-matrix", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        finalization, updated_registry, record = finalize_candidate(
            args.candidate_id,
            load_offline_training_receipt(args.receipt),
            load_autotrain_plan(args.autotrain_plan),
            load_adapter_manifest(args.adapter_manifest),
            load_comparison_matrix(args.comparison_matrix),
            load_registry(args.registry),
        )
        write_finalization_bundle(
            finalization,
            updated_registry,
            args.output_dir,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "candidate_id": record.candidate_id,
                    "candidate_record_digest": record.record_digest,
                    "finalization_digest": finalization.finalization_digest,
                    "updated_registry_digest": updated_registry.registry_digest,
                    "output_dir": args.output_dir,
                    "automatic_promotion_allowed": False,
                    "production_deployment_allowed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except CandidateFinalizationBlocked as exc:
        print(f"sentinel-finalize-candidate: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-finalize-candidate: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
