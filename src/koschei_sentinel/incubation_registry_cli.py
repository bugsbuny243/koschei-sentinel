from __future__ import annotations

import argparse
import json

from koschei_sentinel.incubation_registry import (
    CandidateRegistrationBlocked,
    append_candidate,
    build_candidate_record,
    load_adapter_manifest,
    load_autotrain_plan,
    load_comparison_matrix,
    load_registry,
    write_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register a benchmark-passing Koschei Sentinel adapter as an "
            "offline incubation candidate"
        )
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--autotrain-plan", required=True)
    parser.add_argument("--adapter-manifest", required=True)
    parser.add_argument("--comparison-matrix", required=True)
    parser.add_argument("--registry", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = build_candidate_record(
            args.candidate_id,
            load_autotrain_plan(args.autotrain_plan),
            load_adapter_manifest(args.adapter_manifest),
            load_comparison_matrix(args.comparison_matrix),
        )
        registry = append_candidate(load_registry(args.registry), record)
        write_registry(registry, args.registry)
        print(
            json.dumps(
                {
                    "ok": True,
                    "candidate": record.model_dump(mode="json"),
                    "registry_digest": registry.registry_digest,
                    "registered_candidates": len(registry.records),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except CandidateRegistrationBlocked as exc:
        print(f"sentinel-register-candidate: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-register-candidate: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
