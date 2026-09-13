from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_target_candidate_matrix import (
    load_production_target_candidate_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the reviewed public-model compatibility matrix for the 397B/35B target"
    )
    parser.add_argument("--matrix", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        matrix = load_production_target_candidate_matrix(args.matrix)
        payload = {
            "schema_version": "sentinel.production-target-candidate-matrix-result.v1",
            "valid": True,
            "as_of": matrix.as_of,
            "target": matrix.target.model_dump(mode="json"),
            "exact_public_base_found": matrix.decision.exact_public_base_found,
            "production_binding_allowed": matrix.decision.production_binding_allowed,
            "systems_validation_primary": matrix.decision.systems_validation_primary,
            "active_width_reference": matrix.decision.active_width_reference,
            "balanced_moe_reference": matrix.decision.balanced_moe_reference,
            "reviewed_candidate_count": len(matrix.candidates),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-production-candidate-matrix: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
