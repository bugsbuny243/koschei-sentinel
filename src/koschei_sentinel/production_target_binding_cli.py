from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_target_binding import (
    load_production_target_binding,
    require_verified_production_target_binding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the Koschei Sentinel 397B/35B production target binding"
    )
    parser.add_argument("--binding", required=True)
    parser.add_argument(
        "--require-production-ready",
        action="store_true",
        help="Fail unless the binding is verified and explicitly authorizes production training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        binding = load_production_target_binding(args.binding)
        if args.require_production_ready:
            require_verified_production_target_binding(binding)
        payload = {
            "status": "PASS",
            "schema_version": binding.schema_version,
            "total_parameters": binding.total_parameters,
            "active_parameters": binding.active_parameters,
            "model_ref": binding.model_ref,
            "verification_status": binding.verification_status,
            "production_training_allowed": binding.production_training_allowed,
            "blockers": binding.blockers,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-production-target-binding: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
