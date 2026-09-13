from __future__ import annotations

import argparse

from koschei_sentinel.production_model_source import load_production_model_source_intake
from koschei_sentinel.production_target_binding import (
    load_production_target_binding,
    require_verified_production_target_binding,
)
from koschei_sentinel.production_target_provenance_authoring import (
    build_signed_production_target_provenance,
    write_provenance,
)
from koschei_sentinel.promotion import load_owner_private_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Author and sign provenance for a verified 397B/35B production target"
    )
    parser.add_argument("--binding", required=True)
    parser.add_argument("--source-intake", required=True)
    parser.add_argument("--owner-private-key", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        binding = require_verified_production_target_binding(
            load_production_target_binding(args.binding)
        )
        intake = load_production_model_source_intake(args.source_intake)
        private_key = load_owner_private_key(args.owner_private_key)
        provenance = build_signed_production_target_provenance(
            binding=binding,
            intake=intake,
            owner_private_key=private_key,
        )
        write_provenance(args.output, provenance)
        print(args.output)
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-production-provenance-author: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
