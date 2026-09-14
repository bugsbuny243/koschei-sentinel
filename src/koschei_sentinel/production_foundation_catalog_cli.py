from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_foundation_catalog import build_foundation_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic multi-shard foundation catalog")
    parser.add_argument("--manifest", action="append", required=True, dest="manifests")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = build_foundation_catalog(args.manifests, output_path=args.output)
        print(json.dumps(catalog.model_dump(mode="json"), sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-foundation-catalog: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
