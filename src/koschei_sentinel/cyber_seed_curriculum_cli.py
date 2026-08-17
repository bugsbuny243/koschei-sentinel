from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_seed_curriculum import write_seed_curriculum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic 32-scenario synthetic policy-reviewed Cyber SFT seed curriculum"
        )
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Repository-local root such as build/cyber-training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = write_seed_curriculum(args.output_root)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-seed-curriculum: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
