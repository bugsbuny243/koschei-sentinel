from __future__ import annotations

import argparse
import json

from koschei_sentinel.cloud_runtime import (
    load_cloud_runtime_profile,
    preflight_cloud_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Sentinel's provider-neutral ephemeral cloud runtime contract"
    )
    parser.add_argument("--profile", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_cloud_runtime_profile(args.profile)
        result = preflight_cloud_runtime(profile)
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if result.launch_ready else 2
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cloud-runtime-preflight: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
