from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_megatron_candidate import (
    build_cyber_megatron_candidate,
    verify_cyber_megatron_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot or verify a content-addressed Qwen3.5-397B-A17B "
            "Megatron-SWIFT candidate"
        )
    )
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--config", required=True)
    snapshot.add_argument("--plan", required=True)
    snapshot.add_argument("--checkpoint", required=True)
    snapshot.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--config", required=True)
    verify.add_argument("--plan", required=True)
    verify.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            result = build_cyber_megatron_candidate(
                config_path=args.config,
                plan_path=args.plan,
                checkpoint_dir=args.checkpoint,
                output_path=args.output,
                root=args.root,
            )
            payload = result.model_dump(mode="json")
            code = 0
        else:
            result = verify_cyber_megatron_candidate(
                manifest_path=args.manifest,
                config_path=args.config,
                plan_path=args.plan,
                checkpoint_dir=args.checkpoint,
                root=args.root,
            )
            payload = result.model_dump(mode="json")
            code = 0 if result.valid else 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return code
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-megatron-candidate: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
