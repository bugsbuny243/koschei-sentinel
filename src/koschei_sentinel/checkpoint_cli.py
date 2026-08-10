from __future__ import annotations

import argparse
import json

from koschei_sentinel.checkpoint_lineage import (
    finalize_checkpoint,
    load_execution_envelope,
    prepare_execution_envelope,
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
    write_execution_envelope,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and verify Koschei Sentinel Stage 2 checkpoint lineage; "
            "this command never starts model training"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Bind an immutable continued-pretraining plan into an execution envelope",
    )
    prepare.add_argument("--plan", required=True, help="Path to Stage 2 plan JSON")
    prepare.add_argument(
        "--output", required=True, help="No-replace path for execution-envelope JSON"
    )
    prepare.add_argument(
        "--root", default=".", help="Repository root used to resolve plan/output lineage"
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="Hash completed checkpoint files and write checkpoint-manifest.json",
    )
    finalize.add_argument(
        "--envelope", required=True, help="Path to execution-envelope JSON"
    )
    finalize.add_argument(
        "--root", default=".", help="Repository root used by the execution envelope"
    )

    verify = subparsers.add_parser(
        "verify",
        help="Re-hash checkpoint files and verify an existing checkpoint manifest",
    )
    verify.add_argument(
        "--envelope", required=True, help="Path to execution-envelope JSON"
    )
    verify.add_argument(
        "--manifest", required=True, help="Path to checkpoint-manifest.json"
    )
    verify.add_argument(
        "--root", default=".", help="Repository root used by the execution envelope"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_execution_envelope(args.plan, root=args.root)
            write_execution_envelope(result, args.output)
        elif args.command == "finalize":
            envelope = load_execution_envelope(args.envelope)
            result = finalize_checkpoint(envelope, root=args.root)
            write_checkpoint_manifest(result, envelope, root=args.root)
        else:
            envelope = load_execution_envelope(args.envelope)
            result = verify_checkpoint_manifest(
                args.manifest,
                envelope,
                root=args.root,
            )
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-stage2-checkpoint: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
