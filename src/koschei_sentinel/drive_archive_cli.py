from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.drive_archive import (
    build_drive_archive_manifest,
    load_drive_archive_manifest,
    restore_and_verify_with_rclone,
    upload_and_verify_with_rclone,
    verify_local_archive,
    write_drive_archive_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, verify, push, and restore SHA256-bound Sentinel Drive archives"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build an immutable local archive manifest")
    plan.add_argument("--artifact-root", required=True)
    plan.add_argument("--artifact-id", required=True)
    plan.add_argument(
        "--kind",
        required=True,
        choices=[
            "CHECKPOINT",
            "DATASET_RELEASE",
            "MODEL_ADAPTER",
            "EVALUATION",
            "GENERIC_ARTIFACT",
        ],
    )
    plan.add_argument("--source-manifest")
    plan.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify", help="Verify local bytes against a manifest")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--artifact-root", required=True)

    push = subparsers.add_parser(
        "push", help="Upload an immutable archive to Google Drive and verify remote bytes"
    )
    push.add_argument("--manifest", required=True)
    push.add_argument("--artifact-root", required=True)
    push.add_argument("--receipt-output")

    pull = subparsers.add_parser(
        "pull", help="Restore a Drive archive into an empty directory and verify it"
    )
    pull.add_argument("--manifest", required=True)
    pull.add_argument("--destination", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            manifest = build_drive_archive_manifest(
                args.artifact_root,
                artifact_id=args.artifact_id,
                artifact_kind=args.kind,
                source_manifest=args.source_manifest,
            )
            write_drive_archive_manifest(manifest, args.output)
            print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0

        if args.command == "verify":
            manifest = load_drive_archive_manifest(args.manifest)
            result = verify_local_archive(manifest, args.artifact_root)
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0 if result.verified else 2

        if args.command == "push":
            receipt = upload_and_verify_with_rclone(args.manifest, args.artifact_root)
            payload = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            if args.receipt_output:
                Path(args.receipt_output).write_text(payload, encoding="utf-8")
            print(payload, end="")
            return 0

        if args.command == "pull":
            result = restore_and_verify_with_rclone(args.manifest, args.destination)
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0 if result.verified else 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-drive-archive: {exc}")
        return 2
    raise AssertionError("unreachable Drive archive command")


if __name__ == "__main__":
    raise SystemExit(main())
