from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.web4_research_snapshot import (
    build_web4_research_snapshot_receipt,
    verify_web4_research_snapshot_receipt,
    write_web4_research_snapshot_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-research-snapshot",
        description=(
            "Create or verify a fail-closed Web4 research snapshot receipt from an "
            "operator-supplied local file without network fetching or training authorization."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--source-id", required=True)
    capture.add_argument("--snapshot", type=Path, required=True)
    capture.add_argument("--captured-at", required=True)
    capture.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/corpus/web4-v1.sources.proposed.jsonl"),
    )
    capture.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/corpus/web4-v1.sources.proposed.jsonl"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            receipt = build_web4_research_snapshot_receipt(
                source_id=args.source_id,
                snapshot_path=args.snapshot,
                captured_at=args.captured_at,
                source_registry_path=args.sources,
            )
            write_web4_research_snapshot_receipt(receipt, args.output)
        else:
            receipt = verify_web4_research_snapshot_receipt(
                receipt_path=args.receipt,
                snapshot_path=args.snapshot,
                source_registry_path=args.sources,
            )
    except (OSError, ValueError) as exc:
        print(f"Web4 research snapshot rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
