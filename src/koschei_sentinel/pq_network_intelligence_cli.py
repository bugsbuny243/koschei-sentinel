from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    verify_pq_network_materialization,
    verify_pq_research_snapshot_receipt,
    write_pq_research_snapshot_receipt,
)

_DEFAULT_WATCH = Path("configs/corpus/pq-network-watch.v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-pq-network-intelligence",
        description=(
            "Capture byte-bound PQ research evidence and materialize non-authorizing "
            "Sentinel PQ intelligence records without network fetching."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--source-id", required=True)
    capture.add_argument("--snapshot", type=Path, required=True)
    capture.add_argument("--captured-at", required=True)
    capture.add_argument("--watch", type=Path, default=_DEFAULT_WATCH)
    capture.add_argument("--output", type=Path, required=True)

    verify_snapshot = subparsers.add_parser("verify-snapshot")
    verify_snapshot.add_argument("--receipt", type=Path, required=True)
    verify_snapshot.add_argument("--snapshot", type=Path, required=True)
    verify_snapshot.add_argument("--watch", type=Path, default=_DEFAULT_WATCH)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--claim", type=Path, required=True)
    materialize.add_argument("--snapshot-receipt", type=Path, required=True)
    materialize.add_argument("--snapshot", type=Path, required=True)
    materialize.add_argument("--watch", type=Path, default=_DEFAULT_WATCH)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--receipt-output", type=Path, required=True)

    verify_materialization = subparsers.add_parser("verify-materialization")
    verify_materialization.add_argument("--claim", type=Path, required=True)
    verify_materialization.add_argument("--snapshot-receipt", type=Path, required=True)
    verify_materialization.add_argument("--snapshot", type=Path, required=True)
    verify_materialization.add_argument("--watch", type=Path, default=_DEFAULT_WATCH)
    verify_materialization.add_argument("--record", type=Path, required=True)
    verify_materialization.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = build_pq_research_snapshot_receipt(
                source_id=args.source_id,
                snapshot_path=args.snapshot,
                captured_at=args.captured_at,
                watch_registry_path=args.watch,
            )
            write_pq_research_snapshot_receipt(result, args.output)
            payload = result.model_dump(mode="json")
        elif args.command == "verify-snapshot":
            result = verify_pq_research_snapshot_receipt(
                receipt_path=args.receipt,
                snapshot_path=args.snapshot,
                watch_registry_path=args.watch,
            )
            payload = result.model_dump(mode="json")
        elif args.command == "materialize":
            record, receipt = materialize_pq_network_research_record(
                claim_path=args.claim,
                snapshot_receipt_path=args.snapshot_receipt,
                snapshot_path=args.snapshot,
                watch_registry_path=args.watch,
                output_path=args.output,
                materialization_receipt_path=args.receipt_output,
            )
            payload = {
                "record": record.model_dump(mode="json"),
                "materialization_receipt": receipt.model_dump(mode="json"),
            }
        else:
            result = verify_pq_network_materialization(
                claim_path=args.claim,
                snapshot_receipt_path=args.snapshot_receipt,
                snapshot_path=args.snapshot,
                watch_registry_path=args.watch,
                record_path=args.record,
                materialization_receipt_path=args.receipt,
            )
            payload = result.model_dump(mode="json")
    except (OSError, TypeError, ValueError) as exc:
        print(f"PQ network intelligence rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
