from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_snapshot_materializer import (
    SnapshotMaterializationSpec,
    materialize_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize approved pinned Cyber Corpus v3 snapshots and emit SHA-256 receipts"
    )
    parser.add_argument("--spec", required=True, help="Snapshot materialization spec JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = SnapshotMaterializationSpec.model_validate_json(
            Path(args.spec).read_text(encoding="utf-8")
        )
        receipts = materialize_spec(spec)
        print(
            json.dumps(
                {
                    "sources": len(receipts),
                    "artifacts": sum(row.artifacts for row in receipts),
                    "training_authorized_artifacts": sum(
                        row.training_authorized_artifacts for row in receipts
                    ),
                    "receipts": [row.model_dump(mode="json") for row in receipts],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-materialize: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
