from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_perception import PerceptionBatch
from koschei_sentinel.perception_fusion import fuse_perception_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse multiple Sentinel perception batches with fail-closed provenance checks"
    )
    parser.add_argument(
        "--batch",
        action="append",
        required=True,
        help="Perception batch JSON; repeat for multiple sensors",
    )
    parser.add_argument("--fused-batch-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        batches = [
            PerceptionBatch.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in args.batch
        ]
        result = fuse_perception_batches(batches, fused_batch_id=args.fused_batch_id)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        batch_path = output / "perception-batch.json"
        receipt_path = output / "fusion-receipt.json"
        batch_path.write_text(
            json.dumps(
                result.perception_batch.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        receipt_path.write_text(
            json.dumps(result.receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "fused_batch_id": result.perception_batch.batch_id,
                    "observations": result.receipt.observation_count,
                    "unique_evidence": result.receipt.unique_evidence_count,
                    "source_principals": result.receipt.source_principals,
                    "fused_batch_sha256": result.receipt.fused_batch_sha256,
                    "perception_batch": str(batch_path),
                    "fusion_receipt": str(receipt_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-perception-fuse: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
