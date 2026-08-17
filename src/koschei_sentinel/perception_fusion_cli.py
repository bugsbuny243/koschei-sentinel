from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_perception import PerceptionBatch
from koschei_sentinel.perception_assurance import fuse_admitted_perception_batches
from koschei_sentinel.perception_fusion import fuse_perception_batches
from koschei_sentinel.perception_source_registry import (
    AdmittedPerceptionBatch,
    PerceptionSourceRegistry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse Sentinel perception with admission and provenance verification"
    )
    parser.add_argument(
        "--admitted",
        action="append",
        help="Admitted perception batch JSON; repeat for multiple sensors",
    )
    parser.add_argument(
        "--batch",
        action="append",
        help="Raw PerceptionBatch JSON; development-only and requires --allow-unadmitted-dev",
    )
    parser.add_argument("--registry", help="Perception source registry for admitted fusion")
    parser.add_argument("--allow-unadmitted-dev", action="store_true")
    parser.add_argument("--fused-batch-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if bool(args.admitted) == bool(args.batch):
            raise ValueError("provide exactly one of --admitted or --batch")

        assurance_payload: dict[str, object] | None = None
        if args.admitted:
            if not args.registry:
                raise ValueError("admitted perception fusion requires --registry")
            registry = PerceptionSourceRegistry.model_validate_json(
                Path(args.registry).read_text(encoding="utf-8")
            )
            admitted = [
                AdmittedPerceptionBatch.model_validate_json(Path(path).read_text(encoding="utf-8"))
                for path in args.admitted
            ]
            assured = fuse_admitted_perception_batches(
                admitted,
                registry=registry,
                fused_batch_id=args.fused_batch_id,
            )
            perception_batch = assured.perception_batch
            receipt = assured.fusion_receipt
            assurance_payload = assured.assurance.model_dump(mode="json")
        else:
            if not args.allow_unadmitted_dev:
                raise ValueError(
                    "raw perception fusion is development-only; pass --allow-unadmitted-dev explicitly"
                )
            batches = [
                PerceptionBatch.model_validate_json(Path(path).read_text(encoding="utf-8"))
                for path in args.batch
            ]
            result = fuse_perception_batches(batches, fused_batch_id=args.fused_batch_id)
            perception_batch = result.perception_batch
            receipt = result.receipt

        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        batch_path = output / "perception-batch.json"
        receipt_path = output / "fusion-receipt.json"
        assurance_path = output / "assurance-summary.json"

        batch_path.write_text(
            json.dumps(perception_batch.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_path.write_text(
            json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if assurance_payload is not None:
            assurance_path.write_text(
                json.dumps(assurance_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        summary: dict[str, object] = {
            "fused_batch_id": perception_batch.batch_id,
            "observations": receipt.observation_count,
            "unique_evidence": receipt.unique_evidence_count,
            "source_principals": receipt.source_principals,
            "fused_batch_sha256": receipt.fused_batch_sha256,
            "production_admitted": assurance_payload is not None,
            "perception_batch": str(batch_path),
            "fusion_receipt": str(receipt_path),
        }
        if assurance_payload is not None:
            summary["independence_domains"] = assurance_payload["independence_domains"]
            summary["independent_domain_count"] = assurance_payload["independent_domain_count"]
            summary["assurance_summary"] = str(assurance_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-perception-fuse: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
