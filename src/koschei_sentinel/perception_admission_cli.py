from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.perception_adapter_sdk import AdapterBatchResult
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceRegistry,
    admit_perception_adapter_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admit a Sentinel perception adapter result against an enrolled source registry"
    )
    parser.add_argument("--adapter-result", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = AdapterBatchResult.model_validate_json(
            Path(args.adapter_result).read_text(encoding="utf-8")
        )
        registry = PerceptionSourceRegistry.model_validate_json(
            Path(args.registry).read_text(encoding="utf-8")
        )
        admitted = admit_perception_adapter_result(result, registry)

        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        envelope_path = output / "admitted-perception-batch.json"
        batch_path = output / "perception-batch.json"
        receipt_path = output / "admission-receipt.json"

        envelope_path.write_text(
            json.dumps(admitted.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        batch_path.write_text(
            json.dumps(
                admitted.perception_batch.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        receipt_path.write_text(
            json.dumps(admitted.admission.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "admission_id": admitted.admission.admission_id,
                    "adapter_id": admitted.admission.adapter_id,
                    "source_principals": admitted.admission.source_principals,
                    "independence_domains": admitted.admission.independence_domains,
                    "perception_batch_sha256": admitted.admission.perception_batch_sha256,
                    "admitted_perception_batch": str(envelope_path),
                    "admission_receipt": str(receipt_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-perception-admit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
