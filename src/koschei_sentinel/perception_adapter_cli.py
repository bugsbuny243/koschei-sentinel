from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.perception_adapter_profiles import (
    generic_cicd_adapter,
    generic_cloud_iam_adapter,
    generic_endpoint_process_adapter,
    generic_signer_wallet_adapter,
)
from koschei_sentinel.perception_adapter_sdk import (
    DeclarativeAdapterSpec,
    DeclarativePerceptionAdapter,
    SanitizedTelemetryEvent,
    run_perception_adapter_batch,
)

_PROFILE_FACTORIES = {
    "generic.endpoint-process": generic_endpoint_process_adapter,
    "generic.cloud-iam": generic_cloud_iam_adapter,
    "generic.cicd": generic_cicd_adapter,
    "generic.signer-wallet": generic_signer_wallet_adapter,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize sanitized telemetry through a fail-closed Sentinel perception adapter"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", choices=sorted(_PROFILE_FACTORIES))
    selection.add_argument("--spec", help="Declarative perception adapter spec JSON")
    parser.add_argument(
        "--event",
        action="append",
        required=True,
        help="Sanitized telemetry event JSON; repeat for a batch",
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_adapter(args: argparse.Namespace) -> DeclarativePerceptionAdapter:
    if args.profile:
        return _PROFILE_FACTORIES[args.profile]()
    spec = DeclarativeAdapterSpec.model_validate_json(
        Path(args.spec).read_text(encoding="utf-8")
    )
    return DeclarativePerceptionAdapter(spec)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        adapter = _load_adapter(args)
        events = [
            SanitizedTelemetryEvent.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in args.event
        ]
        result = run_perception_adapter_batch(adapter, events, batch_id=args.batch_id)

        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        batch_path = output / "perception-batch.json"
        result_path = output / "adapter-result.json"
        batch_path.write_text(
            json.dumps(
                result.perception_batch.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "adapter_id": adapter.descriptor.adapter_id,
                    "batch_id": result.perception_batch.batch_id,
                    "observations": len(result.perception_batch.observations),
                    "batch_sha256": result.batch_sha256,
                    "perception_batch": str(batch_path),
                    "adapter_result": str(result_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-perception-adapt: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
