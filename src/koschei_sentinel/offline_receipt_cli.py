from __future__ import annotations

import argparse
import json

from koschei_sentinel.offline_receipt import (
    OfflineReceiptBlocked,
    build_offline_training_receipt,
    load_adapter_manifest,
    load_offline_job,
    write_offline_training_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify adapter files against a sealed offline training job and "
            "write a non-production completion receipt"
        )
    )
    parser.add_argument("--job", required=True)
    parser.add_argument("--adapter-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_offline_training_receipt(
            load_offline_job(args.job),
            load_adapter_manifest(args.adapter_manifest),
        )
        write_offline_training_receipt(receipt, args.output)
        print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except OfflineReceiptBlocked as exc:
        print(f"sentinel-job-receipt: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-job-receipt: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
