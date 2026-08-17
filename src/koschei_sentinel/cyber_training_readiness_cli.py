from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_training import load_cyber_sft_config
from koschei_sentinel.cyber_training_readiness import audit_cyber_training_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether a Koschei Sentinel Cyber SFT run is ready to execute"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Also check installed training packages, CUDA visibility and configured memory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_cyber_sft_config(args.config)
        report = audit_cyber_training_readiness(
            config,
            check_runtime=args.check_runtime,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.ready_to_execute else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-training-readiness: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
