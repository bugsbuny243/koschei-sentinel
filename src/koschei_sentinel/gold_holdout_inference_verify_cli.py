from __future__ import annotations

import argparse
import json

from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-verify Gold HOLDOUT inference output against its input pack and "
            "independently verified Cyber SFT candidate export"
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--candidate-export", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_gold_holdout_inference_output(
            args.output_dir,
            args.inference_pack,
            args.candidate_export,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-infer-verify: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
