from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_scale_gate import evaluate_9b_scale_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a verified Qwen3.5 0.8B micro run may scale to 9B"
    )
    parser.add_argument("--micro-export", required=True)
    parser.add_argument("--target-config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_9b_scale_gate(
            micro_export_dir=args.micro_export,
            target_config_path=args.target_config,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.allowed_to_attempt_9b else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-scale-gate: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
