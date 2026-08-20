from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_export import build_cyber_sft_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an atomic portable candidate export from a freshly verified Cyber SFT run"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--training-source", required=True)
    parser.add_argument("--model-preflight", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_cyber_sft_export(
            config_path=args.config,
            plan_path=args.plan,
            training_source_path=args.training_source,
            model_preflight_path=args.model_preflight,
            verification_path=args.verification,
            attestation_path=args.attestation,
            output_dir=args.output_dir,
            root=args.root,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-export: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
