from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_run_attestation import build_cyber_sft_run_attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed attestation for a verified Koschei Sentinel Cyber SFT run"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-preflight", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--profile", choices=("micro", "normal", "lowmem"), required=True)
    parser.add_argument("--repository-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        attestation = build_cyber_sft_run_attestation(
            config_path=args.config,
            plan_path=args.plan,
            run_dir=args.run_dir,
            model_preflight_path=args.model_preflight,
            verification_path=args.verification,
            selected_profile=args.profile,
            repository_commit=args.repository_commit,
        )
        print(json.dumps(attestation.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-attest: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
