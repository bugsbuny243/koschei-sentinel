from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_run_reuse import evaluate_completed_run_reuse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed reuse check for a completed Koschei Sentinel Cyber SFT run"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_completed_run_reuse(
            run_dir=args.run_dir,
            source_binding_path=args.training_source,
            config_path=args.config,
            plan_path=args.plan,
            repository_commit=args.repository_commit,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.reusable else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-reuse-check: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
