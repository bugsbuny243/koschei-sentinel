from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_sft_training_source import (
    CyberSFTTrainingSourceBinding,
    build_training_source_binding,
    verify_training_source_binding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify a pre-training Koschei Sentinel Cyber SFT source binding"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = Path(args.output)
    try:
        expected = build_training_source_binding(
            config_path=args.config,
            plan_path=args.plan,
            repository_commit=args.repository_commit,
        )
        if destination.exists():
            existing = CyberSFTTrainingSourceBinding.model_validate_json(
                destination.read_bytes()
            )
            verify_training_source_binding(
                existing,
                config_path=args.config,
                plan_path=args.plan,
                repository_commit=args.repository_commit,
            )
            binding = existing
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(expected.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            binding = expected
        print(json.dumps(binding.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-source-bind: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
