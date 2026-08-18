from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_model_access_preflight import audit_model_access
from koschei_sentinel.cyber_sft_training import load_cyber_sft_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify public access, exact revision and Qwen3.5 text-only CausalLM mapping "
            "before a Cyber SFT GPU run"
        )
    )
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_cyber_sft_config(args.config)
        report = audit_model_access(config)
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.ready else 1
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-model-preflight: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
