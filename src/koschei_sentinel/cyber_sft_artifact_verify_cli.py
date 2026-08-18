from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_artifact_verify import verify_cyber_sft_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a completed Koschei Sentinel Cyber SFT run and its training receipt"
    )
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_cyber_sft_run(args.run_dir)
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-verify: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
