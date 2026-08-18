from __future__ import annotations

import argparse
import json

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline-verify a portable Koschei Sentinel Cyber SFT export bundle"
    )
    parser.add_argument("--export-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_cyber_sft_export(args.export_dir)
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft-export-verify: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
