from __future__ import annotations

import argparse
import json

from koschei_sentinel.defense_reflex_gold_release_audit import (
    audit_gold_defense_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a human-reviewed Gold Defense Reflex release before training or evaluation"
        )
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output", help="Optional JSON audit report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_gold_defense_release(args.release_dir)
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report.valid else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-audit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
