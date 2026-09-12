from __future__ import annotations

import argparse
import json

from koschei_sentinel.launch_readiness import audit_launch_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed market launch readiness audit for Koschei Sentinel"
    )
    parser.add_argument("--finalization", required=True)
    parser.add_argument(
        "--holdout-evidence",
        action="append",
        default=[],
        help="Independent holdout evidence JSON; repeat for multiple artifacts",
    )
    parser.add_argument("--production-authority")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_launch_readiness(
        finalization_path=args.finalization,
        holdout_evidence_paths=args.holdout_evidence,
        production_authority_path=args.production_authority,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.market_release_ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
