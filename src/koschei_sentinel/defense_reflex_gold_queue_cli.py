from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldReviewSplitPolicy,
    write_gold_review_queue,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pre-review, pre-split Gold Defense Reflex queue from Cyber Range scenarios"
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="Cyber Range scenario JSON; repeat for multiple scenarios",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-policy",
        help="Optional Gold review split policy JSON; defaults to 80/10/10",
    )
    return parser


def _load_scenario(path: str) -> CyberRangeScenario:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read scenario {path}: {exc}") from exc
    try:
        return CyberRangeScenario.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"invalid Cyber Range scenario: {path}") from exc


def _load_policy(path: str | None) -> GoldReviewSplitPolicy | None:
    if path is None:
        return None
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read split policy {path}: {exc}") from exc
    try:
        return GoldReviewSplitPolicy.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"invalid Gold review split policy: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = [_load_scenario(path) for path in args.scenario]
        policy = _load_policy(args.split_policy)
        manifest = write_gold_review_queue(
            scenarios,
            args.output_dir,
            policy=policy,
        )
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-queue: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
