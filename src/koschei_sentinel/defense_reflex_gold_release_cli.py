from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_gold_queue import GoldDefenseReviewPacket
from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from koschei_sentinel.defense_reflex_gold_review import GoldReviewedPacket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a split-safe human Gold Defense Reflex release with training and holdout isolation"
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="Reviewed Cyber Range scenario JSON; repeat in packet/review order",
    )
    parser.add_argument(
        "--packet",
        action="append",
        required=True,
        help="Pre-review Gold packet JSON; repeat in scenario order",
    )
    parser.add_argument(
        "--reviewed",
        action="append",
        required=True,
        help="Human-reviewed Gold packet JSON; repeat in scenario order",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _load(path: str, model_type, label: str):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    try:
        return model_type.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        counts = {len(args.scenario), len(args.packet), len(args.reviewed)}
        if len(counts) != 1:
            raise ValueError(
                "--scenario, --packet and --reviewed must be repeated the same number of times"
            )
        rows = [
            (
                _load(scenario_path, CyberRangeScenario, "scenario"),
                _load(packet_path, GoldDefenseReviewPacket, "Gold packet"),
                _load(reviewed_path, GoldReviewedPacket, "Gold reviewed packet"),
            )
            for scenario_path, packet_path, reviewed_path in zip(
                args.scenario,
                args.packet,
                args.reviewed,
                strict=True,
            )
        ]
        manifest = write_gold_defense_release(rows, args.output_dir)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-release: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
