from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeReport, CyberRangeScenario
from koschei_sentinel.cyber_world_model_episode import build_world_model_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable temporal Cyber World Model episode from a range scenario/report"
    )
    parser.add_argument("--scenario", required=True, help="Cyber Range scenario JSON")
    parser.add_argument("--report", required=True, help="Cyber Range report JSON")
    parser.add_argument("--output", required=True, help="World Model episode JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenario = CyberRangeScenario.model_validate_json(
            Path(args.scenario).read_text(encoding="utf-8")
        )
        report = CyberRangeReport.model_validate_json(
            Path(args.report).read_text(encoding="utf-8")
        )
        episode = build_world_model_episode(scenario, report)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(episode.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-world-model-episode: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
