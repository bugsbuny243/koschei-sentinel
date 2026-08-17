from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_corpus_v2 import write_defense_reflex_v2_release
from koschei_sentinel.defense_reflex_review import ReviewedCorrectionTrajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build evidence-grounded Defense Reflex Corpus v2"
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="JSON list of {scenario, correction} file paths",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("pairs file must contain a non-empty JSON list")
        pairs = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"pairs[{index}] must be an object")
            scenario_path = item.get("scenario")
            correction_path = item.get("correction")
            if not isinstance(scenario_path, str) or not isinstance(correction_path, str):
                raise ValueError(f"pairs[{index}] requires scenario and correction paths")
            scenario = CyberRangeScenario.model_validate_json(
                Path(scenario_path).read_text(encoding="utf-8")
            )
            correction = ReviewedCorrectionTrajectory.model_validate_json(
                Path(correction_path).read_text(encoding="utf-8")
            )
            pairs.append((scenario, correction))
        manifest = write_defense_reflex_v2_release(pairs, args.output_dir)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-reflex-v2-build: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
