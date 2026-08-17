from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.causal_defense_corpus import write_causal_defense_release
from koschei_sentinel.cyber_world_model_episode import CyberWorldModelEpisode
from koschei_sentinel.defense_reflex_review import ReviewedCorrectionTrajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed Causal Defense Corpus from reviewed episode/correction pairs"
    )
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="Cyber World Model episode JSON; repeat once per pair",
    )
    parser.add_argument(
        "--correction",
        action="append",
        required=True,
        help="Reviewed correction JSON; repeat in the same order as --episode",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if len(args.episode) != len(args.correction):
            raise ValueError("--episode and --correction counts must match exactly")
        pairs: list[tuple[CyberWorldModelEpisode, ReviewedCorrectionTrajectory]] = []
        for index, (episode_path, correction_path) in enumerate(
            zip(args.episode, args.correction, strict=True),
            1,
        ):
            try:
                episode = CyberWorldModelEpisode.model_validate_json(
                    Path(episode_path).read_text(encoding="utf-8")
                )
                correction = ReviewedCorrectionTrajectory.model_validate_json(
                    Path(correction_path).read_text(encoding="utf-8")
                )
            except ValueError as exc:
                raise ValueError(f"invalid causal-defense pair {index}: {exc}") from exc
            pairs.append((episode, correction))

        manifest = write_causal_defense_release(pairs, args.output_dir)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-causal-defense-build: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
