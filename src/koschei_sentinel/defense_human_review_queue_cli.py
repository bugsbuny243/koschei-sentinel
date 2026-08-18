from __future__ import annotations

import argparse
import json

from koschei_sentinel.defense_human_review_queue import build_human_review_queue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a blinded human-review queue from Defense Reflex v3 examples"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_human_review_queue(
            examples_path=args.examples,
            output_dir=args.output_dir,
        )
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"defense-human-review-queue: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
