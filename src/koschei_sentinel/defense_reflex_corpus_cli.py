from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.defense_reflex_corpus import write_defense_reflex_release
from koschei_sentinel.defense_reflex_review import ReviewedCorrectionTrajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed Defense Reflex Corpus release from reviewed corrections"
    )
    parser.add_argument("--input", required=True, help="Reviewed correction JSONL")
    parser.add_argument("--output-dir", required=True, help="Release output directory")
    return parser


def _load(path: Path) -> list[ReviewedCorrectionTrajectory]:
    rows: list[ReviewedCorrectionTrajectory] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        try:
            rows.append(ReviewedCorrectionTrajectory.model_validate_json(text))
        except ValueError as exc:
            raise ValueError(f"invalid reviewed correction at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("reviewed correction input is empty")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        corrections = _load(Path(args.input))
        manifest = write_defense_reflex_release(corrections, args.output_dir)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-reflex-build: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
