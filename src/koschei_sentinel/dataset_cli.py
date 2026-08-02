from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.dataset import export_jsonl, load_records, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-dataset-export",
        description="Export validated and pseudonymized ARVIS records as deterministic JSONL.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")

    records = load_records(args.input)
    manifest = export_jsonl(records, output_path=args.output, dry_run=args.dry_run)
    if args.manifest:
        write_manifest(manifest, args.manifest)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    if manifest.rejected_records:
        print("export rejected: no training data was written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
