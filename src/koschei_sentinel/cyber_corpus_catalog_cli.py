from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_corpus_catalog import audit_catalog, load_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the fail-closed Cyber Corpus v3 source catalog before collection"
    )
    parser.add_argument("--catalog", required=True, help="Source catalog JSONL")
    parser.add_argument("--output", help="Optional audit JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_catalog(load_catalog(args.catalog))
        payload = json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            destination = Path(args.output)
            if destination.exists():
                raise FileExistsError(f"catalog audit already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if audit.ready_for_collection else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-catalog-audit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
