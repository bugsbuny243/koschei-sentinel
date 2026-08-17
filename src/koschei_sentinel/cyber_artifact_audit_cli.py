from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_corpus_catalog import (
    audit_artifacts,
    load_artifacts,
    load_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Cyber Corpus v3 artifact manifests against approved source, "
            "revision, license-inheritance, duplicate, and eval-exclusion gates"
        )
    )
    parser.add_argument("--sources", required=True, help="Approved source catalog JSONL")
    parser.add_argument("--artifacts", required=True, help="Collected artifact manifest JSONL")
    parser.add_argument("--output", help="Optional immutable audit JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_artifacts(
            load_artifacts(args.artifacts),
            load_catalog(args.sources),
        )
        payload = json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            destination = Path(args.output)
            if destination.exists():
                raise FileExistsError(f"artifact audit already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if audit.ready_for_ingestion else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-artifact-audit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
