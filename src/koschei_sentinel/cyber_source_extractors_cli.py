from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_source_extractors import (
    extract_nvd_snapshot,
    extract_osv_snapshot,
    extract_rustsec_snapshot,
    write_extracted_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pinned cyber source snapshots into artifact manifests with fail-closed license resolution"
    )
    parser.add_argument("--provider", required=True, choices=["rustsec", "osv", "nvd"])
    parser.add_argument("--input", required=True, help="Pinned local snapshot path")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--snapshot-sha256")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kwargs = {
            "source_id": args.source_id,
            "source_revision": args.revision,
            "snapshot_digest": args.snapshot_sha256,
        }
        if args.provider == "rustsec":
            rows = extract_rustsec_snapshot(args.input, **kwargs)
        elif args.provider == "osv":
            rows = extract_osv_snapshot(args.input, **kwargs)
        else:
            rows = extract_nvd_snapshot(args.input, **kwargs)
        write_extracted_release(rows, manifest_path=args.manifest, corpus_path=args.corpus)
        trainable = sum(row.artifact.training_authorization for row in rows)
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "artifacts": len(rows),
                    "training_authorized_artifacts": trainable,
                    "blocked_or_review_required_artifacts": len(rows) - trainable,
                    "manifest": str(Path(args.manifest)),
                    "corpus": str(Path(args.corpus)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-extract: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
