from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_source_catalog import (
    audit_source_catalog,
    load_source_catalog,
    load_source_catalog_policy,
    write_catalog_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify many immutable blockchain-security source snapshots as one diverse, "
            "lineage-bound training corpus"
        )
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--corpus-output")
    parser.add_argument("--manifest-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.corpus_output) != bool(args.manifest_output):
        print(
            "sentinel-blockchain-compose: corpus-output and manifest-output "
            "must be supplied together"
        )
        return 2
    try:
        result = audit_source_catalog(
            load_source_catalog(args.catalog),
            load_source_catalog_policy(args.policy),
            root=args.root,
        )
        if args.corpus_output:
            write_catalog_outputs(
                result,
                corpus_path=args.corpus_output,
                manifest_path=args.manifest_output,
            )
        manifest = result.manifest
        print(
            json.dumps(
                {
                    "ready": manifest.ready,
                    "catalog_id": manifest.catalog_id,
                    "sources": manifest.sources,
                    "documents": manifest.documents,
                    "high_trust_share_bps": manifest.high_trust_share_bps,
                    "synthetic_source_share_bps": manifest.synthetic_source_share_bps,
                    "max_single_source_document_share_bps": (
                        manifest.max_single_source_document_share_bps
                    ),
                    "combined_corpus_digest": manifest.combined_corpus_digest,
                    "violations": manifest.violations,
                    "corpus_output": args.corpus_output,
                    "manifest_output": args.manifest_output,
                    "network_access": False,
                    "training_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if manifest.ready else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-compose: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
