from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.blockchain_reviewed_catalog import (
    audit_reviewed_source_catalog,
    load_reviewed_source_catalog,
)
from koschei_sentinel.blockchain_source_catalog import (
    audit_source_catalog,
    load_source_catalog,
    load_source_catalog_policy,
    write_catalog_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify immutable or document-reviewed blockchain-security sources as one "
            "diverse, lineage-bound training corpus"
        )
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--corpus-output")
    parser.add_argument("--manifest-output")
    return parser


def _catalog_schema(path: str) -> str:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("invalid blockchain source catalog JSON") from exc
    schema = payload.get("schema_version")
    if not isinstance(schema, str):
        raise ValueError("blockchain source catalog schema_version is missing")
    return schema


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.corpus_output) != bool(args.manifest_output):
        print(
            "sentinel-blockchain-compose: corpus-output and manifest-output "
            "must be supplied together"
        )
        return 2
    try:
        policy = load_source_catalog_policy(args.policy)
        schema = _catalog_schema(args.catalog)
        if schema == "sentinel.reviewed-source-catalog.v1":
            result = audit_reviewed_source_catalog(
                load_reviewed_source_catalog(args.catalog),
                policy,
                root=args.root,
            )
            catalog_mode = "REVIEWED_DOCUMENT"
        elif schema == "sentinel.blockchain-source-catalog.v1":
            result = audit_source_catalog(
                load_source_catalog(args.catalog),
                policy,
                root=args.root,
            )
            catalog_mode = "IMMUTABLE_INGEST"
        else:
            raise ValueError(f"unsupported blockchain source catalog schema: {schema}")

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
                    "catalog_mode": catalog_mode,
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
