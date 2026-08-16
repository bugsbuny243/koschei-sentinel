from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
)
from koschei_sentinel.blockchain_security_ingest import (
    BlockchainSourceSnapshotSpec,
    ingest_snapshot,
    inspect_snapshot,
    load_snapshot_spec,
    snapshot_digest,
    write_ingest_result,
)
from koschei_sentinel.pretraining_corpus import RightsBasis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare immutable local source snapshots and ingest them into canonical "
            "Koschei Sentinel blockchain-security corpus rows"
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser(
        "prepare",
        help="Hash one local source snapshot and write a no-replace trusted spec",
    )
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument("--source-class", required=True, choices=[item.value for item in BlockchainSourceClass])
    prepare.add_argument("--rights-basis", required=True, choices=[item.value for item in RightsBasis])
    prepare.add_argument("--snapshot", required=True)
    prepare.add_argument("--chain", action="append", required=True, choices=[item.value for item in ChainFamily])
    prepare.add_argument("--threat", action="append", required=True, choices=[item.value for item in ThreatDomain])
    prepare.add_argument("--family-ref", action="append", default=[])
    prepare.add_argument("--max-file-bytes", type=int, default=100_000)
    prepare.add_argument("--root", default=".")
    prepare.add_argument("--output", required=True)

    build = subcommands.add_parser(
        "build",
        help="Verify one prepared snapshot spec and emit canonical corpus + manifest",
    )
    build.add_argument("--spec", required=True)
    build.add_argument("--root", default=".")
    build.add_argument("--corpus-output", required=True)
    build.add_argument("--manifest-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            provisional = BlockchainSourceSnapshotSpec(
                source_id=args.source_id,
                source_class=args.source_class,
                rights_basis=args.rights_basis,
                snapshot_path=args.snapshot,
                expected_snapshot_digest="0" * 64,
                chain_families=args.chain,
                threat_domains=args.threat,
                family_refs=args.family_ref,
                max_file_bytes=args.max_file_bytes,
            )
            _, files = inspect_snapshot(provisional, root=args.root)
            spec = provisional.model_copy(
                update={"expected_snapshot_digest": snapshot_digest(files)}
            )
            destination = Path(args.output)
            if destination.exists():
                raise FileExistsError(f"snapshot spec already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "source_id": spec.source_id,
                        "snapshot_digest": spec.expected_snapshot_digest,
                        "files": len(files),
                        "output": str(destination),
                        "network_access": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        spec = load_snapshot_spec(args.spec)
        result = ingest_snapshot(spec, root=args.root)
        write_ingest_result(
            result,
            corpus_path=args.corpus_output,
            manifest_path=args.manifest_output,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "source_id": result.manifest.source_id,
                    "snapshot_digest": result.manifest.snapshot_digest,
                    "documents": result.manifest.documents,
                    "corpus_file_digest": result.manifest.corpus_file_digest,
                    "corpus_output": args.corpus_output,
                    "manifest_output": args.manifest_output,
                    "network_access": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-ingest: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
