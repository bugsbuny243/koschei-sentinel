from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_security_release import (
    build_blockchain_security_release,
    verify_blockchain_security_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a source-isolated, family-isolated blockchain-security "
            "train/validation/test release"
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build")
    build.add_argument("--catalog", required=True)
    build.add_argument("--catalog-policy", required=True)
    build.add_argument("--source-catalog-manifest", required=True)
    build.add_argument("--corpus", required=True)
    build.add_argument("--holdout", required=True)
    build.add_argument("--blockchain-policy", required=True)
    build.add_argument("--corpus-audit", required=True)
    build.add_argument("--release-policy", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--root", default=".")

    verify = subcommands.add_parser("verify")
    verify.add_argument("release")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_blockchain_security_release(
                catalog_path=args.catalog,
                catalog_policy_path=args.catalog_policy,
                source_catalog_manifest_path=args.source_catalog_manifest,
                corpus_path=args.corpus,
                holdout_path=args.holdout,
                blockchain_policy_path=args.blockchain_policy,
                corpus_audit_path=args.corpus_audit,
                release_policy_path=args.release_policy,
                output_dir=args.output,
                root=args.root,
            )
            output = args.output
        else:
            manifest = verify_blockchain_security_release(args.release)
            output = args.release

        print(
            json.dumps(
                {
                    "ok": True,
                    "documents": manifest.documents,
                    "sources": manifest.sources,
                    "components": manifest.components,
                    "benchmark_suite_digest": manifest.benchmark_suite_digest,
                    "source_corpus_digest": manifest.source_corpus_digest,
                    "splits": {
                        name: {
                            "documents": split.documents,
                            "sources": split.sources,
                            "components": split.components,
                            "digest": split.digest,
                        }
                        for name, split in sorted(manifest.splits.items())
                    },
                    "output": output,
                    "source_leakage_detected": False,
                    "family_leakage_detected": False,
                    "training_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-release: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
