from __future__ import annotations

import argparse
import json

from koschei_sentinel.pretraining_corpus import (
    audit_pretraining_corpus,
    load_pretraining_documents,
    load_pretraining_holdout,
    load_pretraining_policy,
    write_pretraining_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a privacy-safe, rights-declared Sentinel Stage 2 continued-"
            "pretraining corpus against held-out benchmark and incident-family gates"
        )
    )
    parser.add_argument("--corpus", required=True, help="Canonical JSONL corpus")
    parser.add_argument("--holdout", required=True, help="Held-out fingerprint set JSON")
    parser.add_argument("--policy", required=True, help="Stage 2 corpus policy JSON")
    parser.add_argument("--output", required=True, help="Write immutable audit JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_pretraining_corpus(
            load_pretraining_documents(args.corpus),
            load_pretraining_holdout(args.holdout),
            load_pretraining_policy(args.policy),
        )
        write_pretraining_audit(audit, args.output)
        print(
            json.dumps(
                {
                    "ready": audit.ready,
                    "policy_id": audit.policy_id,
                    "documents": audit.documents,
                    "unique_families": audit.unique_families,
                    "corpus_digest": audit.corpus_digest,
                    "holdout_digest": audit.holdout_digest,
                    "benchmark_suite_digest": audit.benchmark_suite_digest,
                    "violations": audit.violations,
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if audit.ready else 3
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-pretrain-audit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
