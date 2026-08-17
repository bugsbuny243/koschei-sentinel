from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_security_corpus import (
    audit_blockchain_security_corpus,
    load_blockchain_security_documents,
    load_blockchain_security_policy,
    write_blockchain_security_audit,
)
from koschei_sentinel.pretraining_corpus import load_pretraining_holdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a rights-declared, privacy-safe, multi-chain blockchain-security "
            "continued-pretraining corpus against diversity and held-out gates"
        )
    )
    parser.add_argument("--corpus", required=True, help="Canonical JSONL corpus")
    parser.add_argument("--holdout", required=True, help="Held-out benchmark/family JSON")
    parser.add_argument("--policy", required=True, help="Blockchain corpus policy JSON")
    parser.add_argument("--output", help="Optional no-replace audit JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        documents = load_blockchain_security_documents(args.corpus)
        holdout = load_pretraining_holdout(args.holdout)
        policy = load_blockchain_security_policy(args.policy)
        audit = audit_blockchain_security_corpus(documents, holdout, policy)
        if args.output:
            write_blockchain_security_audit(audit, args.output)
        print(json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if audit.ready else 2
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"sentinel-blockchain-corpus-audit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
