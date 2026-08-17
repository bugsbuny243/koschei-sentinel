from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_collection_batch import CollectionBatchSpec, seal_collection_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal an audited Cyber Corpus v3 collection batch")
    parser.add_argument("--spec", required=True, help="Collection batch spec JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for sealed batch")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = CollectionBatchSpec.model_validate(payload)
    seal = seal_collection_batch(spec, output_dir=args.output_dir)
    print(json.dumps(seal.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if seal.ready_for_training_pipeline else 2


if __name__ == "__main__":
    raise SystemExit(main())
