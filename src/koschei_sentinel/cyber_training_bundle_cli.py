from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_collection_batch import CollectionBatchSeal
from koschei_sentinel.cyber_training_bundle import build_cyber_training_bundle
from koschei_sentinel.defense_reflex_corpus import DefenseReflexCorpusManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable Sentinel cyber training bundle from sealed training planes"
    )
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--foundation-model", required=True)
    parser.add_argument("--foundation-revision", required=True)
    parser.add_argument("--knowledge-seal", required=True, help="Cyber Corpus seal.json")
    parser.add_argument("--defense-reflex-manifest", required=True, help="Defense Reflex manifest.json")
    parser.add_argument("--eval-holdout-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        knowledge = CollectionBatchSeal.model_validate_json(
            Path(args.knowledge_seal).read_text(encoding="utf-8")
        )
        reflex = DefenseReflexCorpusManifest.model_validate_json(
            Path(args.defense_reflex_manifest).read_text(encoding="utf-8")
        )
        bundle = build_cyber_training_bundle(
            bundle_id=args.bundle_id,
            foundation_model_ref=args.foundation_model,
            foundation_model_revision=args.foundation_revision,
            knowledge_seal=knowledge,
            reflex_manifest=reflex,
            eval_holdout_sha256=args.eval_holdout_sha256,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-training-bundle: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
