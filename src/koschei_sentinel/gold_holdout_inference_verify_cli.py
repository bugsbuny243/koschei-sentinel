from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from koschei_sentinel.gold_holdout_pack_preflight import (
    preflight_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_pack_signing import (
    load_gold_holdout_pack_signature,
    verify_gold_holdout_inference_pack_signature,
)
from koschei_sentinel.gold_review_signing import load_reviewer_public_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-verify Gold HOLDOUT inference output against its input pack and "
            "independently verified Cyber SFT candidate export"
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--inference-pack-signature", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--candidate-export", required=True)
    return parser


def _verify_signed_pack(
    *,
    inference_pack: str,
    signature_path: str,
    reviewer_public_key_path: str,
) -> None:
    preflight_gold_holdout_inference_pack(inference_pack)
    proof = load_gold_holdout_pack_signature(signature_path)
    reviewer_public_key = load_reviewer_public_key(reviewer_public_key_path)
    verify_gold_holdout_inference_pack_signature(
        proof,
        Path(inference_pack) / "manifest.json",
        reviewer_public_key,
    )


def _assert_raw_candidate_export(candidate_export: str) -> None:
    report = verify_cyber_sft_export(candidate_export)
    if not report.valid:
        detail = "; ".join(report.violations[:5])
        raise ValueError(
            "Gold HOLDOUT candidate export failed raw-path verification"
            + (f": {detail}" if detail else "")
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _verify_signed_pack(
            inference_pack=args.inference_pack,
            signature_path=args.inference_pack_signature,
            reviewer_public_key_path=args.reviewer_public_key,
        )
        _assert_raw_candidate_export(args.candidate_export)
        report = verify_gold_holdout_inference_output(
            args.output_dir,
            args.inference_pack,
            args.candidate_export,
        )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-infer-verify: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
