from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    GoldHoldoutInferenceRunReceipt,
    build_gold_holdout_inference_plan,
    execute_gold_holdout_inference,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from koschei_sentinel.gold_holdout_pack_admission import (
    admit_signed_gold_holdout_pack,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute answer-key-isolated Gold HOLDOUT inference with a verified "
            "Cyber SFT candidate export"
        )
    )
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--inference-pack-signature", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--candidate-export", required=True)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--generation-policy", required=True)
    parser.add_argument("--plan-output")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load the pinned base model + verified adapter and generate HOLDOUT predictions",
    )
    return parser


def _load_policy(path: str) -> GoldHoldoutGenerationPolicy:
    try:
        return GoldHoldoutGenerationPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Gold HOLDOUT generation policy: {path}") from exc


def _verify_signed_pack(
    *,
    inference_pack: str,
    signature_path: str,
    reviewer_public_key_path: str,
) -> None:
    admit_signed_gold_holdout_pack(
        inference_pack=inference_pack,
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
    )


def _snapshot_signed_pack(
    *,
    inference_pack: str,
    signature_path: str,
    reviewer_public_key_path: str,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"Gold HOLDOUT pack snapshot already exists: {destination}")
    shutil.copytree(inference_pack, destination, symlinks=True)
    _verify_signed_pack(
        inference_pack=str(destination),
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
    )
    return destination


def _assert_raw_candidate_export(candidate_export: str) -> None:
    report = verify_cyber_sft_export(candidate_export)
    if not report.valid:
        detail = "; ".join(report.violations[:5])
        raise ValueError(
            "Gold HOLDOUT candidate export failed raw-path verification"
            + (f": {detail}" if detail else "")
        )


def _execute_atomic(
    *,
    inference_pack: str,
    candidate_export: str,
    model_ref: str,
    output_dir: str,
    generation_policy: GoldHoldoutGenerationPolicy,
) -> GoldHoldoutInferenceRunReceipt:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Gold HOLDOUT inference output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    staging = staging_root / "output"
    try:
        receipt = execute_gold_holdout_inference(
            inference_pack_dir=inference_pack,
            candidate_export_dir=candidate_export,
            model_ref=model_ref,
            output_dir=staging,
            generation_policy=generation_policy,
        )
        verification = verify_gold_holdout_inference_output(
            staging,
            inference_pack,
            candidate_export,
        )
        if not verification.valid:
            detail = "; ".join(verification.violations[:5])
            raise ValueError(
                "fresh Gold HOLDOUT inference output failed offline verification"
                + (f": {detail}" if detail else "")
            )
        os.replace(staging, destination)
        return receipt
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_policy = _load_policy(args.generation_policy)
        _verify_signed_pack(
            inference_pack=args.inference_pack,
            signature_path=args.inference_pack_signature,
            reviewer_public_key_path=args.reviewer_public_key,
        )
        _assert_raw_candidate_export(args.candidate_export)

        with tempfile.TemporaryDirectory(prefix="gold-holdout-pack-snapshot-") as temp_dir:
            snapshot = _snapshot_signed_pack(
                inference_pack=args.inference_pack,
                signature_path=args.inference_pack_signature,
                reviewer_public_key_path=args.reviewer_public_key,
                destination=Path(temp_dir) / "pack",
            )
            plan = build_gold_holdout_inference_plan(
                inference_pack_dir=snapshot,
                candidate_export_dir=args.candidate_export,
                model_ref=args.model_ref,
                generation_policy=selected_policy,
            )
            if args.plan_output:
                destination = Path(args.plan_output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
            if not args.execute:
                print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
                return 0
            if args.output_dir is None:
                raise ValueError("--output-dir is required with --execute")
            receipt = _execute_atomic(
                inference_pack=str(snapshot),
                candidate_export=args.candidate_export,
                model_ref=args.model_ref,
                output_dir=args.output_dir,
                generation_policy=selected_policy,
            )
        print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-infer: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
