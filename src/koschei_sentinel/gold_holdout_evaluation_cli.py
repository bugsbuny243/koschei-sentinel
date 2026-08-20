from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from koschei_sentinel.cyber_sft_candidate_snapshot import (
    snapshot_verified_cyber_sft_export,
)
from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutInferenceManifest,
    GoldHoldoutPrediction,
    evaluate_gold_holdout_predictions,
    export_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutInferenceRunReceipt,
    _load_inference_pack,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from koschei_sentinel.gold_holdout_pack_admission import (
    GoldHoldoutPackAdmission,
    admit_signed_gold_holdout_pack,
    snapshot_admitted_gold_holdout_pack,
)
from koschei_sentinel.gold_holdout_pack_preflight import (
    preflight_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_pack_signing import (
    sign_gold_holdout_inference_pack,
    verify_gold_holdout_inference_pack_signature,
)
from koschei_sentinel.gold_holdout_zero_prediction import (
    build_zero_prediction_gold_report,
)
from koschei_sentinel.gold_review_signing import (
    audit_gold_release_review_signatures,
    load_reviewer_private_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export answer-key-isolated Gold HOLDOUT inputs or evaluate predictions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-inputs")
    export_parser.add_argument("--release-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--reviewer-private-key", required=True)
    export_parser.add_argument("--signature-output", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--release-dir", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--policy")
    evaluate_parser.add_argument("--output")

    output_parser = subparsers.add_parser("evaluate-output")
    output_parser.add_argument("--release-dir", required=True)
    output_parser.add_argument("--inference-pack", required=True)
    output_parser.add_argument("--inference-pack-signature", required=True)
    output_parser.add_argument("--reviewer-public-key", required=True)
    output_parser.add_argument("--inference-output", required=True)
    output_parser.add_argument("--candidate-export", required=True)
    output_parser.add_argument("--policy", required=True)
    output_parser.add_argument("--output")
    return parser


def _load_predictions(
    path: str,
    *,
    allow_empty: bool = False,
) -> list[GoldHoldoutPrediction]:
    rows: list[GoldHoldoutPrediction] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read Gold HOLDOUT predictions: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(GoldHoldoutPrediction.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid Gold HOLDOUT prediction at line {line_number}"
            ) from exc
    if not rows and not allow_empty:
        raise ValueError("Gold HOLDOUT predictions file is empty")
    return rows


def _load_policy(path: str | None) -> GoldHoldoutEvaluationPolicy | None:
    if path is None:
        return None
    try:
        return GoldHoldoutEvaluationPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Gold HOLDOUT evaluation policy: {path}") from exc


def _write_report(report, output: str | None) -> None:
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(payload, encoding="utf-8")
    print(payload, end="")


def _assert_raw_candidate_export(candidate_export: str) -> None:
    report = verify_cyber_sft_export(candidate_export)
    if not report.valid:
        detail = "; ".join(report.violations[:5])
        raise ValueError(
            "Gold HOLDOUT candidate export failed raw-path verification"
            + (f": {detail}" if detail else "")
        )


def _verify_signed_pack(
    *,
    inference_pack: str,
    signature_path: str,
    reviewer_public_key_path: str,
) -> GoldHoldoutPackAdmission:
    return admit_signed_gold_holdout_pack(
        inference_pack=inference_pack,
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
    )


def _export_inputs_atomic(release_dir: str, output_dir: str) -> GoldHoldoutInferenceManifest:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Gold HOLDOUT inference pack output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    staging = staging_root / "pack"
    try:
        manifest = export_gold_holdout_inference_pack(release_dir, staging)
        rows, verified_manifest, _manifest_raw = _load_inference_pack(staging)
        if verified_manifest != manifest or len(rows) != manifest.case_count:
            raise ValueError(
                "fresh Gold HOLDOUT inference pack differs from sealed loader verification"
            )
        preflight_gold_holdout_inference_pack(staging)
        os.replace(staging, destination)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _export_signed_inputs(
    *,
    release_dir: str,
    output_dir: str,
    reviewer_private_key_path: str,
    signature_output: str,
):
    destination = Path(output_dir)
    signature_destination = Path(signature_output)
    if destination.exists():
        raise FileExistsError(
            f"Gold HOLDOUT inference pack output already exists: {destination}"
        )
    if signature_destination.exists():
        raise FileExistsError(
            f"Gold HOLDOUT pack signature output already exists: {signature_destination}"
        )

    reviewer_private_key = load_reviewer_private_key(reviewer_private_key_path)
    signature_audit = audit_gold_release_review_signatures(
        release_dir,
        reviewer_private_key.public_key(),
    )
    if not signature_audit.valid:
        detail = "; ".join(signature_audit.violations[:5])
        raise ValueError(
            "Gold HOLDOUT pack export requires a valid signed Gold release"
            + (f": {detail}" if detail else "")
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.signed-staging-",
            dir=destination.parent,
        )
    )
    staged_pack = transaction_root / "pack"
    signature_temp: Path | None = None
    signature_published = False
    try:
        manifest = _export_inputs_atomic(release_dir, str(staged_pack))
        proof = sign_gold_holdout_inference_pack(
            staged_pack / "manifest.json",
            reviewer_private_key,
            review_signature_audit_sha256=signature_audit.audit_sha256,
        )
        verify_gold_holdout_inference_pack_signature(
            proof,
            staged_pack / "manifest.json",
            reviewer_private_key.public_key(),
        )

        signature_destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{signature_destination.name}.staging-",
            dir=signature_destination.parent,
            delete=False,
        )
        signature_temp = Path(handle.name)
        with handle:
            handle.write(
                json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n"
            )
        os.replace(signature_temp, signature_destination)
        signature_temp = None
        signature_published = True

        # Final pack publication is intentionally last: a final pack path is never
        # visible unless its detached trusted signature has already been published.
        os.replace(staged_pack, destination)
        return manifest, proof
    except (OSError, TypeError, ValueError):
        if signature_published:
            signature_destination.unlink(missing_ok=True)
        raise
    finally:
        if signature_temp is not None:
            signature_temp.unlink(missing_ok=True)
        shutil.rmtree(transaction_root, ignore_errors=True)


def _evaluate_verified_output(args):
    admission = _verify_signed_pack(
        inference_pack=args.inference_pack,
        signature_path=args.inference_pack_signature,
        reviewer_public_key_path=args.reviewer_public_key,
    )
    _assert_raw_candidate_export(args.candidate_export)
    with tempfile.TemporaryDirectory(prefix="gold-holdout-eval-snapshot-") as temp_dir:
        snapshot_root = Path(temp_dir)
        inference_snapshot = snapshot_admitted_gold_holdout_pack(
            admission,
            args.inference_pack,
            snapshot_root / "pack",
        )
        candidate_snapshot = snapshot_verified_cyber_sft_export(
            args.candidate_export,
            snapshot_root / "candidate-export",
        )
        verification = verify_gold_holdout_inference_output(
            args.inference_output,
            inference_snapshot,
            candidate_snapshot,
        )
        if not verification.valid:
            raise ValueError("Gold HOLDOUT inference output verification failed")
        inference_manifest = GoldHoldoutInferenceManifest.model_validate_json(
            (inference_snapshot / "manifest.json").read_bytes()
        )

    release_audit = audit_gold_defense_release(args.release_dir)
    if not release_audit.valid:
        raise ValueError("Gold HOLDOUT release audit is invalid")
    if inference_manifest.source_gold_audit_sha256 != release_audit.audit_sha256:
        raise ValueError("inference pack was exported from a different Gold release audit")

    receipt = GoldHoldoutInferenceRunReceipt.model_validate_json(
        (Path(args.inference_output) / "receipt.json").read_bytes()
    )
    selected_policy = _load_policy(args.policy)
    predictions = _load_predictions(
        str(Path(args.inference_output) / "predictions.jsonl"),
        allow_empty=True,
    )
    if predictions:
        report = evaluate_gold_holdout_predictions(
            args.release_dir,
            predictions,
            policy=selected_policy,
        )
    else:
        report = build_zero_prediction_gold_report(
            args.release_dir,
            model_ref=receipt.model_ref,
            model_revision=receipt.model_revision,
            adapter_digest=receipt.adapter_digest,
            policy=selected_policy,
        )

    identity = (report.model_ref, report.model_revision, report.adapter_digest)
    expected_identity = (receipt.model_ref, receipt.model_revision, receipt.adapter_digest)
    if identity != expected_identity:
        raise ValueError("Gold HOLDOUT evaluation identity differs from inference receipt")
    if report.case_count != verification.case_count:
        raise ValueError("Gold HOLDOUT evaluation case count differs from inference verification")
    if report.prediction_count != verification.prediction_count:
        raise ValueError("Gold HOLDOUT evaluation prediction count differs from inference verification")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export-inputs":
            result, proof = _export_signed_inputs(
                release_dir=args.release_dir,
                output_dir=args.output_dir,
                reviewer_private_key_path=args.reviewer_private_key,
                signature_output=args.signature_output,
            )
            print(
                json.dumps(
                    {
                        "manifest": result.model_dump(mode="json"),
                        "pack_signature": proof.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "evaluate-output":
            report = _evaluate_verified_output(args)
            _write_report(report, args.output)
            return 0 if report.passed else 1

        predictions = _load_predictions(args.predictions)
        report = evaluate_gold_holdout_predictions(
            args.release_dir,
            predictions,
            policy=_load_policy(args.policy),
        )
        _write_report(report, args.output)
        return 0 if report.passed else 1
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-eval: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
