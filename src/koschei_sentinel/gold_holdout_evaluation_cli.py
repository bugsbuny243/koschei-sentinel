from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutInferenceManifest,
    GoldHoldoutPrediction,
    evaluate_gold_holdout_predictions,
    export_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_inference_runner import GoldHoldoutInferenceRunReceipt
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from koschei_sentinel.gold_holdout_zero_prediction import (
    build_zero_prediction_gold_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export answer-key-isolated Gold HOLDOUT inputs or evaluate predictions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-inputs")
    export_parser.add_argument("--release-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--release-dir", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--policy")
    evaluate_parser.add_argument("--output")

    output_parser = subparsers.add_parser("evaluate-output")
    output_parser.add_argument("--release-dir", required=True)
    output_parser.add_argument("--inference-pack", required=True)
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


def _evaluate_verified_output(args):
    verification = verify_gold_holdout_inference_output(
        args.inference_output,
        args.inference_pack,
        args.candidate_export,
    )
    if not verification.valid:
        raise ValueError("Gold HOLDOUT inference output verification failed")

    release_audit = audit_gold_defense_release(args.release_dir)
    if not release_audit.valid:
        raise ValueError("Gold HOLDOUT release audit is invalid")
    inference_manifest = GoldHoldoutInferenceManifest.model_validate_json(
        (Path(args.inference_pack) / "manifest.json").read_bytes()
    )
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
            result = export_gold_holdout_inference_pack(
                args.release_dir,
                args.output_dir,
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
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
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-eval: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
