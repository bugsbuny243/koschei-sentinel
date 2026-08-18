from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutPrediction,
    evaluate_gold_holdout_predictions,
    export_gold_holdout_inference_pack,
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
    return parser


def _load_predictions(path: str) -> list[GoldHoldoutPrediction]:
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
    if not rows:
        raise ValueError("Gold HOLDOUT predictions file is empty")
    return rows


def _load_policy(path: str | None) -> GoldHoldoutEvaluationPolicy | None:
    if path is None:
        return None
    try:
        return GoldHoldoutEvaluationPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Gold HOLDOUT evaluation policy: {path}") from exc


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

        predictions = _load_predictions(args.predictions)
        report = evaluate_gold_holdout_predictions(
            args.release_dir,
            predictions,
            policy=_load_policy(args.policy),
        )
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report.passed else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-eval: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
