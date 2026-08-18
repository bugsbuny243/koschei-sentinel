from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_sft_text_trainer import execute_cyber_sft_text
from koschei_sentinel.cyber_sft_training import load_cyber_sft_config, plan_cyber_sft
from koschei_sentinel.cyber_training_readiness import audit_cyber_training_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute evidence-grounded Koschei Sentinel Cyber SFT"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-output")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the text-only Qwen3.5 causal-LM QLoRA executor after local corpus planning; "
            "default is a network-free dry run"
        ),
    )
    return parser


def _assert_gold_execution_gate(config, plan) -> None:
    if config.stage.value != "DEFENSE_REFLEX" or plan.corpus_promotion_eligible is not True:
        return
    readiness = audit_cyber_training_readiness(
        config,
        check_runtime=False,
        check_tokenization=False,
    )
    if not readiness.gold_release_audit_checked or readiness.gold_release_audit_valid is not True:
        detail = "; ".join(readiness.blockers[:8]) or "Gold release audit was not verified"
        raise RuntimeError(
            "promotion-eligible Defense Reflex execution requires a valid Gold release audit: "
            + detail
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_cyber_sft_config(args.config)
        plan = plan_cyber_sft(config)
        if args.plan_output:
            destination = Path(args.plan_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.execute:
            _assert_gold_execution_gate(config, plan)
            result = execute_cyber_sft_text(config, plan)
        else:
            result = plan
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-cyber-sft: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
