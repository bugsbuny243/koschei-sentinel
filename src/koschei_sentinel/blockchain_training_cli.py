from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_training import (
    BlockchainTrainingPlan,
    execute_blockchain_training,
    load_blockchain_training_config,
    plan_blockchain_training,
    write_blockchain_training_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or explicitly execute Koschei Sentinel blockchain-security "
            "continued pretraining"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_blockchain_training_config(args.config)
        if args.execute:
            try:
                plan = BlockchainTrainingPlan.model_validate_json(
                    open(args.plan, encoding="utf-8").read()
                )
            except (OSError, ValueError) as exc:
                raise ValueError("execution requires a valid existing blockchain plan") from exc
            manifest = execute_blockchain_training(config, plan, root=args.root)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "executed": True,
                        "run_id": manifest.run_id,
                        "base_model": manifest.base_model,
                        "base_revision": manifest.base_revision,
                        "source_corpus_digest": manifest.source_corpus_digest,
                        "benchmark_suite_digest": manifest.benchmark_suite_digest,
                        "train_documents": manifest.train_documents,
                        "validation_documents": manifest.validation_documents,
                        "held_out_test_documents": manifest.held_out_test_documents,
                        "held_out_test_consumed": manifest.held_out_test_consumed,
                        "train_chunks": manifest.train_chunks,
                        "validation_chunks": manifest.validation_chunks,
                        "adapter_digest": manifest.adapter_digest,
                        "output": manifest.output_dir,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        plan = plan_blockchain_training(config, root=args.root)
        write_blockchain_training_plan(plan, args.plan)
        print(
            json.dumps(
                {
                    "ok": True,
                    "executed": False,
                    "run_id": plan.run_id,
                    "base_model": plan.base_model,
                    "base_revision": plan.base_revision,
                    "source_corpus_digest": plan.source_corpus_digest,
                    "benchmark_suite_digest": plan.benchmark_suite_digest,
                    "train_documents": plan.train_documents,
                    "validation_documents": plan.validation_documents,
                    "held_out_test_documents": plan.held_out_test_documents,
                    "estimated_document_steps": plan.estimated_document_steps,
                    "plan": args.plan,
                    "training_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-train: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
