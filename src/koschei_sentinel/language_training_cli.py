from __future__ import annotations

import argparse
import json

from koschei_sentinel.language_training import (
    execute_language_training,
    load_language_training_config,
    plan_language_training,
    write_language_training_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute offline QLoRA language-foundation training from a pinned "
            "Koschei Language release"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute GPU training after writing and validating the immutable plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_language_training_config(args.config)
        plan = plan_language_training(config, root=args.root)
        write_language_training_plan(plan, args.plan_output)
        manifest = None
        if args.execute:
            manifest = execute_language_training(config, plan, root=args.root)
        payload: dict[str, object] = {
            "ok": True,
            "lineage_stage": plan.lineage_stage,
            "authority": plan.authority,
            "run_id": plan.run_id,
            "base_model": plan.base_model,
            "base_revision": plan.base_revision,
            "source_repository": plan.source_repository,
            "source_commit": plan.source_commit,
            "source_corpus_sha256": plan.source_corpus_sha256,
            "train_documents": plan.train_documents,
            "validation_documents": plan.validation_documents,
            "held_out_test_documents": plan.test_documents,
            "plan_output": args.plan_output,
            "training_started": args.execute,
            "production_authority": False,
        }
        if manifest is not None:
            payload.update(
                {
                    "adapter_digest": manifest.adapter_digest,
                    "adapter_files": len(manifest.adapter_files),
                    "train_chunks": manifest.train_chunks,
                    "validation_chunks": manifest.validation_chunks,
                    "train_loss": manifest.train_loss,
                    "eval_loss": manifest.eval_loss,
                    "output_dir": manifest.output_dir,
                    "training_complete": True,
                }
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-language-train: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
