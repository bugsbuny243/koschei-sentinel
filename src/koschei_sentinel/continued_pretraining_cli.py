from __future__ import annotations

import argparse
import json

from koschei_sentinel.continued_pretraining import (
    load_continued_pretraining_config,
    plan_continued_pretraining,
    write_continued_pretraining_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a Koschei Sentinel Stage 2 continued-pretraining run. "
            "This command never loads a model and never starts training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = plan_continued_pretraining(
            load_continued_pretraining_config(args.config),
            root=args.root,
        )
        write_continued_pretraining_plan(plan, args.output)
        print(
            json.dumps(
                {
                    "ok": True,
                    "lineage_stage": plan.lineage_stage,
                    "run_id": plan.run_id,
                    "base_model": plan.base_model,
                    "base_revision": plan.base_revision,
                    "documents": plan.documents,
                    "unique_families": plan.unique_families,
                    "benchmark_suite_digest": plan.benchmark_suite_digest,
                    "corpus_digest": plan.corpus_digest,
                    "estimated_optimizer_steps": plan.estimated_optimizer_steps,
                    "output": args.output,
                    "training_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-stage2-plan: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
