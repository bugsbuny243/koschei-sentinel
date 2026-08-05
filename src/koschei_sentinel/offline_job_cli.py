from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.offline_job import (
    OfflineJobBlocked,
    build_offline_training_job,
    load_autotrain_plan,
    write_offline_training_job,
)
from koschei_sentinel.training import (
    load_training_config,
    plan_training,
    write_training_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sealed, manually dispatched offline GPU training job "
            "from an approved Sentinel autotrain plan"
        )
    )
    parser.add_argument("--autotrain-plan", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-plan-output", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        job_path = Path(args.output)
        training_plan_path = Path(args.training_plan_output)
        if job_path.exists():
            raise FileExistsError(f"offline job already exists: {job_path}")
        if training_plan_path.exists():
            raise FileExistsError(f"training plan already exists: {training_plan_path}")

        autotrain = load_autotrain_plan(args.autotrain_plan)
        config = load_training_config(args.config)
        plan = plan_training(config)
        job = build_offline_training_job(
            autotrain,
            config,
            plan,
            config_path=args.config,
            training_plan_path=args.training_plan_output,
        )
        write_training_plan(plan, training_plan_path)
        write_offline_training_job(job, job_path)
        print(json.dumps(job.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except OfflineJobBlocked as exc:
        print(f"sentinel-job-plan: blocked: {exc}")
        return 3
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-job-plan: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
