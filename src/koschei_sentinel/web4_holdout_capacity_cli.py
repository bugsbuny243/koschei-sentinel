from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.promotion import load_owner_public_key
from koschei_sentinel.training import atomic_write
from koschei_sentinel.web4_holdout_capacity import build_web4_holdout_capacity_report
from koschei_sentinel.web4_holdout_release import Web4HoldoutRelease


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-holdout-capacity",
        description=(
            "Measure an owner-signed Web4 research HOLDOUT release against the committed "
            "benchmark capacity policy without granting model, training, or production authority."
        ),
    )
    parser.add_argument("--release", required=True)
    parser.add_argument("--benchmark-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--output")
    return parser


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate


def _preflight_output(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Web4 HOLDOUT capacity output already exists: {destination}")
    current = destination.parent
    while True:
        if current.is_symlink():
            raise ValueError("Web4 HOLDOUT capacity output path must not traverse symlink directories")
        if current.parent == current:
            break
        current = current.parent
    return destination


def _load_release(path: str | Path) -> Web4HoldoutRelease:
    release_path = _regular_file(path, "Web4 HOLDOUT release")
    try:
        return Web4HoldoutRelease.model_validate_json(release_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid Web4 HOLDOUT release") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = _preflight_output(args.output) if args.output is not None else None
        release = _load_release(args.release)
        benchmark_policy = _regular_file(args.benchmark_policy, "Web4 benchmark policy")
        owner_public_key_path = _regular_file(args.owner_public_key, "owner public key")
        report = build_web4_holdout_capacity_report(
            release=release,
            owner_public_key=load_owner_public_key(owner_public_key_path),
            benchmark_policy_path=benchmark_policy,
        )
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if output is not None:
            atomic_write(output, payload)
        print(payload, end="")
        return 0 if report.research_benchmark_capacity_ready else 1
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-web4-holdout-capacity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
