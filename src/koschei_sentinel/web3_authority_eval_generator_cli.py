from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from koschei_sentinel.web3_authority_eval_generator import (
    generate_eval_bundle_from_paths,
    write_generated_eval_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic Web3 authority eval variants")
    parser.add_argument("--config", type=Path, default=Path("configs/training/web3-authority-evals.v1.json"))
    parser.add_argument("--seeds", type=Path, default=Path("configs/training/web3-authority-eval-seeds.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = generate_eval_bundle_from_paths(args.config, args.seeds)
    write_generated_eval_bundle(bundle, args.output)
    family_counts = Counter(case.family for case in bundle.cases)
    print(f"PASS generated_cases={len(bundle.cases)} families={len(family_counts)} output={args.output}")
    for family, count in sorted(family_counts.items()):
        print(f"{family}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
