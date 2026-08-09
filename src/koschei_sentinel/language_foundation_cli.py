from __future__ import annotations

import argparse
import json

from koschei_sentinel.language_foundation import (
    LanguageFoundationBlocked,
    build_language_foundation_release,
    verify_language_foundation_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify leakage-safe Koschei language foundation data"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser(
        "build",
        help="Build train/validation/test from a pinned corpus",
    )
    build.add_argument("--corpus", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--seed", default="koschei-language-foundation-v1")
    build.add_argument("--expect-source-commit")

    verify = subcommands.add_parser(
        "verify",
        help="Verify a materialized language foundation release",
    )
    verify.add_argument("release")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_language_foundation_release(
                args.corpus,
                output_dir=args.output,
                split_seed=args.seed,
                expected_source_commit=args.expect_source_commit,
            )
        else:
            manifest = verify_language_foundation_release(args.release)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, LanguageFoundationBlocked, OSError, ValueError) as exc:
        print(f"sentinel-language-foundation: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
