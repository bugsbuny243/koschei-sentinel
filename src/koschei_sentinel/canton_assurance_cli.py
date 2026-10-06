"""Read-only CLI for the Canton assurance proof-of-work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.canton_assurance import run_proof


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Canton evidence assurance proof")
    parser.add_argument("fixture", type=Path, help="Path to a bounded JSON fixture")
    args = parser.parse_args()

    source = json.loads(args.fixture.read_text(encoding="utf-8"))
    result = run_proof(source)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["finding"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
