from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeReport
from koschei_sentinel.defense_reflex_candidates import mine_defense_reflex_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine fail-closed Defense Reflex review candidates from one Cyber Range report"
    )
    parser.add_argument("--report", required=True, help="Cyber Range scenario report JSON")
    parser.add_argument("--output-dir", required=True)
    return parser


def _filename(candidate_id: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    return f"candidate-{digest}.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = CyberRangeReport.model_validate_json(
            Path(args.report).read_text(encoding="utf-8")
        )
        candidates = mine_defense_reflex_candidates(report)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        index_rows: list[dict[str, str]] = []
        for candidate in candidates:
            filename = _filename(candidate.candidate_id)
            payload = json.dumps(candidate.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            (output / filename).write_text(payload, encoding="utf-8")
            index_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "failure_type": candidate.failure_type.value,
                    "file": filename,
                    "source_report_sha256": candidate.source_report_sha256,
                }
            )

        index = {
            "schema_version": "sentinel.defense-reflex-candidate-index.v1",
            "scenario_id": report.scenario_id,
            "candidate_count": len(candidates),
            "candidates": sorted(index_rows, key=lambda row: row["candidate_id"]),
        }
        (output / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(index, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-reflex-mine: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
