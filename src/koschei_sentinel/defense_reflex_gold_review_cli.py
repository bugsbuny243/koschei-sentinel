from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_gold_queue import GoldDefenseReviewPacket
from koschei_sentinel.defense_reflex_gold_review import (
    GoldHumanReviewSpec,
    review_gold_packet,
)
from koschei_sentinel.gold_review_signing import sign_gold_reviewed_packet
from koschei_sentinel.gold_reviewer_trust import load_trusted_reviewer_private_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply and Ed25519-sign a fail-closed human review to one pre-split "
            "Gold Defense Reflex packet"
        )
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--review-spec", required=True)
    parser.add_argument("--reviewer-private-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--signature-output", required=True)
    return parser


def _load(path: str, model_type, label: str):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    try:
        return model_type.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        packet = _load(args.packet, GoldDefenseReviewPacket, "Gold review packet")
        scenario = _load(args.scenario, CyberRangeScenario, "Cyber Range scenario")
        spec = _load(args.review_spec, GoldHumanReviewSpec, "Gold human review spec")
        reviewer_private_key = load_trusted_reviewer_private_key(
            reviewer_private_key_path=args.reviewer_private_key,
            trust_policy_path=args.reviewer_trust_policy,
            owner_public_key_path=args.owner_public_key,
        )
        reviewed = review_gold_packet(packet, scenario, spec)
        signature = sign_gold_reviewed_packet(reviewed, reviewer_private_key)

        output = Path(args.output)
        signature_output = Path(args.signature_output)
        for destination, label in (
            (output, "Gold reviewed packet"),
            (signature_output, "Gold review signature"),
        ):
            if destination.exists():
                raise FileExistsError(f"{label} output already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            json.dumps(reviewed.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        signature_output.write_text(
            json.dumps(signature.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(reviewed.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-review: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
