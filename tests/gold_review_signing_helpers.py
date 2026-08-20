import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_review_signing import sign_gold_reviewed_packet
from tests.test_defense_reflex_gold_release import _release_rows


def attach_signed_review_proofs(release_dir):
    _policy, rows = _release_rows()
    reviewer_private_key = Ed25519PrivateKey.generate()
    proofs = [
        sign_gold_reviewed_packet(reviewed, reviewer_private_key)
        for _scenario, _packet, reviewed in rows
    ]
    ordered = sorted(
        proofs,
        key=lambda row: (row.split.value, row.scenario_id, row.review_sha256),
    )
    payload = "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in ordered
    )
    (release_dir / "review-signatures.jsonl").write_text(payload, encoding="utf-8")
    return reviewer_private_key.public_key()
