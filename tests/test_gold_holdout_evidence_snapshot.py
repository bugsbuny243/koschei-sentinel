from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_evaluation_evidence as evidence_module


def test_signed_evidence_rebuild_consumes_revalidated_snapshot(monkeypatch) -> None:
    observed_pack = None
    proof = SimpleNamespace(
        review_signature_audit_sha256="b" * 64,
        proof_sha256="c" * 64,
    )

    monkeypatch.setattr(
        evidence_module,
        "audit_gold_defense_release",
        lambda *_args: SimpleNamespace(valid=True, audit_sha256="a" * 64),
    )
    monkeypatch.setattr(
        evidence_module,
        "verify_admitted_gold_holdout_pack",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence_module,
        "snapshot_admitted_gold_holdout_pack",
        lambda admission, inference_pack, destination: Path("revalidated-snapshot"),
    )

    def stop_after_snapshot(*, pack, **_kwargs):
        nonlocal observed_pack
        observed_pack = Path(pack)
        raise ValueError("stop after signed snapshot admission")

    monkeypatch.setattr(
        evidence_module,
        "_load_pack_evaluation_state",
        stop_after_snapshot,
    )

    with pytest.raises(ValueError, match="stop after signed snapshot admission"):
        evidence_module.build_gold_holdout_evaluation_evidence(
            release_dir="release",
            inference_pack_dir="pack",
            inference_output_dir="output",
            candidate_export_dir="candidate-export",
            reviewer_public_key=object(),
            inference_pack_signature_proof=proof,
        )

    assert observed_pack == Path("revalidated-snapshot")
