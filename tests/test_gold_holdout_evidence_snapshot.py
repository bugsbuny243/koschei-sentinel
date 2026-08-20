from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_evaluation_evidence as evidence_module


def test_signed_evidence_rebuild_consumes_sealed_source_snapshots(monkeypatch) -> None:
    observed_release = None
    observed_pack = None
    observed_candidate = None
    proof = SimpleNamespace(
        source_gold_audit_sha256="a" * 64,
        review_signature_audit_sha256="b" * 64,
        proof_sha256="c" * 64,
    )

    monkeypatch.setattr(
        evidence_module,
        "verify_admitted_gold_holdout_pack",
        lambda *_args, **_kwargs: None,
    )

    def fake_release_snapshot(
        release_dir,
        destination,
        *,
        reviewer_public_key,
        expected_release_audit_sha256,
        expected_review_signature_audit_sha256,
    ):
        nonlocal observed_release
        assert release_dir == "release"
        assert expected_release_audit_sha256 == "a" * 64
        assert expected_review_signature_audit_sha256 == "b" * 64
        observed_release = Path("revalidated-release")
        return observed_release, SimpleNamespace(
            release_audit=SimpleNamespace(valid=True, audit_sha256="a" * 64),
            review_signature_audit=SimpleNamespace(valid=True, audit_sha256="b" * 64),
        )

    monkeypatch.setattr(
        evidence_module,
        "snapshot_verified_gold_release",
        fake_release_snapshot,
    )
    monkeypatch.setattr(
        evidence_module,
        "snapshot_admitted_gold_holdout_pack",
        lambda admission, inference_pack, destination: Path("revalidated-pack"),
    )
    monkeypatch.setattr(
        evidence_module,
        "snapshot_verified_cyber_sft_export",
        lambda candidate_export, destination: Path("revalidated-candidate"),
    )

    def stop_after_snapshots(
        output_dir,
        destination,
        *,
        inference_pack_dir,
        candidate_export_dir,
    ):
        nonlocal observed_pack, observed_candidate
        assert output_dir == "output"
        observed_pack = Path(inference_pack_dir)
        observed_candidate = Path(candidate_export_dir)
        raise ValueError("stop after sealed source snapshots")

    monkeypatch.setattr(
        evidence_module,
        "snapshot_verified_gold_holdout_inference_output",
        stop_after_snapshots,
    )

    with pytest.raises(ValueError, match="stop after sealed source snapshots"):
        evidence_module.build_gold_holdout_evaluation_evidence(
            release_dir="release",
            inference_pack_dir="pack",
            inference_output_dir="output",
            candidate_export_dir="candidate-export",
            reviewer_public_key=object(),
            inference_pack_signature_proof=proof,
        )

    assert observed_release == Path("revalidated-release")
    assert observed_pack == Path("revalidated-pack")
    assert observed_candidate == Path("revalidated-candidate")
