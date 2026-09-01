import pytest

from koschei_sentinel.defense_reflex_gold_production_release import (
    GoldProductionReleaseReceipt,
    _sha256_canonical,
    build_gold_production_release,
)
from koschei_sentinel.defense_reflex_gold_production_release_cli import main as cli_main
from tests.test_defense_reflex_gold_capacity import _write_artifacts


def _build_from_fixture(tmp_path, *, receipt_output=None):
    dirs, policy, reviewer_key, trust, owner_key = _write_artifacts(tmp_path)
    output = tmp_path / "production-release"
    return (
        lambda: build_gold_production_release(
            scenario_dir=dirs["scenarios"],
            packet_dir=dirs["packets"],
            review_dir=dirs["reviews"],
            signature_dir=dirs["signatures"],
            split_policy_path=policy,
            reviewer_public_key_path=reviewer_key,
            reviewer_trust_policy_path=trust,
            owner_public_key_path=owner_key,
            output_dir=output,
            receipt_output=receipt_output,
        ),
        output,
        (dirs, policy, reviewer_key, trust, owner_key),
    )


def test_production_release_rejects_capacity_shortfall_without_output(tmp_path) -> None:
    receipt = tmp_path / "production-receipt.json"
    build, output, _fixture = _build_from_fixture(tmp_path, receipt_output=receipt)

    with pytest.raises(
        ValueError,
        match="Gold production release capacity gate is not ready",
    ):
        build()

    assert not output.exists()
    assert not receipt.exists()


def test_production_release_preflights_existing_receipt_before_input_reads(tmp_path) -> None:
    receipt = tmp_path / "production-receipt.json"
    receipt.write_text("already exists\n", encoding="utf-8")
    output = tmp_path / "production-release"

    with pytest.raises(FileExistsError, match="production release receipt exists"):
        build_gold_production_release(
            scenario_dir=tmp_path / "missing-scenarios",
            packet_dir=tmp_path / "missing-packets",
            review_dir=tmp_path / "missing-reviews",
            signature_dir=tmp_path / "missing-signatures",
            split_policy_path=tmp_path / "missing-split-policy.json",
            reviewer_public_key_path=tmp_path / "missing-reviewer.pem",
            reviewer_trust_policy_path=tmp_path / "missing-trust.json",
            owner_public_key_path=tmp_path / "missing-owner.pem",
            output_dir=output,
            receipt_output=receipt,
        )

    assert not output.exists()
    assert receipt.read_text(encoding="utf-8") == "already exists\n"


def test_production_release_rejects_receipt_inside_release_directory(tmp_path) -> None:
    output = tmp_path / "production-release"

    with pytest.raises(ValueError, match="receipt must be outside the release directory"):
        build_gold_production_release(
            scenario_dir=tmp_path / "missing-scenarios",
            packet_dir=tmp_path / "missing-packets",
            review_dir=tmp_path / "missing-reviews",
            signature_dir=tmp_path / "missing-signatures",
            split_policy_path=tmp_path / "missing-split-policy.json",
            reviewer_public_key_path=tmp_path / "missing-reviewer.pem",
            reviewer_trust_policy_path=tmp_path / "missing-trust.json",
            owner_public_key_path=tmp_path / "missing-owner.pem",
            output_dir=output,
            receipt_output=output / "receipt.json",
        )

    assert not output.exists()


def test_production_release_receipt_self_hash_is_bound() -> None:
    unsigned = {
        "schema_version": "sentinel.gold-production-release-receipt.v1",
        "split_policy_sha256": "1" * 64,
        "capacity_sha256": "2" * 64,
        "release_manifest_sha256": "3" * 64,
        "release_sha256": "4" * 64,
        "release_audit_sha256": "5" * 64,
        "review_signature_audit_sha256": "6" * 64,
        "reviewer_trust_policy_digest": "7" * 64,
        "owner_key_fingerprint": "8" * 64,
        "train_examples": 1,
        "validation_examples": 1,
        "holdout_cases": 50,
        "minimum_holdout_cases": 50,
        "production_ready": True,
    }
    receipt = GoldProductionReleaseReceipt(
        **unsigned,
        receipt_sha256=_sha256_canonical(unsigned),
    )

    assert receipt.production_ready is True
    assert receipt.holdout_cases == 50
    with pytest.raises(ValueError, match="receipt self-hash does not verify"):
        receipt.model_copy(update={"holdout_cases": 51}).model_dump()


def test_production_release_cli_rejects_capacity_shortfall(tmp_path, capsys) -> None:
    build, output, fixture = _build_from_fixture(tmp_path)
    del build
    dirs, policy, reviewer_key, trust, owner_key = fixture

    exit_code = cli_main(
        [
            "--scenarios",
            str(dirs["scenarios"]),
            "--packets",
            str(dirs["packets"]),
            "--reviews",
            str(dirs["reviews"]),
            "--signatures",
            str(dirs["signatures"]),
            "--split-policy",
            str(policy),
            "--reviewer-public-key",
            str(reviewer_key),
            "--reviewer-trust-policy",
            str(trust),
            "--owner-public-key",
            str(owner_key),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 2
    assert "capacity gate is not ready" in capsys.readouterr().out
    assert not output.exists()
