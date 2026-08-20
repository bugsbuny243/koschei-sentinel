from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_evaluation_cli as cli_module


def test_signed_export_rejects_pack_manifest_from_different_release_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "holdout-pack"
    signature = tmp_path / "holdout-pack.signature.json"
    private_key = SimpleNamespace(public_key=lambda: object())
    signing_called = False

    monkeypatch.setattr(
        cli_module,
        "load_trusted_reviewer_private_key",
        lambda **_kwargs: private_key,
    )
    monkeypatch.setattr(
        cli_module,
        "audit_gold_defense_release",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=True,
            violations=[],
            audit_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "audit_gold_release_review_signatures",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=True,
            violations=[],
            audit_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "snapshot_verified_gold_release",
        lambda *args, **kwargs: (
            Path("release-snapshot"),
            SimpleNamespace(
                release_audit=SimpleNamespace(valid=True, audit_sha256="a" * 64),
                review_signature_audit=SimpleNamespace(
                    valid=True,
                    audit_sha256="b" * 64,
                ),
            ),
        ),
    )

    def fake_export(_release, output):
        output = Path(output)
        output.mkdir()
        (output / "manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            case_count=1,
            source_gold_audit_sha256="c" * 64,
        )

    def forbidden_sign(*_args, **_kwargs):
        nonlocal signing_called
        signing_called = True
        raise AssertionError("mismatched pack must fail before signing")

    monkeypatch.setattr(cli_module, "_export_inputs_atomic", fake_export)
    monkeypatch.setattr(cli_module, "sign_gold_holdout_inference_pack", forbidden_sign)

    with pytest.raises(ValueError, match="manifest differs from release snapshot audit"):
        cli_module._export_signed_inputs(
            release_dir="release",
            output_dir=str(destination),
            reviewer_private_key_path="reviewer.pem",
            reviewer_trust_policy_path="reviewer-trust.json",
            owner_public_key_path="owner-public.pem",
            signature_output=str(signature),
        )

    assert signing_called is False
    assert not destination.exists()
    assert not signature.exists()
