import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutInferenceManifest
from koschei_sentinel.gold_holdout_pack_signing import (
    sign_gold_holdout_inference_pack,
    verify_gold_holdout_inference_pack_signature,
)


def _manifest(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = GoldHoldoutInferenceManifest(
        case_count=1,
        case_ids=["gold-holdout:test"],
        inputs_sha256="a" * 64,
        source_gold_audit_sha256="b" * 64,
    )
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_gold_holdout_pack_signature_verifies_exact_manifest_bytes(tmp_path) -> None:
    manifest_path = _manifest(tmp_path)
    private_key = Ed25519PrivateKey.generate()

    proof = sign_gold_holdout_inference_pack(manifest_path, private_key)
    verify_gold_holdout_inference_pack_signature(
        proof,
        manifest_path,
        private_key.public_key(),
    )

    assert proof.signature_verified is True
    assert proof.inputs_sha256 == "a" * 64
    assert proof.source_gold_audit_sha256 == "b" * 64
    assert proof.proof_sha256


def test_gold_holdout_pack_signature_rejects_manifest_byte_drift(tmp_path) -> None:
    manifest_path = _manifest(tmp_path)
    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_holdout_inference_pack(manifest_path, private_key)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not bind this inference manifest"):
        verify_gold_holdout_inference_pack_signature(
            proof,
            manifest_path,
            private_key.public_key(),
        )


def test_gold_holdout_pack_signature_rejects_rehashed_inputs_manifest(tmp_path) -> None:
    manifest_path = _manifest(tmp_path)
    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_holdout_inference_pack(manifest_path, private_key)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs_sha256"] = "c" * 64
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not bind this inference manifest"):
        verify_gold_holdout_inference_pack_signature(
            proof,
            manifest_path,
            private_key.public_key(),
        )


def test_gold_holdout_pack_signature_rejects_wrong_reviewer_key(tmp_path) -> None:
    manifest_path = _manifest(tmp_path)
    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_holdout_inference_pack(manifest_path, private_key)

    with pytest.raises(ValueError, match="untrusted reviewer key"):
        verify_gold_holdout_inference_pack_signature(
            proof,
            manifest_path,
            Ed25519PrivateKey.generate().public_key(),
        )
