from __future__ import annotations

import json

import koschei_sentinel.cyber_sft_export_cli as cli_module
from koschei_sentinel.cyber_sft_export_verify import CyberSFTExportVerification


def _report() -> CyberSFTExportVerification:
    return CyberSFTExportVerification(
        run_id="cli-export-test",
        valid=True,
        attestation_sha256_verified=True,
        config_sha256_verified=True,
        plan_sha256_verified=True,
        training_source_sha256_verified=True,
        model_preflight_sha256_verified=True,
        verification_sha256_verified=True,
        model_runtime_sha256_verified=True,
        resume_runtime_sha256_verified=True,
        fresh_run_verification_valid=True,
        receipt_binding_verified=True,
        adapter_digest_verified=True,
        corpus_examples_sha256_verified=True,
        corpus_manifest_sha256_verified=True,
        profile_binding_verified=True,
        repository_commit_binding_verified=True,
        violations=[],
    )


def test_export_cli_forwards_all_source_artifacts(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def fake_build(**kwargs):
        observed.update(kwargs)
        return _report()

    monkeypatch.setattr(cli_module, "build_cyber_sft_export", fake_build)
    status = cli_module.main(
        [
            "--config",
            "build/config.json",
            "--plan",
            "build/plan.json",
            "--training-source",
            "build/source.json",
            "--model-preflight",
            "build/preflight.json",
            "--verification",
            "build/verification.json",
            "--attestation",
            "build/attestation.json",
            "--output-dir",
            "build/export",
            "--root",
            "/repo",
        ]
    )

    assert status == 0
    assert observed == {
        "config_path": "build/config.json",
        "plan_path": "build/plan.json",
        "training_source_path": "build/source.json",
        "model_preflight_path": "build/preflight.json",
        "verification_path": "build/verification.json",
        "attestation_path": "build/attestation.json",
        "output_dir": "build/export",
        "root": "/repo",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["run_id"] == "cli-export-test"


def test_export_cli_fails_closed_on_source_error(monkeypatch, capsys) -> None:
    def fail_build(**_kwargs):
        raise ValueError("stale attestation")

    monkeypatch.setattr(cli_module, "build_cyber_sft_export", fail_build)
    status = cli_module.main(
        [
            "--config",
            "config.json",
            "--plan",
            "plan.json",
            "--training-source",
            "source.json",
            "--model-preflight",
            "preflight.json",
            "--verification",
            "verification.json",
            "--attestation",
            "attestation.json",
            "--output-dir",
            "export",
        ]
    )

    assert status == 2
    assert "sentinel-cyber-sft-export: stale attestation" in capsys.readouterr().out
