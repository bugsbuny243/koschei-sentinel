from __future__ import annotations

from fastapi.testclient import TestClient

from koschei_sentinel.service import app


def _case() -> dict[str, object]:
    return {
        "schema_version": "sentinel.case.v1",
        "case_id": "web3-case-1",
        "target_ref": "MintABC",
        "network": "solana-mainnet",
        "signed_verdict": {
            "grade": "D",
            "signature": "signature-123456",
            "triggered_rules": ["URD-C001"],
            "summary": "ARVIS deterministic verdict",
        },
        "evidence": [
            {
                "evidence_id": "arvis-evidence-1",
                "kind": "pump_sybil_radar",
                "statement": "Verified evidence statement.",
                "confidence": "VERIFIED",
                "rule_ids": ["URD-C001"],
                "attributes": {"source": "arvis"},
            }
        ],
        "limitations": [],
    }


def test_opinion_api_requires_bearer_when_token_configured(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("SENTINEL_API_TOKEN", "fabric-secret-token")
    client = TestClient(app)

    missing = client.post("/v1/opinions", json=_case())
    assert missing.status_code == 401

    wrong = client.post(
        "/v1/opinions",
        json=_case(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert wrong.status_code == 401

    accepted = client.post(
        "/v1/opinions",
        json=_case(),
        headers={"Authorization": "Bearer fabric-secret-token"},
    )
    assert accepted.status_code == 200
    body = accepted.json()
    assert body["opinion"]["verdict_signature"] == "signature-123456"
    assert "deterministic verdict is final" in body["opinion"]["authority"]


def test_production_fails_closed_without_api_token(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SENTINEL_API_TOKEN", raising=False)
    client = TestClient(app)

    response = client.post("/v1/opinions", json=_case())
    assert response.status_code == 503
    assert response.json()["detail"] == "sentinel_api_token_not_configured"

    # Health remains available to the deployment platform.
    assert client.get("/health").status_code == 200


def test_local_compatibility_remains_open_without_token(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("SENTINEL_API_TOKEN", raising=False)
    client = TestClient(app)

    assert client.post("/v1/opinions", json=_case()).status_code == 200
