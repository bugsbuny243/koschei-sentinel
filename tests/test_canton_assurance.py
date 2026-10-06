from koschei_sentinel.canton_assurance import run_proof


def _complete_fixture() -> dict[str, object]:
    return {
        "source_class": "public_test_fixture",
        "subject_ref": "canton-test-app",
        "artifact_version": "fixture-v1",
        "provenance": "public/reproducible-test-fixture",
        "integrity": "fixture-integrity-ok",
        "expected_evidence": True,
        "observed_evidence": True,
    }


def test_identical_input_is_deterministic_and_evidence_grounded() -> None:
    first = run_proof(_complete_fixture())
    second = run_proof(_complete_fixture())

    assert first == second
    assert first["evidence"]["schema"] == "canton.security.evidence.v1"
    assert first["finding"]["status"] == "pass"
    assert first["finding"]["evidence_ids"] == [first["evidence"]["evidence_id"]]


def test_missing_expected_evidence_fails_with_same_evidence_reference() -> None:
    fixture = _complete_fixture()
    fixture["observed_evidence"] = False
    result = run_proof(fixture)

    assert result["finding"]["status"] == "fail"
    assert result["finding"]["severity"] == "high"
    assert result["finding"]["evidence_ids"] == [result["evidence"]["evidence_id"]]


def test_missing_provenance_returns_insufficient_evidence_not_fabricated_pass() -> None:
    fixture = _complete_fixture()
    fixture["provenance"] = None
    result = run_proof(fixture)

    assert result["finding"]["status"] == "insufficient_evidence"
    assert result["finding"]["severity"] == "unknown"
    assert "provenance" in result["finding"]["reason"]
