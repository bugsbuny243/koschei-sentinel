from koschei_sentinel.anonymize import anonymize_record

SALT = "unit-test-salt-with-minimum-length"


def test_anonymizer_redacts_and_pseudonymizes() -> None:
    output = anonymize_record(
        {
            "email": "person@example.com",
            "wallet": "RawWalletAddress",
            "note": "Call +90 555 555 55 55 or person@example.com",
        },
        salt=SALT,
    )
    assert output["email"] == "[REDACTED]"
    assert output["wallet"].startswith("wallet_")
    assert "RawWalletAddress" not in str(output)
    assert "person@example.com" not in str(output)


def test_pseudonyms_are_deterministic() -> None:
    first = anonymize_record({"target": "abc"}, salt=SALT)
    second = anonymize_record({"target": "abc"}, salt=SALT)
    assert first == second
