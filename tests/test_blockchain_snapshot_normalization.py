from __future__ import annotations

from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.blockchain_snapshot_normalization import (
    is_license_evidence_path,
    license_text_matches_spdx,
    normalize_snapshot_text,
    select_license_evidence,
)


def test_license_evidence_path_accepts_dual_license_names() -> None:
    assert is_license_evidence_path("LICENSE")
    assert is_license_evidence_path("LICENSE-MIT")
    assert is_license_evidence_path("LICENSE-APACHE")
    assert is_license_evidence_path("COPYING.txt")
    assert not is_license_evidence_path("docs/LICENSE")
    assert not is_license_evidence_path("LICENSES/MIT.txt")


def test_dual_license_repository_can_match_each_declared_license() -> None:
    candidates = {
        "LICENSE-MIT": (
            "MIT License\nPermission is hereby granted, free of charge, to any person "
            "obtaining a copy of this software and associated documentation files "
            "(the Software), to deal in the Software without restriction.\n"
        ),
        "LICENSE-APACHE": "Apache License\nVersion 2.0, January 2004\n",
    }

    assert select_license_evidence("MIT", candidates) == "LICENSE-MIT"
    assert select_license_evidence("Apache-2.0", candidates) == "LICENSE-APACHE"


def test_incident_transaction_hash_is_redacted_instead_of_dropped() -> None:
    transaction_hash = "a" * 64
    text = f"Incident transaction: 0x{transaction_hash}\n"

    normalized = normalize_snapshot_text(
        text,
        salt="0123456789abcdef0123456789abcdef",
    )

    assert normalized.changed is True
    assert normalized.detected_sensitive_kinds == ("long_hex_secret",)
    assert transaction_hash not in normalized.text
    assert "[REDACTED_SECRET]" in normalized.text
    assert detect_sensitive_text(normalized.text) == []


def test_email_and_solana_address_are_normalized_deterministically() -> None:
    address = "11111111111111111111111111111111"
    text = f"Contact analyst@example.com; account {address}.\n"
    salt = "0123456789abcdef0123456789abcdef"

    first = normalize_snapshot_text(text, salt=salt)
    second = normalize_snapshot_text(text, salt=salt)

    assert first == second
    assert first.changed is True
    assert "analyst@example.com" not in first.text
    assert address not in first.text
    assert "[REDACTED_EMAIL]" in first.text
    assert "address_" in first.text
    assert detect_sensitive_text(first.text) == []


def test_clean_text_is_unchanged() -> None:
    text = "Access-control review with no raw account identifiers.\n"

    normalized = normalize_snapshot_text(
        text,
        salt="0123456789abcdef0123456789abcdef",
    )

    assert normalized.text == text
    assert normalized.changed is False
    assert normalized.detected_sensitive_kinds == ()


def test_unsupported_license_never_matches() -> None:
    assert not license_text_matches_spdx("GPL-3.0", "GNU GENERAL PUBLIC LICENSE")
