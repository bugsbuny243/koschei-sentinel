from __future__ import annotations

from pathlib import Path
import tempfile

from koschei_sentinel.language_foundation import (
    build_language_foundation_release,
    load_source_corpus,
    verify_language_foundation_release,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VECTOR = (
    REPO_ROOT
    / "fixtures"
    / "language_foundation"
    / "language-foundation-contract-vector.v1.json"
)
EXPECTED_COMMIT = "a" * 40
EXPECTED_CORPUS_SHA256 = "873e609284da5e8834bf103659ae42b90cee27710d11d4bc0ad6136200151144"


def test_shared_language_foundation_vector_is_accepted_byte_for_contract() -> None:
    corpus = load_source_corpus(VECTOR)
    assert corpus.source_commit == EXPECTED_COMMIT
    assert corpus.corpus_sha256 == EXPECTED_CORPUS_SHA256
    assert corpus.document_count == 6
    assert corpus.family_count == 5


def test_shared_vector_builds_and_verifies_leakage_safe_release() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        release = Path(temporary) / "release"
        manifest = build_language_foundation_release(
            VECTOR,
            output_dir=release,
            expected_source_commit=EXPECTED_COMMIT,
            expected_source_corpus_sha256=EXPECTED_CORPUS_SHA256,
            split_seed="cross-repo-contract-vector-v1",
        )
        assert manifest.leakage_detected is False
        assert manifest.documents == 6
        assert manifest.families == 5
        verified = verify_language_foundation_release(release)
        assert verified == manifest
