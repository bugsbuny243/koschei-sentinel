from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from koschei_sentinel.language_foundation import (
    LanguageFoundationBlocked,
    build_language_foundation_release,
    canonical_json,
    verify_language_foundation_release,
)


def _source_document(path: str, text: str, *, kind: str) -> dict[str, str]:
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if kind == "koschei_source":
        family = f"example:{Path(path).parts[1]}"
    else:
        family = f"reference:{path}"
    document_id = hashlib.sha256(
        f"{kind}\0{family}\0{path}\0{source_sha}".encode("utf-8")
    ).hexdigest()
    return {
        "document_id": document_id,
        "family": family,
        "kind": kind,
        "path": path,
        "source_sha256": source_sha,
        "text": text,
    }


def _write_source_corpus(path: Path, *, commit: str = "a" * 40) -> dict[str, object]:
    documents = [
        _source_document("README.md", "# Koschei\n", kind="reference"),
        _source_document("README.tr.md", "# Koschei TR\n", kind="reference"),
        _source_document("docs/capabilities.md", "No ambient authority.\n", kind="reference"),
        _source_document("docs/types.md", "Option and Result.\n", kind="reference"),
        _source_document("examples/hello/main.ks", 'fn main() { println("hello") }\n', kind="koschei_source"),
        _source_document("examples/loops/main.ks", "fn main() { let mut x = 3 }\n", kind="koschei_source"),
        _source_document("examples/supply_chain/analytics.ks", "fn track() { }\n", kind="koschei_source"),
        _source_document("examples/supply_chain/main.ks", "import analytics\n", kind="koschei_source"),
        _source_document("examples/maps/main.ks", "fn main() { }\n", kind="koschei_source"),
        _source_document("examples/options/main.ks", "fn main() { let x = None }\n", kind="koschei_source"),
        _source_document("examples/results/main.ks", "fn main() { }\n", kind="koschei_source"),
    ]
    documents.sort(key=lambda item: item["path"])
    total_bytes = sum(len(item["text"].encode("utf-8")) for item in documents)
    payload = {
        "schema_version": "koschei.language-foundation-corpus.v1",
        "generator_version": "koschei-foundation-export/v1",
        "source_repository": "bugsbuny243/koschei-lang",
        "source_commit": commit,
        "document_count": len(documents),
        "family_count": len({item["family"] for item in documents}),
        "total_bytes": total_bytes,
        "documents": documents,
    }
    payload["corpus_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _families(release: Path, split: str) -> set[str]:
    return {
        json.loads(line)["family"]
        for line in (release / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_release_is_deterministic_and_has_no_family_leakage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        first = root / "release-a"
        second = root / "release-b"
        first_manifest = build_language_foundation_release(corpus, output_dir=first, split_seed="seed-v1")
        second_manifest = build_language_foundation_release(corpus, output_dir=second, split_seed="seed-v1")
        assert first_manifest.model_dump(mode="json") == second_manifest.model_dump(mode="json")
        train = _families(first, "train")
        validation = _families(first, "validation")
        test = _families(first, "test")
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert "example:supply_chain" in train | validation | test
        verify_language_foundation_release(first)


def test_source_tampering_is_rejected_before_split() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        payload = _write_source_corpus(corpus)
        payload["documents"][0]["text"] += "tampered"
        corpus.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(LanguageFoundationBlocked, match="hash mismatch"):
            build_language_foundation_release(corpus, output_dir=root / "release")


def test_expected_source_commit_is_a_hard_pin() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus, commit="b" * 40)
        with pytest.raises(LanguageFoundationBlocked, match="does not match"):
            build_language_foundation_release(
                corpus,
                output_dir=root / "release",
                expected_source_commit="c" * 40,
            )


def test_release_tampering_is_detected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        release = root / "release"
        build_language_foundation_release(corpus, output_dir=release)
        train = release / "train.jsonl"
        train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(LanguageFoundationBlocked, match="digest mismatch"):
            verify_language_foundation_release(release)


def test_release_directory_is_no_replace() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        release = root / "release"
        build_language_foundation_release(corpus, output_dir=release)
        with pytest.raises(FileExistsError):
            build_language_foundation_release(corpus, output_dir=release)
