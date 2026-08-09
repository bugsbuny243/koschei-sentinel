from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from koschei_sentinel.language_foundation import (
    LanguageFoundationBlocked,
    _publish_directory_no_replace,
    build_language_foundation_release,
    canonical_json,
    load_source_corpus,
    verify_language_foundation_release,
)


def _reference_family_path(relative: str) -> str:
    if relative in {"README.md", "README.tr.md"}:
        return "README"
    path = Path(relative)
    name = path.name
    for suffix in (".tr.md", ".en.md"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)] + ".md").as_posix()
    return relative


def _source_document(path: str, text: str, *, kind: str) -> dict[str, str]:
    source_sha = hashlib.sha256(text.encode()).hexdigest()
    if kind == "koschei_source":
        parts = Path(path).parts
        family = f"example:{parts[1]}" if len(parts) >= 3 else "example:top-level"
    else:
        family = f"reference:{_reference_family_path(path)}"
    document_id = hashlib.sha256(
        f"{kind}\0{family}\0{path}\0{source_sha}".encode()
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
        _source_document(
            "docs/capabilities.md",
            "No ambient authority.\n",
            kind="reference",
        ),
        _source_document("docs/types.md", "Option and Result.\n", kind="reference"),
        _source_document(
            "examples/app.ks",
            "import risk\nfn main() { risk.label(1) }\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/risk.ks",
            "fn label(value: Int) -> Int { return value }\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/hello/main.ks",
            'fn main() { println("hello") }\n',
            kind="koschei_source",
        ),
        _source_document(
            "examples/loops/main.ks",
            "fn main() { let mut x = 3 }\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/supply_chain/analytics.ks",
            "fn track() { }\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/supply_chain/main.ks",
            "import analytics\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/maps/main.ks",
            "fn main() { }\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/options/main.ks",
            "fn main() { let x = None }\n",
            kind="koschei_source",
        ),
        _source_document(
            "examples/results/main.ks",
            "fn main() { }\n",
            kind="koschei_source",
        ),
    ]
    documents.sort(key=lambda item: item["path"])
    total_bytes = sum(len(item["text"].encode()) for item in documents)
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
    payload["corpus_sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _trusted_build(
    corpus: Path,
    output: Path,
    *,
    seed: str = "koschei-language-foundation-v1",
):
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    return build_language_foundation_release(
        corpus,
        output_dir=output,
        split_seed=seed,
        expected_source_commit=payload["source_commit"],
        expected_source_corpus_sha256=payload["corpus_sha256"],
    )


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
        payload = _write_source_corpus(corpus)
        readme_families = {
            item["family"]
            for item in payload["documents"]
            if item["path"].startswith("README")
        }
        assert readme_families == {"reference:README"}
        top_level_families = {
            item["family"]
            for item in payload["documents"]
            if item["path"] in {"examples/app.ks", "examples/risk.ks"}
        }
        assert top_level_families == {"example:top-level"}

        first = root / "release-a"
        second = root / "release-b"
        first_manifest = _trusted_build(corpus, first, seed="seed-v1")
        second_manifest = _trusted_build(corpus, second, seed="seed-v1")
        assert first_manifest.model_dump(mode="json") == second_manifest.model_dump(mode="json")
        train = _families(first, "train")
        validation = _families(first, "validation")
        test = _families(first, "test")
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert "example:supply_chain" in train | validation | test
        assert "example:top-level" in train | validation | test
        verify_language_foundation_release(first)


def test_source_tampering_is_rejected_before_split() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        original = _write_source_corpus(corpus)
        payload = json.loads(json.dumps(original))
        payload["documents"][0]["text"] += "tampered"
        corpus.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(LanguageFoundationBlocked, match="hash mismatch"):
            build_language_foundation_release(
                corpus,
                output_dir=root / "release",
                expected_source_commit=original["source_commit"],
                expected_source_corpus_sha256=original["corpus_sha256"],
            )


def test_rehashed_malicious_corpus_fails_trusted_digest_pin() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        original = _write_source_corpus(corpus)
        payload = json.loads(json.dumps(original))
        changed = payload["documents"][0]
        changed["text"] += "malicious but internally rehashed"
        changed["source_sha256"] = hashlib.sha256(changed["text"].encode()).hexdigest()
        changed["document_id"] = hashlib.sha256(
            (
                f"{changed['kind']}\0{changed['family']}\0{changed['path']}\0"
                f"{changed['source_sha256']}"
            ).encode()
        ).hexdigest()
        payload["total_bytes"] = sum(
            len(item["text"].encode()) for item in payload["documents"]
        )
        digest_payload = dict(payload)
        digest_payload.pop("corpus_sha256")
        payload["corpus_sha256"] = hashlib.sha256(
            canonical_json(digest_payload).encode()
        ).hexdigest()
        corpus.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(LanguageFoundationBlocked, match="trusted expected digest"):
            build_language_foundation_release(
                corpus,
                output_dir=root / "release",
                expected_source_commit=original["source_commit"],
                expected_source_corpus_sha256=original["corpus_sha256"],
            )


def test_duplicate_source_json_members_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        corpus = Path(temporary) / "corpus.json"
        corpus.write_text(
            '{"schema_version":"one","schema_version":"two"}',
            encoding="utf-8",
        )
        with pytest.raises(LanguageFoundationBlocked, match="duplicate JSON"):
            load_source_corpus(corpus)


def test_expected_source_commit_is_a_hard_pin() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        payload = _write_source_corpus(corpus, commit="b" * 40)
        with pytest.raises(LanguageFoundationBlocked, match="does not match"):
            build_language_foundation_release(
                corpus,
                output_dir=root / "release",
                expected_source_commit="c" * 40,
                expected_source_corpus_sha256=payload["corpus_sha256"],
            )


def test_release_tampering_is_detected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        release = root / "release"
        _trusted_build(corpus, release)
        train = release / "train.jsonl"
        train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(LanguageFoundationBlocked, match="digest mismatch"):
            verify_language_foundation_release(release)


def test_manifest_corpus_digest_must_match_released_documents() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        release = root / "release"
        _trusted_build(corpus, release)
        manifest_path = release / "language-foundation-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_corpus_sha256"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(LanguageFoundationBlocked, match="reconstruct"):
            verify_language_foundation_release(release)


def test_manifest_provenance_fields_are_required() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        release = root / "release"
        _trusted_build(corpus, release)
        manifest_path = release / "language-foundation-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["source_schema"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(LanguageFoundationBlocked, match="manifest is invalid"):
            verify_language_foundation_release(release)


def test_release_directory_is_no_replace() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus.json"
        _write_source_corpus(corpus)
        release = root / "release"
        _trusted_build(corpus, release)
        with pytest.raises(FileExistsError):
            _trusted_build(corpus, release)


def test_publish_never_replaces_a_raced_destination_directory() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staged = root / "staged"
        staged.mkdir()
        (staged / "train.jsonl").write_text("safe\n", encoding="utf-8")
        destination = root / "release"
        destination.mkdir()
        sentinel = destination / "owned-by-other-process"
        sentinel.write_text("keep\n", encoding="utf-8")

        with pytest.raises(FileExistsError):
            _publish_directory_no_replace(staged, destination)
        assert sentinel.read_text(encoding="utf-8") == "keep\n"
