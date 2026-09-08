from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from koschei_sentinel.strict_json import strict_json_loads
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkCaseProposal,
    Web4BenchmarkIntakePacket,
    build_web4_benchmark_intake,
    write_web4_benchmark_intake_packet,
)
from koschei_sentinel.web4_research_snapshot import (
    build_web4_research_snapshot_receipt,
    verify_web4_research_snapshot_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/web4/benchmark-intake"
SOURCES = ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"


def build(**overrides: object) -> Web4BenchmarkIntakePacket:
    args = {
        "proposal_path": FIXTURE / "proposal.json",
        "answer_key_path": FIXTURE / "answer-key.json",
        "source_registry_path": SOURCES,
        "benchmark_policy_path": ROOT / "evals/web4-security-benchmark.v1.json",
        "intake_policy_path": ROOT / "evals/web4-benchmark-intake-policy.v1.json",
        "snapshot_root": FIXTURE,
    }
    args.update(overrides)
    return build_web4_benchmark_intake(**args)


def reseal(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("packet_sha256", None)
    payload["packet_sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return payload


class Web4IntakeIntegrityTests(unittest.TestCase):
    def test_valid_fixture_keeps_its_v1_hash_and_closed_authorizations(self) -> None:
        packet = build()
        self.assertEqual(packet.packet_sha256, "5799dfef198d589f794537cd458f7b2f3b3afe1a98063c517ee11fb4d0d1a1cd")
        self.assertEqual(Web4BenchmarkIntakePacket.model_validate_json(packet.model_dump_json()), packet)
        for flag in ("human_reviewed", "contains_answer_key", "training_authorization",
                     "evaluation_authorization", "promotion_eligible"):
            self.assertIs(getattr(packet, flag), False)

    def test_changed_input_with_rehashed_outer_packet_is_rejected(self) -> None:
        payload = build().model_dump(mode="json")
        payload["model_input"]["scenario"] = "changed after intake"
        with self.assertRaisesRegex(ValueError, "model input hash"):
            Web4BenchmarkIntakePacket.model_validate(reseal(payload))

    def test_changed_inner_input_hash_is_rejected(self) -> None:
        payload = build().model_dump(mode="json")
        payload["model_input_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "model input hash"):
            Web4BenchmarkIntakePacket.model_validate(reseal(payload))

    def test_rehashed_split_material_tampering_is_rejected(self) -> None:
        for field in ("case_id", "family", "split_seed", "source_snapshot_sha256s", "source_snapshot_receipt_sha256s"):
            with self.subTest(field=field):
                payload = build().model_dump(mode="json")
                if field.endswith("sha256s"):
                    source_ref = payload["source_refs"][0]
                    payload[field][source_ref] = "a" * 64
                else:
                    payload[field] += "-changed"
                with self.assertRaisesRegex(ValueError, "split material hash"):
                    Web4BenchmarkIntakePacket.model_validate(reseal(payload))

    def test_source_refs_cannot_be_empty_or_duplicated(self) -> None:
        for empty in (True, False):
            with self.subTest(empty=empty):
                payload = build().model_dump(mode="json")
                payload["source_refs"] = [] if empty else payload["source_refs"] * 2
                if empty:
                    for field in ("source_revision_status", "source_snapshot_sha256s",
                                  "source_snapshot_receipt_sha256s", "source_match_verified",
                                  "source_provenance_review_status"):
                        payload[field] = {}
                with self.assertRaises(ValueError):
                    Web4BenchmarkIntakePacket.model_validate(reseal(payload))

    def test_invalid_source_hashes_are_rejected(self) -> None:
        for field in ("source_snapshot_sha256s", "source_snapshot_receipt_sha256s"):
            for digest in ("invalid", "A" * 64, "a" * 63, "a" * 64 + "\n"):
                with self.subTest(field=field, digest=digest):
                    payload = build().model_dump(mode="json")
                    payload[field][payload["source_refs"][0]] = digest
                    with self.assertRaises(ValueError):
                        Web4BenchmarkIntakePacket.model_validate(reseal(payload))

    def test_empty_input_and_invalid_timestamp_are_rejected(self) -> None:
        for field, value in (("model_input", {}), ("created_at", "2026-09-08T00:00:00"), ("created_at", "invalid-date-value")):
            with self.subTest(field=field, value=value):
                payload = build().model_dump(mode="json")
                payload[field] = value
                with self.assertRaises(ValueError):
                    Web4BenchmarkIntakePacket.model_validate(reseal(payload))

    def test_model_input_cannot_hide_non_json_values(self) -> None:
        for value in (float("nan"), float("inf"), {"a", "b"}, ({"answer_key": "hidden"},)):
            with self.subTest(value=repr(value)):
                proposal = json.loads((FIXTURE / "proposal.json").read_text())
                proposal["model_input"]["extra"] = value
                with self.assertRaises(ValueError):
                    Web4BenchmarkCaseProposal.model_validate(proposal)

    def test_nested_answer_key_fields_are_rejected(self) -> None:
        proposal = json.loads((FIXTURE / "proposal.json").read_text())
        proposal["model_input"]["extra"] = [{"nested": {"Ground-Truth": "hidden"}}]
        with self.assertRaisesRegex(ValueError, "answer-key-like"):
            Web4BenchmarkCaseProposal.model_validate(proposal)

    def test_writer_revalidates_mutable_model_input(self) -> None:
        packet = build()
        packet.model_input["scenario"] = "changed after validation"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet.json"
            with self.assertRaises(ValueError):
                write_web4_benchmark_intake_packet(packet, output)
            self.assertFalse(output.exists())

    def test_writer_round_trip_preserves_valid_packet(self) -> None:
        packet = build()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet.json"
            write_web4_benchmark_intake_packet(packet, output)
            self.assertEqual(Web4BenchmarkIntakePacket.model_validate_json(output.read_text()), packet)
            with self.assertRaises(FileExistsError):
                write_web4_benchmark_intake_packet(packet, output)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        for raw in ('{"scope":1,"scope":2}', '{"nested":{"scope":1,"scope":2}}',
                    '{"scope":1,"sc\\u006fpe":2}', '{"x":NaN}', '{"x":Infinity}',
                    '{"x":-Infinity}', '{"x":1e999}', '{"x":-1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json_loads(raw)

    def test_strict_json_preserves_valid_values(self) -> None:
        value = {"nested": [None, True, False, 42, 1.25, "kanıt"]}
        self.assertEqual(strict_json_loads(json.dumps(value).encode()), value)

    def test_intake_rejects_ambiguous_proposal_json(self) -> None:
        for extra in ('"case_id":"ignored",', '"extra":NaN,', '"extra":{"key":1,"key":2},'):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "proposal.json"
                raw = (FIXTURE / "proposal.json").read_text()
                path.write_text("{" + extra + raw.lstrip()[1:])
                with self.assertRaisesRegex(ValueError, "invalid Web4 benchmark proposal JSON"):
                    build(proposal_path=path)

    def test_both_registry_readers_reject_duplicate_members(self) -> None:
        raw = SOURCES.read_text()
        first, *rest = raw.splitlines()
        duplicate = '{"training_authorization":true,' + first.lstrip()[1:]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sources.jsonl"
            path.write_text("\n".join([duplicate, *rest]) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid Web4 source row"):
                build(source_registry_path=path)
            with self.assertRaisesRegex(ValueError, "invalid Web4 source row"):
                build_web4_research_snapshot_receipt(
                    source_id="nist.ai-agent-identity-authority-2026",
                    snapshot_path=FIXTURE / "source-snapshot.txt",
                    captured_at="2026-09-08T00:00:00Z",
                    source_registry_path=path,
                )

    def test_receipt_reader_rejects_duplicate_members(self) -> None:
        raw = (FIXTURE / "source-snapshot.receipt.json").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text('{"source_match_verified":true,' + raw.lstrip()[1:])
            with self.assertRaisesRegex(ValueError, "invalid Web4 research snapshot receipt JSON"):
                verify_web4_research_snapshot_receipt(
                    receipt_path=path,
                    snapshot_path=FIXTURE / "source-snapshot.txt",
                    source_registry_path=SOURCES,
                )


if __name__ == "__main__":
    unittest.main()
