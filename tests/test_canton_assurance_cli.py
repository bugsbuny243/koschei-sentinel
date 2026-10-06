import json
import sys

from koschei_sentinel import canton_assurance_cli


def test_cli_emits_machine_readable_evidence(monkeypatch, capsys, tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "source_class": "public_test_fixture",
                "subject_ref": "canton-test-app",
                "artifact_version": "fixture-v1",
                "provenance": "public/reproducible-test-fixture",
                "integrity": "fixture-integrity-ok",
                "expected_evidence": True,
                "observed_evidence": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["canton-assurance", str(fixture)])

    assert canton_assurance_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["finding"]["status"] == "pass"
    assert output["finding"]["evidence_ids"] == [output["evidence"]["evidence_id"]]
