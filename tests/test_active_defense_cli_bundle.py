import json

from koschei_sentinel.active_defense_cli import main


def test_active_defense_cli_emits_plan_interception_and_execution(tmp_path, capsys):
    evidence_sha = "a" * 64
    graph = {
        "schema_version": "sentinel.cyber-state-graph.v1",
        "graph_id": "graph:cli-test",
        "entities": [
            {"entity_id": "identity:user", "entity_type": "IDENTITY", "labels": {}},
            {"entity_id": "device:host", "entity_type": "DEVICE", "labels": {}},
        ],
        "relations": [
            {
                "relation_id": "rel:auth",
                "source_entity_id": "identity:user",
                "target_entity_id": "device:host",
                "relation_type": "authenticates_to",
                "status": "OBSERVED",
                "confidence": 0.95,
                "evidence": [
                    {
                        "evidence_id": "evidence:auth",
                        "source": "edr",
                        "content_sha256": evidence_sha,
                        "timestamp": None,
                    }
                ],
                "rationale": "observed test authentication",
            }
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    assert main(["--graph", str(graph_path), "--critical", "device:host"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "sentinel.active-defense-bundle.v1"
    assert payload["graph_id"] == "graph:cli-test"
    assert "active_defense_plan" in payload
    assert "interception_plan" in payload
    assert "execution_state" in payload
    assert payload["execution_state"]["graph_id"] == "graph:cli-test"
