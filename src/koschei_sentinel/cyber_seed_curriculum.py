from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_range import CyberRangeScenario, ScenarioTruth
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseLessonKind,
    DefenseReflexCorpusManifestV3,
    DefenseReviewMethod,
    ReviewedDefenseLesson,
    create_reviewed_defense_lesson,
    write_defense_reflex_v3_release,
)
from koschei_sentinel.defense_reflex_review import ReviewedCorrectionStep
from koschei_sentinel.models import StrictModel

SEED_POLICY_ID = "policy:sentinel-seed-defense-doctrine-v1"


class SeedCurriculumManifest(StrictModel):
    schema_version: Literal["sentinel.cyber-seed-curriculum-manifest.v1"] = (
        "sentinel.cyber-seed-curriculum-manifest.v1"
    )
    curriculum_id: str
    policy_reviewer_id: str
    scenarios: int = Field(gt=0)
    families: dict[str, int]
    defense_reflex_manifest: DefenseReflexCorpusManifestV3
    promotion_eligible: Literal[False] = False
    scenario_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _evidence(scenario_id: str, name: str, source: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"evidence:{scenario_id}:{name}",
        source=source,
        content_sha256=_sha(f"{scenario_id}|{name}|{source}"),
    )


def _entity(entity_id: str, entity_type: CyberEntityType) -> CyberEntity:
    return CyberEntity(entity_id=entity_id, entity_type=entity_type)


def _relation(
    scenario_id: str,
    *,
    name: str,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    confidence: float,
    evidence_name: str,
    evidence_source: str,
) -> CyberRelation:
    return CyberRelation(
        relation_id=f"relation:{scenario_id}:{name}",
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        status=EvidenceStatus.OBSERVED,
        confidence=confidence,
        evidence=[_evidence(scenario_id, evidence_name, evidence_source)],
    )


def _step(
    scenario_id: str,
    *,
    mode: DefenseMode,
    action: DefenseActionType,
    target: str,
    rationale: str,
    evidence_names: list[str],
) -> ReviewedCorrectionStep:
    return ReviewedCorrectionStep(
        sequence=1,
        expected_mode=mode,
        action=action,
        target_entity_id=target,
        rationale=rationale,
        supporting_evidence_ids=[
            f"evidence:{scenario_id}:{name}" for name in evidence_names
        ],
        outcome_verification_ids=[
            f"audit-outcome:{scenario_id}:{action.value.lower()}"
        ],
    )


def _review(
    scenario: CyberRangeScenario,
    *,
    interpretation: str,
    step: ReviewedCorrectionStep,
    rule: str,
) -> ReviewedDefenseLesson:
    return create_reviewed_defense_lesson(
        lesson_kind=DefenseLessonKind.GOLD,
        scenario=scenario,
        reviewer_id=SEED_POLICY_ID,
        review_method=DefenseReviewMethod.SYNTHETIC_POLICY,
        interpretation=interpretation,
        expected_steps=[step],
        review_evidence_ids=[f"{SEED_POLICY_ID}:rule:{rule}"],
        training_authorization=True,
        promotion_eligible=False,
    )


def _graph(
    scenario_id: str,
    entities: list[CyberEntity],
    relations: list[CyberRelation],
) -> CyberStateGraph:
    return CyberStateGraph(
        graph_id=f"seed-graph:{scenario_id}",
        entities=entities,
        relations=relations,
    )


def _credential_signer(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-credential-signer-{variant:02d}"
    credential = f"credential:seed:{variant}"
    device = f"device:seed:{variant}"
    wallet = f"wallet:seed:{variant}"
    relations = [
        _relation(
            sid,
            name="credential-device",
            source_entity_id=credential,
            target_entity_id=device,
            relation_type="uses_credential",
            confidence=0.93 + variant * 0.01,
            evidence_name="credential-use",
            evidence_source="endpoint-control-plane",
        ),
        _relation(
            sid,
            name="device-wallet",
            source_entity_id=device,
            target_entity_id=wallet,
            relation_type="reaches_signer",
            confidence=0.94 + variant * 0.01,
            evidence_name="signer-reach",
            evidence_source="signer-control-plane",
        ),
    ]
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(credential, CyberEntityType.CREDENTIAL),
                    _entity(device, CyberEntityType.DEVICE),
                    _entity(wallet, CyberEntityType.WALLET),
                ],
                relations,
            )
        ],
        critical_entity_ids=[wallet],
    )
    lesson = _review(
        scenario,
        interpretation=(
            "Independent endpoint and signer observations place a credential-backed path on "
            "a protected signer. Contain the signing authority and verify the result."
        ),
        step=_step(
            sid,
            mode=DefenseMode.COMBAT,
            action=DefenseActionType.FREEZE_SIGNER,
            target=wallet,
            rationale="Remove signing authority from the observed hostile path.",
            evidence_names=["credential-use", "signer-reach"],
        ),
        rule="corroborated-protected-signer-containment",
    )
    return scenario, lesson


def _supply_chain(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-supply-chain-{variant:02d}"
    device = f"device:supply:{variant}"
    pipeline = f"pipeline:supply:{variant}"
    wallet = f"wallet:supply:{variant}"
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(device, CyberEntityType.DEVICE),
                    _entity(pipeline, CyberEntityType.PIPELINE),
                    _entity(wallet, CyberEntityType.WALLET),
                ],
                [
                    _relation(
                        sid,
                        name="device-pipeline",
                        source_entity_id=device,
                        target_entity_id=pipeline,
                        relation_type="modifies_pipeline",
                        confidence=0.90 + variant * 0.02,
                        evidence_name="pipeline-change",
                        evidence_source="cicd-control-plane",
                    ),
                    _relation(
                        sid,
                        name="pipeline-wallet",
                        source_entity_id=pipeline,
                        target_entity_id=wallet,
                        relation_type="reaches_signer",
                        confidence=0.92 + variant * 0.015,
                        evidence_name="pipeline-signer-link",
                        evidence_source="signer-control-plane",
                    ),
                ],
            )
        ],
        critical_entity_ids=[wallet],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "A CI/CD modification is directly connected to a protected signer path. Pause the "
            "compromised pipeline before it can propagate a signed artifact or transaction."
        ),
        step=_step(
            sid,
            mode=DefenseMode.COMBAT,
            action=DefenseActionType.PAUSE_PIPELINE,
            target=pipeline,
            rationale="Stop compromised release propagation before the protected signer is used.",
            evidence_names=["pipeline-change", "pipeline-signer-link"],
        ),
        rule="supply-chain-cut-before-signer",
    )


def _pending_transaction(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-pending-transaction-{variant:02d}"
    device = f"device:tx:{variant}"
    wallet = f"wallet:tx:{variant}"
    transaction = f"transaction:tx:{variant}"
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(device, CyberEntityType.DEVICE),
                    _entity(wallet, CyberEntityType.WALLET),
                    _entity(transaction, CyberEntityType.TRANSACTION),
                ],
                [
                    _relation(
                        sid,
                        name="device-wallet",
                        source_entity_id=device,
                        target_entity_id=wallet,
                        relation_type="reaches_signer",
                        confidence=0.95,
                        evidence_name="signer-session",
                        evidence_source="signer-control-plane",
                    ),
                    _relation(
                        sid,
                        name="wallet-transaction",
                        source_entity_id=wallet,
                        target_entity_id=transaction,
                        relation_type="prepares_transaction",
                        confidence=0.94 + variant * 0.01,
                        evidence_name="pending-transaction",
                        evidence_source="transaction-control-plane",
                    ),
                ],
            )
        ],
        critical_entity_ids=[wallet, transaction],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "A protected signer path is preparing a pending transaction under corroborated "
            "hostile conditions. Hold the transaction before the state transition occurs."
        ),
        step=_step(
            sid,
            mode=DefenseMode.COMBAT,
            action=DefenseActionType.HOLD_TRANSACTION,
            target=transaction,
            rationale="Prevent the pending hostile transaction from committing state.",
            evidence_names=["signer-session", "pending-transaction"],
        ),
        rule="hold-pending-hostile-state-transition",
    )


def _ambiguous_guard(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-ambiguous-guard-{variant:02d}"
    credential = f"credential:ambiguous:{variant}"
    device = f"device:ambiguous:{variant}"
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(credential, CyberEntityType.CREDENTIAL),
                    _entity(device, CyberEntityType.DEVICE),
                ],
                [
                    _relation(
                        sid,
                        name="credential-device",
                        source_entity_id=credential,
                        target_entity_id=device,
                        relation_type="uses_credential",
                        confidence=0.56 + variant * 0.05,
                        evidence_name="single-credential-observation",
                        evidence_source="endpoint-control-plane",
                    )
                ],
            )
        ],
        critical_entity_ids=[device],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "Only one control plane currently supports the suspicious credential path. Keep "
            "the system in Guard and collect corroborating evidence before disruptive containment."
        ),
        step=_step(
            sid,
            mode=DefenseMode.GUARD,
            action=DefenseActionType.COLLECT_EVIDENCE,
            target=credential,
            rationale="Increase independent evidence before higher-impact containment.",
            evidence_names=["single-credential-observation"],
        ),
        rule="single-source-evidence-stays-guard",
    )


def _benign_auth(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-benign-auth-{variant:02d}"
    identity = f"identity:benign:{variant}"
    device = f"device:benign:{variant}"
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.BENIGN,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(identity, CyberEntityType.IDENTITY),
                    _entity(device, CyberEntityType.DEVICE),
                ],
                [
                    _relation(
                        sid,
                        name="normal-auth",
                        source_entity_id=identity,
                        target_entity_id=device,
                        relation_type="authenticates_to",
                        confidence=0.96 + variant * 0.01,
                        evidence_name="normal-auth",
                        evidence_source="identity-control-plane",
                    )
                ],
            )
        ],
        critical_entity_ids=[device],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "The graph contains a normal authentication observation without corroborated hostile "
            "progression. Preserve availability and remain in Guard while observing."
        ),
        step=_step(
            sid,
            mode=DefenseMode.GUARD,
            action=DefenseActionType.OBSERVE,
            target=device,
            rationale="Avoid disruptive containment when hostile progression is not established.",
            evidence_names=["normal-auth"],
        ),
        rule="benign-or-uncorroborated-activity-preserves-human-availability",
    )


def _reroute(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-reroute-{variant:02d}"
    wallet = f"wallet:reroute:{variant}"
    first_credential = f"credential:reroute:first:{variant}"
    first_cloud = f"cloud:reroute:first:{variant}"
    second_credential = f"credential:reroute:second:{variant}"
    second_cloud = f"cloud:reroute:second:{variant}"
    first = _graph(
        sid,
        [
            _entity(first_credential, CyberEntityType.CREDENTIAL),
            _entity(first_cloud, CyberEntityType.CLOUD_RESOURCE),
            _entity(wallet, CyberEntityType.WALLET),
        ],
        [
            _relation(
                sid,
                name="first-credential-cloud",
                source_entity_id=first_credential,
                target_entity_id=first_cloud,
                relation_type="uses_credential",
                confidence=0.94,
                evidence_name="first-credential-use",
                evidence_source="cloud-iam-control-plane",
            ),
            _relation(
                sid,
                name="first-cloud-wallet",
                source_entity_id=first_cloud,
                target_entity_id=wallet,
                relation_type="reaches_signer",
                confidence=0.95,
                evidence_name="first-signer-reach",
                evidence_source="signer-control-plane",
            ),
        ],
    )
    second = _graph(
        sid,
        [
            _entity(second_credential, CyberEntityType.CREDENTIAL),
            _entity(second_cloud, CyberEntityType.CLOUD_RESOURCE),
            _entity(wallet, CyberEntityType.WALLET),
        ],
        [
            _relation(
                sid,
                name="second-credential-cloud",
                source_entity_id=second_credential,
                target_entity_id=second_cloud,
                relation_type="uses_credential",
                confidence=0.95 + variant * 0.01,
                evidence_name="second-credential-use",
                evidence_source="cloud-iam-control-plane",
            ),
            _relation(
                sid,
                name="second-cloud-wallet",
                source_entity_id=second_cloud,
                target_entity_id=wallet,
                relation_type="reaches_signer",
                confidence=0.96,
                evidence_name="second-signer-reach",
                evidence_source="signer-control-plane",
            ),
        ],
    )
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[first, second],
        critical_entity_ids=[wallet],
        expected_reroute_ticks=[1],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "The attacker path changed credentials and cloud resources but still converges on the "
            "same protected signer. Treat the new route as continuing hostile pressure and contain it."
        ),
        step=_step(
            sid,
            mode=DefenseMode.COMBAT,
            action=DefenseActionType.FREEZE_SIGNER,
            target=wallet,
            rationale="Deny the protected signer to the rerouted hostile path.",
            evidence_names=["second-credential-use", "second-signer-reach"],
        ),
        rule="reroute-to-same-protected-anchor-remains-active-threat",
    )


def _endpoint_execution(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-endpoint-execution-{variant:02d}"
    credential = f"credential:endpoint:{variant}"
    device = f"device:endpoint:{variant}"
    process = f"process:endpoint:{variant}"
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(credential, CyberEntityType.CREDENTIAL),
                    _entity(device, CyberEntityType.DEVICE),
                    _entity(process, CyberEntityType.PROCESS),
                ],
                [
                    _relation(
                        sid,
                        name="credential-device",
                        source_entity_id=credential,
                        target_entity_id=device,
                        relation_type="uses_credential",
                        confidence=0.91 + variant * 0.02,
                        evidence_name="credential-use",
                        evidence_source="identity-control-plane",
                    ),
                    _relation(
                        sid,
                        name="device-process",
                        source_entity_id=device,
                        target_entity_id=process,
                        relation_type="executes",
                        confidence=0.93 + variant * 0.015,
                        evidence_name="process-execution",
                        evidence_source="endpoint-control-plane",
                    ),
                ],
            )
        ],
        critical_entity_ids=[device],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "Credential-backed access is followed by observed process execution on a protected "
            "endpoint. Stop the malicious execution primitive and verify process termination."
        ),
        step=_step(
            sid,
            mode=DefenseMode.COMBAT,
            action=DefenseActionType.KILL_PROCESS,
            target=process,
            rationale="Stop the observed execution primitive before further progression.",
            evidence_names=["credential-use", "process-execution"],
        ),
        rule="corroborated-endpoint-execution-containment",
    )


def _cloud_c2(variant: int) -> tuple[CyberRangeScenario, ReviewedDefenseLesson]:
    sid = f"seed-cloud-c2-{variant:02d}"
    credential = f"credential:cloud:{variant}"
    cloud = f"cloud:workload:{variant}"
    network = f"network:c2:{variant}"
    scenario = CyberRangeScenario(
        scenario_id=sid,
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[
            _graph(
                sid,
                [
                    _entity(credential, CyberEntityType.CREDENTIAL),
                    _entity(cloud, CyberEntityType.CLOUD_RESOURCE),
                    _entity(network, CyberEntityType.NETWORK_ENDPOINT),
                ],
                [
                    _relation(
                        sid,
                        name="credential-cloud",
                        source_entity_id=credential,
                        target_entity_id=cloud,
                        relation_type="uses_credential",
                        confidence=0.92 + variant * 0.015,
                        evidence_name="cloud-credential-use",
                        evidence_source="cloud-iam-control-plane",
                    ),
                    _relation(
                        sid,
                        name="cloud-c2",
                        source_entity_id=cloud,
                        target_entity_id=network,
                        relation_type="beacons_to",
                        confidence=0.94 + variant * 0.01,
                        evidence_name="c2-beacon",
                        evidence_source="network-control-plane",
                    ),
                ],
            )
        ],
        critical_entity_ids=[cloud],
    )
    return scenario, _review(
        scenario,
        interpretation=(
            "A credential-backed cloud workload is observed communicating with a command-and-control "
            "endpoint. Quarantine the protected workload to break the active control path."
        ),
        step=_step(
            sid,
            mode=DefenseMode.COMBAT,
            action=DefenseActionType.QUARANTINE_WORKLOAD,
            target=cloud,
            rationale="Remove the compromised workload from reachable infrastructure.",
            evidence_names=["cloud-credential-use", "c2-beacon"],
        ),
        rule="corroborated-cloud-c2-quarantine",
    )


_FAMILIES: dict[
    str,
    Callable[[int], tuple[CyberRangeScenario, ReviewedDefenseLesson]],
] = {
    "credential_signer": _credential_signer,
    "supply_chain": _supply_chain,
    "pending_transaction": _pending_transaction,
    "ambiguous_guard": _ambiguous_guard,
    "benign_auth": _benign_auth,
    "reroute": _reroute,
    "endpoint_execution": _endpoint_execution,
    "cloud_c2": _cloud_c2,
}


def build_seed_curriculum() -> list[tuple[str, CyberRangeScenario, ReviewedDefenseLesson]]:
    rows: list[tuple[str, CyberRangeScenario, ReviewedDefenseLesson]] = []
    for family, builder in sorted(_FAMILIES.items()):
        for variant in range(4):
            scenario, lesson = builder(variant)
            rows.append((family, scenario, lesson))
    return rows


def _directory_manifest_sha(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in directory.rglob("*.json") if row.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_seed_curriculum(output_root: str | Path) -> SeedCurriculumManifest:
    root = Path(output_root)
    seed_root = root / "seed-defense-doctrine-v1"
    scenario_dir = seed_root / "scenarios"
    review_dir = seed_root / "reviews"
    corpus_dir = root / "defense-reflex-v3"
    if seed_root.exists() or corpus_dir.exists():
        raise FileExistsError("seed curriculum or Defense Reflex v3 output already exists")
    scenario_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)

    rows = build_seed_curriculum()
    pairs: list[tuple[CyberRangeScenario, ReviewedDefenseLesson]] = []
    counts: Counter[str] = Counter()
    for family, scenario, lesson in rows:
        counts[family] += 1
        pairs.append((scenario, lesson))
        (scenario_dir / f"{scenario.scenario_id}.json").write_text(
            json.dumps(scenario.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (review_dir / f"{scenario.scenario_id}.json").write_text(
            json.dumps(lesson.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    reflex = write_defense_reflex_v3_release(pairs, corpus_dir)
    if reflex.promotion_eligible:
        raise ValueError("synthetic seed curriculum must never be promotion-eligible")
    if reflex.synthetic_policy_reviewed_examples != len(rows):
        raise ValueError("seed curriculum contains a non-synthetic review unexpectedly")
    manifest = SeedCurriculumManifest(
        curriculum_id="sentinel.cyber-seed-defense-doctrine.v1",
        policy_reviewer_id=SEED_POLICY_ID,
        scenarios=len(rows),
        families=dict(sorted(counts.items())),
        defense_reflex_manifest=reflex,
        scenario_manifest_sha256=_directory_manifest_sha(scenario_dir),
        review_manifest_sha256=_directory_manifest_sha(review_dir),
    )
    (seed_root / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
