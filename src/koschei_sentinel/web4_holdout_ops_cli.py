from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import load_owner_private_key, load_owner_public_key
from koschei_sentinel.training import atomic_write
from koschei_sentinel.web4_benchmark_intake import Web4BenchmarkIntakePacket
from koschei_sentinel.web4_benchmark_review import (
    Web4BenchmarkAdjudication,
    Web4BenchmarkHumanReview,
)
from koschei_sentinel.web4_holdout_evaluation import (
    Web4HoldoutEvaluationEvidence,
    Web4HoldoutEvaluationPolicy,
    Web4HoldoutInferencePack,
    Web4HoldoutPredictionSet,
    build_web4_holdout_evaluation_evidence,
    build_web4_holdout_inference_pack,
    build_web4_holdout_prediction,
    build_web4_holdout_prediction_set,
)
from koschei_sentinel.web4_holdout_evaluation_decision import (
    Web4HoldoutEvaluationDecisionSpec,
    sign_web4_holdout_evaluation_decision,
)
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    Web4HoldoutReleaseMaterial,
    build_web4_holdout_release,
)
from koschei_sentinel.web4_reviewer_trust import (
    load_web4_review_public_key,
    load_web4_reviewer_trust_policy,
)

_DIGEST = r"^[a-f0-9]{64}$"


class Web4HoldoutReleaseMaterialPath(StrictModel):
    packet: str = Field(min_length=1, max_length=4096)
    review: str = Field(min_length=1, max_length=4096)
    adjudication: str = Field(min_length=1, max_length=4096)
    answer_key: str = Field(min_length=1, max_length=4096)
    reviewer_public_key: str = Field(min_length=1, max_length=4096)
    reviewer_trust_policy: str = Field(min_length=1, max_length=4096)
    adjudicator_public_key: str = Field(min_length=1, max_length=4096)
    adjudicator_trust_policy: str = Field(min_length=1, max_length=4096)


class Web4HoldoutReleaseMaterialManifest(StrictModel):
    schema_version: str
    cases: list[Web4HoldoutReleaseMaterialPath] = Field(min_length=1, max_length=100000)


class Web4OfflinePredictionInput(StrictModel):
    case_id: str = Field(min_length=3, max_length=256)
    answer: dict[str, object]


class Web4OfflinePredictionManifest(StrictModel):
    schema_version: str
    model_ref: str = Field(min_length=3, max_length=512)
    model_revision: str = Field(min_length=1, max_length=256)
    model_artifact_sha256: str = Field(pattern=_DIGEST)
    predictions: list[Web4OfflinePredictionInput]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-holdout-ops",
        description=(
            "Create signed Web4 research HOLDOUT, answer-key-free inference, offline "
            "prediction/evaluation, and owner-decision artifacts without network or GPU use."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    release = commands.add_parser("release")
    release.add_argument("--release-id", required=True)
    release.add_argument("--material-manifest", required=True)
    release.add_argument("--benchmark-policy", required=True)
    release.add_argument("--owner-private-key", required=True)
    release.add_argument("--output", required=True)

    pack = commands.add_parser("pack")
    pack.add_argument("--release", required=True)
    pack.add_argument("--benchmark-policy", required=True)
    pack.add_argument("--owner-public-key", required=True)
    pack.add_argument("--output", required=True)

    predictions = commands.add_parser("predictions")
    predictions.add_argument("--inference-pack", required=True)
    predictions.add_argument("--input", required=True)
    predictions.add_argument("--output", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--release", required=True)
    evaluate.add_argument("--inference-pack", required=True)
    evaluate.add_argument("--prediction-set", required=True)
    evaluate.add_argument("--answer-key-dir", required=True)
    evaluate.add_argument("--benchmark-policy", required=True)
    evaluate.add_argument("--owner-public-key", required=True)
    evaluate.add_argument("--evaluation-policy")
    evaluate.add_argument("--output", required=True)

    decide = commands.add_parser("decide")
    decide.add_argument("--evidence", required=True)
    decide.add_argument("--spec", required=True)
    decide.add_argument("--owner-private-key", required=True)
    decide.add_argument("--output", required=True)
    return parser


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate


def _regular_dir(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return candidate


def _preflight_output(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Web4 HOLDOUT operator output already exists: {destination}")
    current = destination.parent
    while True:
        if current.is_symlink():
            raise ValueError("Web4 HOLDOUT operator output path must not traverse symlinks")
        if current.parent == current:
            break
        current = current.parent
    return destination


def _load_model(path: str | Path, model_type, label: str):
    candidate = _regular_file(path, label)
    try:
        return model_type.model_validate_json(candidate.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc


def _write_model(artifact, path: Path) -> None:
    payload = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(path, payload)


def _manifest_relative_file(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be relative to the material manifest")
    base = root.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks")
    candidate = (root / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"{label} escapes the material manifest directory")
    return _regular_file(candidate, label)


def _load_release_materials(path: str | Path) -> list[Web4HoldoutReleaseMaterial]:
    manifest_path = _regular_file(path, "Web4 HOLDOUT material manifest")
    try:
        manifest = Web4HoldoutReleaseMaterialManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise ValueError("invalid Web4 HOLDOUT material manifest") from exc
    if manifest.schema_version != "sentinel.web4-holdout-release-material-manifest.v1":
        raise ValueError("unsupported Web4 HOLDOUT material manifest schema")

    root = manifest_path.resolve().parent
    materials: list[Web4HoldoutReleaseMaterial] = []
    for index, row in enumerate(manifest.cases):
        prefix = f"Web4 HOLDOUT material[{index}]"
        packet = _load_model(
            _manifest_relative_file(root, row.packet, f"{prefix} packet"),
            Web4BenchmarkIntakePacket,
            f"{prefix} packet",
        )
        review = _load_model(
            _manifest_relative_file(root, row.review, f"{prefix} review"),
            Web4BenchmarkHumanReview,
            f"{prefix} review",
        )
        adjudication = _load_model(
            _manifest_relative_file(root, row.adjudication, f"{prefix} adjudication"),
            Web4BenchmarkAdjudication,
            f"{prefix} adjudication",
        )
        answer_key = _manifest_relative_file(root, row.answer_key, f"{prefix} answer key")
        reviewer_public_key_path = _manifest_relative_file(
            root,
            row.reviewer_public_key,
            f"{prefix} reviewer public key",
        )
        reviewer_policy_path = _manifest_relative_file(
            root,
            row.reviewer_trust_policy,
            f"{prefix} reviewer trust policy",
        )
        adjudicator_public_key_path = _manifest_relative_file(
            root,
            row.adjudicator_public_key,
            f"{prefix} adjudicator public key",
        )
        adjudicator_policy_path = _manifest_relative_file(
            root,
            row.adjudicator_trust_policy,
            f"{prefix} adjudicator trust policy",
        )
        materials.append(
            Web4HoldoutReleaseMaterial(
                packet=packet,
                review=review,
                adjudication=adjudication,
                answer_key_path=answer_key,
                reviewer_public_key=load_web4_review_public_key(reviewer_public_key_path),
                reviewer_trust_policy=load_web4_reviewer_trust_policy(reviewer_policy_path),
                adjudicator_public_key=load_web4_review_public_key(adjudicator_public_key_path),
                adjudicator_trust_policy=load_web4_reviewer_trust_policy(
                    adjudicator_policy_path
                ),
            )
        )
    return materials


def _load_evaluation_policy(path: str | Path | None) -> Web4HoldoutEvaluationPolicy:
    if path is None:
        return Web4HoldoutEvaluationPolicy()
    return _load_model(
        path,
        Web4HoldoutEvaluationPolicy,
        "Web4 HOLDOUT evaluation policy",
    )


def _release(args: argparse.Namespace) -> dict[str, object]:
    output = _preflight_output(args.output)
    materials = _load_release_materials(args.material_manifest)
    benchmark_policy = _regular_file(args.benchmark_policy, "Web4 benchmark policy")
    owner_key_path = _regular_file(args.owner_private_key, "owner private key")
    release = build_web4_holdout_release(
        release_id=args.release_id,
        materials=materials,
        benchmark_policy_path=benchmark_policy,
        owner_private_key=load_owner_private_key(owner_key_path),
    )
    _write_model(release, output)
    return {
        "ok": True,
        "command": "release",
        "release_id": release.release_id,
        "release_sha256": release.release_sha256,
        "artifact_sha256": release.artifact_sha256,
        "case_count": release.case_count,
        "answer_keys_isolated": release.answer_keys_isolated,
        "research_evaluation_authorization": release.research_evaluation_authorization,
        "training_authorization": release.training_authorization,
        "promotion_eligible": release.promotion_eligible,
        "production_activation_allowed": release.production_activation_allowed,
        "output": str(output),
    }


def _pack(args: argparse.Namespace) -> dict[str, object]:
    output = _preflight_output(args.output)
    release = _load_model(args.release, Web4HoldoutRelease, "Web4 HOLDOUT release")
    benchmark_policy = _regular_file(args.benchmark_policy, "Web4 benchmark policy")
    owner_key_path = _regular_file(args.owner_public_key, "owner public key")
    pack = build_web4_holdout_inference_pack(
        release=release,
        owner_public_key=load_owner_public_key(owner_key_path),
        benchmark_policy_path=benchmark_policy,
    )
    _write_model(pack, output)
    return {
        "ok": True,
        "command": "pack",
        "release_id": pack.release_id,
        "release_sha256": pack.release_sha256,
        "pack_sha256": pack.pack_sha256,
        "case_count": pack.case_count,
        "answer_key_excluded": pack.answer_key_excluded,
        "network_access_required": pack.network_access_required,
        "gpu_required": pack.gpu_required,
        "training_authorization": pack.training_authorization,
        "output": str(output),
    }


def _predictions(args: argparse.Namespace) -> dict[str, object]:
    output = _preflight_output(args.output)
    pack = _load_model(
        args.inference_pack,
        Web4HoldoutInferencePack,
        "Web4 HOLDOUT inference pack",
    )
    prediction_manifest = _load_model(
        args.input,
        Web4OfflinePredictionManifest,
        "Web4 offline prediction manifest",
    )
    if prediction_manifest.schema_version != "sentinel.web4-offline-prediction-manifest.v1":
        raise ValueError("unsupported Web4 offline prediction manifest schema")
    if len({row.case_id for row in prediction_manifest.predictions}) != len(
        prediction_manifest.predictions
    ):
        raise ValueError("Web4 offline prediction manifest contains duplicate case IDs")
    pack_cases = {row.case_id: row for row in pack.cases}
    predictions = []
    for row in prediction_manifest.predictions:
        inference_case = pack_cases.get(row.case_id)
        if inference_case is None:
            raise ValueError(f"Web4 offline prediction case is absent from inference pack: {row.case_id}")
        predictions.append(
            build_web4_holdout_prediction(
                inference_case=inference_case,
                release_sha256=pack.release_sha256,
                model_ref=prediction_manifest.model_ref,
                model_revision=prediction_manifest.model_revision,
                model_artifact_sha256=prediction_manifest.model_artifact_sha256,
                answer=row.answer,
            )
        )
    prediction_set = build_web4_holdout_prediction_set(
        inference_pack=pack,
        predictions=predictions,
        model_ref=prediction_manifest.model_ref,
        model_revision=prediction_manifest.model_revision,
        model_artifact_sha256=prediction_manifest.model_artifact_sha256,
    )
    _write_model(prediction_set, output)
    return {
        "ok": True,
        "command": "predictions",
        "release_sha256": prediction_set.release_sha256,
        "inference_pack_sha256": prediction_set.inference_pack_sha256,
        "prediction_set_sha256": prediction_set.prediction_set_sha256,
        "prediction_count": prediction_set.prediction_count,
        "model_ref": prediction_set.model_ref,
        "model_revision": prediction_set.model_revision,
        "model_artifact_sha256": prediction_set.model_artifact_sha256,
        "offline_replay": prediction_set.offline_replay,
        "network_access_required": prediction_set.network_access_required,
        "gpu_required": prediction_set.gpu_required,
        "output": str(output),
    }


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    output = _preflight_output(args.output)
    release = _load_model(args.release, Web4HoldoutRelease, "Web4 HOLDOUT release")
    pack = _load_model(
        args.inference_pack,
        Web4HoldoutInferencePack,
        "Web4 HOLDOUT inference pack",
    )
    prediction_set = _load_model(
        args.prediction_set,
        Web4HoldoutPredictionSet,
        "Web4 HOLDOUT prediction set",
    )
    answer_key_dir = _regular_dir(args.answer_key_dir, "Web4 HOLDOUT answer-key directory")
    benchmark_policy = _regular_file(args.benchmark_policy, "Web4 benchmark policy")
    owner_key_path = _regular_file(args.owner_public_key, "owner public key")
    policy = _load_evaluation_policy(args.evaluation_policy)
    evidence = build_web4_holdout_evaluation_evidence(
        release=release,
        inference_pack=pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_key_dir,
        owner_public_key=load_owner_public_key(owner_key_path),
        benchmark_policy_path=benchmark_policy,
        policy=policy,
    )
    _write_model(evidence, output)
    return {
        "ok": True,
        "command": "evaluate",
        "release_id": evidence.release_id,
        "evidence_sha256": evidence.evidence_sha256,
        "prediction_set_sha256": evidence.prediction_set_sha256,
        "answer_key_bundle_sha256": evidence.answer_key_bundle_sha256,
        "case_count": evidence.case_count,
        "prediction_count": evidence.prediction_count,
        "complete_case_accounting": evidence.complete_case_accounting,
        "answer_key_values_embedded": evidence.answer_key_values_embedded,
        "offline_replay": evidence.offline_replay,
        "network_access_required": evidence.network_access_required,
        "gpu_required": evidence.gpu_required,
        "passed": evidence.passed,
        "training_authorization": evidence.training_authorization,
        "promotion_eligible": evidence.promotion_eligible,
        "production_activation_allowed": evidence.production_activation_allowed,
        "output": str(output),
    }


def _decide(args: argparse.Namespace) -> dict[str, object]:
    output = _preflight_output(args.output)
    evidence = _load_model(
        args.evidence,
        Web4HoldoutEvaluationEvidence,
        "Web4 HOLDOUT evaluation evidence",
    )
    spec = _load_model(
        args.spec,
        Web4HoldoutEvaluationDecisionSpec,
        "Web4 HOLDOUT evaluation decision spec",
    )
    owner_key_path = _regular_file(args.owner_private_key, "owner private key")
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=spec,
        owner_private_key=load_owner_private_key(owner_key_path),
    )
    _write_model(decision, output)
    return {
        "ok": True,
        "command": "decide",
        "decision_id": decision.decision_id,
        "decision": decision.decision.value,
        "decision_sha256": decision.decision_sha256,
        "artifact_sha256": decision.artifact_sha256,
        "evidence_sha256": decision.evidence_sha256,
        "research_evaluation_evidence_accepted": decision.research_evaluation_evidence_accepted,
        "research_comparison_eligible": decision.research_comparison_eligible,
        "model_execution_authorized": decision.model_execution_authorized,
        "training_authorization": decision.training_authorization,
        "promotion_eligible": decision.promotion_eligible,
        "production_activation_allowed": decision.production_activation_allowed,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "release":
            result = _release(args)
        elif args.command == "pack":
            result = _pack(args)
        elif args.command == "predictions":
            result = _predictions(args)
        elif args.command == "evaluate":
            result = _evaluate(args)
        else:
            result = _decide(args)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-web4-holdout-ops: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
