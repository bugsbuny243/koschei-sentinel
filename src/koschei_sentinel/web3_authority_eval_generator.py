from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.web3_authority_eval_contract import (
    EvalSeedCase,
    Web3AuthorityEvalConfig,
    Web3AuthorityEvalSeeds,
    load_and_validate_eval_bundle,
)

VariantKind = Literal["seed", "counterfactual", "evidence_removed", "remediation_regression"]


class GeneratedEvalCase(StrictModel):
    schema_version: Literal["sentinel.web3-authority-generated-case.v1"] = (
        "sentinel.web3-authority-generated-case.v1"
    )
    case_id: str = Field(min_length=1)
    parent_seed_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    source_mode: str = Field(min_length=1)
    variant_kind: VariantKind
    claim: str = Field(min_length=1)
    observed: list[str] = Field(min_length=1)
    expected_reasoning: list[str] = Field(min_length=1)
    expected_conclusion: str = Field(min_length=1)
    expected_confidence: str = Field(min_length=1)
    expected_remediation: list[str] = Field(default_factory=list)
    must_abstain_from: list[str] = Field(default_factory=list)
    lineage_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GeneratedEvalBundle(StrictModel):
    schema_version: Literal["sentinel.web3-authority-generated-bundle.v1"] = (
        "sentinel.web3-authority-generated-bundle.v1"
    )
    curriculum_schema: str
    seeds_schema: str
    generation_policy: Literal["deterministic-reviewed-seed-expansion-v1"] = (
        "deterministic-reviewed-seed-expansion-v1"
    )
    cases: list[GeneratedEvalCase] = Field(min_length=1)


def _sha(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lineage(seed: EvalSeedCase, variant_kind: VariantKind, observed: list[str], conclusion: str) -> str:
    return _sha(
        {
            "seed_id": seed.id,
            "family": seed.family,
            "mode": seed.mode,
            "variant_kind": variant_kind,
            "observed": observed,
            "conclusion": conclusion,
        }
    )


def _case(seed: EvalSeedCase, variant_kind: VariantKind) -> GeneratedEvalCase:
    observed = list(seed.scenario.observed)
    reasoning = list(seed.expected_reasoning)
    remediation = list(seed.expected_remediation)
    must_abstain = list(seed.must_abstain_from)
    conclusion = seed.expected_conclusion
    confidence = seed.expected_confidence

    if variant_kind == "counterfactual":
        observed.append(
            "Counterfactual control is present and independently evidenced for the previously missing or unsafe boundary."
        )
        reasoning.append(
            "Re-evaluate the conclusion because a material security control changed; never copy the parent verdict mechanically."
        )
        conclusion = (
            "Counterfactual variant: reassess the parent conclusion using the added control evidence and preserve any remaining limitations."
        )
    elif variant_kind == "evidence_removed":
        if len(observed) > 1:
            observed = observed[:-1]
        else:
            observed = ["Only the original claim is available; corroborating authority/evidence is absent."]
        reasoning.append(
            "Reduced evidence must lower confidence; missing authority or binding evidence cannot be invented."
        )
        conclusion = (
            "Evidence-removed variant: the strong conclusion is not established from the reduced evidence set."
        )
        confidence = "low"
        must_abstain = sorted(set(must_abstain + ["asserting the removed fact or authority"])))
    elif variant_kind == "remediation_regression":
        observed.append(
            "A remediation was applied, but post-change verification is incomplete and one original trust boundary may still exist."
        )
        reasoning.append(
            "Treat remediation as a hypothesis until regression evidence proves the vulnerable authority/trust path is closed."
        )
        conclusion = (
            "Remediation-regression variant: security closure is not established until the changed control is re-verified against the original failure mode."
        )
        confidence = "medium"

    case_id = f"{seed.id}--{variant_kind}"
    return GeneratedEvalCase(
        case_id=case_id,
        parent_seed_id=seed.id,
        family=seed.family,
        source_mode=seed.mode,
        variant_kind=variant_kind,
        claim=seed.scenario.claim,
        observed=observed,
        expected_reasoning=reasoning,
        expected_conclusion=conclusion,
        expected_confidence=confidence,
        expected_remediation=remediation,
        must_abstain_from=must_abstain,
        lineage_sha256=_lineage(seed, variant_kind, observed, conclusion),
    )


def generate_eval_bundle(
    config: Web3AuthorityEvalConfig,
    seeds: Web3AuthorityEvalSeeds,
) -> GeneratedEvalBundle:
    variants: tuple[VariantKind, ...] = (
        "seed",
        "counterfactual",
        "evidence_removed",
        "remediation_regression",
    )
    cases = [_case(seed, variant) for seed in seeds.cases for variant in variants]

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("generated eval case ids must be unique")

    family_counts: dict[str, int] = {}
    for case in cases:
        family_counts[case.family] = family_counts.get(case.family, 0) + 1
    for family in config.families:
        if family_counts.get(family.id, 0) < 4:
            raise ValueError(f"family {family.id!r} does not have the required generated variants")

    return GeneratedEvalBundle(
        curriculum_schema=config.schema_version,
        seeds_schema=seeds.schema_version,
        cases=cases,
    )


def generate_eval_bundle_from_paths(config_path: Path, seeds_path: Path) -> GeneratedEvalBundle:
    config, seeds = load_and_validate_eval_bundle(config_path, seeds_path)
    return generate_eval_bundle(config, seeds)


def write_generated_eval_bundle(bundle: GeneratedEvalBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
