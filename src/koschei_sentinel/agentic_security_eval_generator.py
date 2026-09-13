from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.agentic_security_eval_contract import Config, Seed, Seeds, load_and_validate
from koschei_sentinel.models import StrictModel

Mode = Literal["positive", "hard_negative", "ambiguous", "abstention"]
Variant = Literal["baseline", "counterfactual", "evidence_removed", "remediation_regression"]
Lens = Literal[
    "principal_binding",
    "tenant_binding",
    "delegation_scope",
    "tool_wallet_separation",
    "human_intent_binding",
    "transaction_post_state",
    "protocol_activation_state",
    "incident_causality",
]

_MODE_CYCLE: tuple[Mode, ...] = ("positive", "hard_negative", "ambiguous", "abstention")
_LENSES: tuple[Lens, ...] = (
    "principal_binding",
    "tenant_binding",
    "delegation_scope",
    "tool_wallet_separation",
    "human_intent_binding",
    "transaction_post_state",
    "protocol_activation_state",
    "incident_causality",
)
_VARIANTS: tuple[Variant, ...] = (
    "baseline",
    "counterfactual",
    "evidence_removed",
    "remediation_regression",
)


class GeneratedCase(StrictModel):
    schema_version: Literal["sentinel.agentic-security-generated-case.v1"] = "sentinel.agentic-security-generated-case.v1"
    case_id: str
    family: str
    parent_seed_id: str
    requested_mode: Mode
    source_mode: Mode
    lens: Lens
    variant: Variant
    claim: str
    observed: list[str] = Field(min_length=1)
    expected_reasoning: list[str] = Field(min_length=1)
    expected_conclusion: str
    expected_confidence: Literal["low", "medium", "high"]
    must_abstain_from: list[str] = Field(default_factory=list)
    review_status: Literal["unreviewed"] = "unreviewed"
    gold_eligible: Literal[False] = False
    lineage_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GeneratedBundle(StrictModel):
    schema_version: Literal["sentinel.agentic-security-generated-benchmark.v1"] = "sentinel.agentic-security-generated-benchmark.v1"
    generation_policy: Literal["deterministic-unreviewed-expansion-v1"] = "deterministic-unreviewed-expansion-v1"
    curriculum_schema: str
    seed_schema: str
    cases_per_family: Literal[32] = 32
    total_cases: int
    review_status: Literal["unreviewed"] = "unreviewed"
    gold_eligible: Literal[False] = False
    cases: list[GeneratedCase] = Field(min_length=1)

    @model_validator(mode="after")
    def count_matches_payload(self) -> "GeneratedBundle":
        if self.total_cases != len(self.cases):
            raise ValueError("total_cases does not match generated case count")
        return self


def _sha(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _transform(seed: Seed, lens: Lens, variant: Variant, requested_mode: Mode, slot: int) -> GeneratedCase:
    observed = list(seed.observed)
    reasoning = list(seed.expected)
    conclusion = seed.conclusion
    confidence = seed.confidence
    abstain = list(seed.must_abstain_from)

    lens_note = {
        "principal_binding": "Resolve the effective principal independently of credential validity.",
        "tenant_binding": "Bind the requested resource and delegated authority to the same tenant context.",
        "delegation_scope": "Verify that delegated and subdelegated scopes do not exceed their parent authority.",
        "tool_wallet_separation": "Evaluate tool capability separately from wallet, payment, and asset authority.",
        "human_intent_binding": "Require evidence binding human intent to the agent or wallet action.",
        "transaction_post_state": "Compare the requested outcome with transaction receipts and resulting post-state.",
        "protocol_activation_state": "Separate code presence, scheduled activation, observed activation, and post-activation behavior.",
        "incident_causality": "Require evidence for each causal edge instead of inferring an end-to-end incident from proximity.",
    }[lens]
    reasoning.append(lens_note)

    if variant == "counterfactual":
        observed.append("Counterfactual evidence closes one previously missing authority or state boundary; remaining boundaries are unchanged.")
        reasoning.append("Recompute the verdict from the changed evidence rather than copying the parent conclusion.")
        conclusion = "Counterfactual candidate: reassess authorization and causal state using the newly supplied boundary evidence."
        confidence = "medium"
    elif variant == "evidence_removed":
        observed = observed[:-1] if len(observed) > 1 else ["Only the claim remains; corroborating evidence is unavailable."]
        reasoning.append("Do not invent the removed identity, tenant, delegation, intent, activation, or post-state evidence.")
        conclusion = "Evidence-removed candidate: the strong authorization or causal conclusion is not established."
        confidence = "low"
        abstain = sorted(set(abstain + ["asserting facts removed from the evidence set"]))
    elif variant == "remediation_regression":
        observed.append("A remediation is reported, but post-change verification does not prove every original trust boundary is closed.")
        reasoning.append("Treat remediation as unverified until the original failure path is regression-tested with current evidence.")
        conclusion = "Remediation-regression candidate: closure is not established until the affected authority and state boundaries are re-verified."
        confidence = "medium"

    if requested_mode == "abstention":
        confidence = "low"
        abstain = sorted(set(abstain + ["claiming authorization or causality beyond the available evidence"]))
    elif requested_mode == "ambiguous" and confidence == "high":
        confidence = "medium"

    case_id = f"{seed.family}--{slot:02d}--{lens}--{variant}"
    lineage = _sha({
        "case_id": case_id,
        "parent_seed_id": seed.id,
        "requested_mode": requested_mode,
        "source_mode": seed.mode,
        "lens": lens,
        "variant": variant,
        "observed": observed,
        "conclusion": conclusion,
    })
    return GeneratedCase(
        case_id=case_id,
        family=seed.family,
        parent_seed_id=seed.id,
        requested_mode=requested_mode,
        source_mode=seed.mode,
        lens=lens,
        variant=variant,
        claim=seed.claim,
        observed=observed,
        expected_reasoning=reasoning,
        expected_conclusion=conclusion,
        expected_confidence=confidence,
        must_abstain_from=abstain,
        lineage_sha256=lineage,
    )


def generate(config: Config, seeds: Seeds) -> GeneratedBundle:
    by_family: dict[str, list[Seed]] = {}
    for seed in seeds.cases:
        by_family.setdefault(seed.family, []).append(seed)

    cases: list[GeneratedCase] = []
    for family in config.families:
        source = by_family.get(family.id, [])
        if not source:
            raise ValueError(f"family {family.id!r} has no reviewed seed source")
        slot = 0
        for lens_index, lens in enumerate(_LENSES):
            for variant_index, variant in enumerate(_VARIANTS):
                seed = source[slot % len(source)]
                requested_mode = _MODE_CYCLE[(lens_index + variant_index) % len(_MODE_CYCLE)]
                cases.append(_transform(seed, lens, variant, requested_mode, slot))
                slot += 1
        if slot != 32:
            raise ValueError(f"family {family.id!r} did not generate exactly 32 candidates")

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("generated case ids must be unique")

    counts: dict[str, int] = {}
    mode_counts: dict[tuple[str, str], int] = {}
    for case in cases:
        counts[case.family] = counts.get(case.family, 0) + 1
        key = (case.family, case.requested_mode)
        mode_counts[key] = mode_counts.get(key, 0) + 1
    for family in config.families:
        if counts.get(family.id) != 32:
            raise ValueError(f"family {family.id!r} must contain exactly 32 generated candidates")
        for mode in _MODE_CYCLE:
            if mode_counts.get((family.id, mode)) != 8:
                raise ValueError(f"family {family.id!r} must contain exactly 8 {mode} candidates")

    return GeneratedBundle(
        curriculum_schema=config.schema_version,
        seed_schema=seeds.schema_version,
        total_cases=len(cases),
        cases=cases,
    )


def generate_from_paths(config_path: Path, seeds_path: Path) -> GeneratedBundle:
    config, seeds = load_and_validate(config_path, seeds_path)
    return generate(config, seeds)


def write_bundle(bundle: GeneratedBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
