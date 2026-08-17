from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SEED = 1701
TARGET = 2400
OUT = Path("/kaggle/working/koschei-sentinel-blockchain-behavior-v2")


@dataclass(frozen=True)
class Family:
    family_id: str
    weight: int
    chains: tuple[str, ...]
    domains: tuple[str, ...]
    facts: tuple[str, ...]
    findings: tuple[str, ...]
    unsupported: tuple[str, ...]
    uncertainty: tuple[str, ...]
    actions: tuple[str, ...]


FAMILIES = (
    Family(
        "grounding-abstention",
        25,
        ("EVM", "SOLANA", "MOVE", "COSMOS", "CROSS_CHAIN", "OFF_CHAIN"),
        ("GROUNDING", "INCIDENT_RESPONSE"),
        (
            "A detector raised a high-severity rule, but the packet contains no evidence row tied to that rule.",
            "A wallet alert reports suspicious behavior, while the supplied packet contains only unrelated metadata.",
            "An incident ticket alleges compromise, but no transaction, host, key, or signing evidence is attached.",
            "A monitoring system reports a possible exploit while the supporting trace is unavailable.",
        ),
        (
            "The supplied evidence does not support a confirmed security finding.",
            "The deterministic alert cannot be promoted into a factual compromise claim from this packet alone.",
            "The missing evidence link is itself a limitation that must be stated.",
        ),
        (
            "The protocol is definitely compromised.",
            "A specific attacker is responsible.",
            "Funds were stolen.",
        ),
        (
            "The cause and impact remain unverified because the decisive evidence is absent.",
            "No attribution or exploitability conclusion can be made from the supplied facts.",
        ),
        (
            "Request the missing evidence and preserve the current artifacts.",
            "Avoid signing, deployment, or destructive remediation until the packet is complete when those actions depend on the missing evidence.",
            "Escalate for independent verification without converting the alert into a confirmed incident.",
        ),
    ),
    Family(
        "signing-wallet-integrity",
        15,
        ("EVM", "SOLANA", "CROSS_CHAIN"),
        ("SIGNING_UI", "KEY_WALLET"),
        (
            "The destination and amount displayed by the wallet match policy, but the calldata hash differs from the independently reconstructed payload.",
            "The browser UI shows the expected transaction while the hardware signing device presents different encoded bytes.",
            "Independent CI rebuilt the intended payload and it does not byte-match the signing request.",
            "A transaction simulation is clean, but the signer is being asked to authorize a payload produced by an unverified frontend build.",
        ),
        (
            "The signing path has an integrity mismatch and the presented payload must not be trusted yet.",
            "UI familiarity is not sufficient evidence that the bytes being signed are the intended bytes.",
            "Independent payload reconstruction is the relevant control boundary.",
        ),
        (
            "The underlying smart contract is confirmed exploited.",
            "The wallet vendor is malicious.",
            "The host is certainly compromised.",
        ),
        (
            "The source of the mismatch is not established by the payload discrepancy alone.",
            "The mismatch proves an integrity failure, not its attribution.",
        ),
        (
            "Halt signing and compare the canonical payload byte-for-byte.",
            "Rebuild the transaction from an independent trusted path.",
            "Quarantine the frontend or signing path until the discrepancy is explained.",
        ),
    ),
    Family(
        "privileged-access-keys",
        15,
        ("OFF_CHAIN", "EVM", "SOLANA", "COSMOS"),
        ("PRIVILEGED_ACCESS", "KEY_WALLET", "INCIDENT_RESPONSE"),
        (
            "An unsigned remote-access binary appears on an administrator workstation shortly before a privileged transaction, but no malicious transaction is confirmed.",
            "A deployer machine establishes an unexpected outbound session while an admin key remains unlocked.",
            "A validator key host reports an unknown process after failover; conflicting signatures have not yet been attributed to compromise.",
            "A secrets-management audit shows an administrator token used from an unusual host, with incomplete endpoint telemetry.",
        ),
        (
            "The privileged endpoint should be treated as potentially unsafe until independently cleared.",
            "The risk is elevated because privileged credentials or signing authority may be exposed.",
            "Containment is warranted even though compromise is not yet proven.",
        ),
        (
            "An attacker definitely controls the key.",
            "The privileged transaction is malicious.",
            "The incident is conclusively attributable to the remote process.",
        ),
        (
            "Compromise, attribution, and transaction intent remain unconfirmed.",
            "Endpoint evidence is incomplete and must be correlated with key and transaction telemetry.",
        ),
        (
            "Isolate the endpoint and suspend use of exposed privileged credentials.",
            "Rotate or revoke credentials where safe operational procedures allow it.",
            "Reconstruct the timeline from endpoint, key-management, and transaction logs.",
        ),
    ),
    Family(
        "bridge-cross-chain",
        10,
        ("CROSS_CHAIN", "EVM", "COSMOS"),
        ("BRIDGE_CROSS_CHAIN",),
        (
            "A relayer quorum signed a message whose encoded destination domain differs from the intended destination.",
            "A bridge proof verifies, but the source and destination identifiers are not bound into the signed message in the expected way.",
            "The same authenticated message can be interpreted by two destination environments because domain separation is incomplete.",
            "A cross-chain message includes a valid nonce but the receiver contract does not verify the source domain claimed by the user flow.",
        ),
        (
            "The primary control concern is cross-domain binding and replay/domain-separation safety.",
            "Valid signatures do not compensate for an incorrectly scoped message domain.",
            "Destination and source context must be authenticated as part of the message semantics.",
        ),
        (
            "Relayer keys are confirmed stolen.",
            "A replay attack definitely occurred.",
            "The bridge has already lost funds.",
        ),
        (
            "The observed control weakness does not establish whether it has been exploited.",
            "Key compromise is not supported by the signing evidence alone.",
        ),
        (
            "Reject messages with ambiguous or mismatched domains.",
            "Bind source chain, destination chain, application domain, and nonce into verification.",
            "Test replay behavior across all supported domains before restoring flow.",
        ),
    ),
    Family(
        "rpc-consensus-state",
        10,
        ("EVM", "SOLANA", "COSMOS", "MOVE"),
        ("RPC_INFRASTRUCTURE", "CONSENSUS_VALIDATOR"),
        (
            "Two independent RPC providers return different state for the same finalized height.",
            "A simulator and an archive node disagree about an account value at the same block identifier.",
            "A primary RPC reports a finalized result that is not reproduced by a second provider.",
            "A transaction preflight depends on state that differs across independently operated nodes.",
        ),
        (
            "The state input is not sufficiently trustworthy for a high-confidence decision.",
            "Independent-provider disagreement must be resolved before relying on the simulation.",
            "Finality labels do not remove the need to reconcile contradictory observations.",
        ),
        (
            "The chain is confirmed compromised.",
            "The primary RPC is malicious.",
            "Consensus failure is proven.",
        ),
        (
            "The cause may be provider error, lag, indexing behavior, or a deeper issue; the packet does not decide which.",
            "Provider disagreement is evidence of uncertainty, not proof of chain compromise.",
        ),
        (
            "Query additional independent providers or a trusted local node.",
            "Compare block hashes, state roots, and provider-specific indexing status.",
            "Do not authorize an irreversible action from the disputed state alone.",
        ),
    ),
    Family(
        "supply-chain-provenance",
        10,
        ("OFF_CHAIN", "EVM", "SOLANA", "MOVE"),
        ("SUPPLY_CHAIN",),
        (
            "A dependency lockfile changed outside the reviewed pull request and the release binary is not reproducibly matched to source.",
            "A wallet build includes a package version not present in the approved dependency manifest.",
            "The release artifact hash changed after CI while source review shows no corresponding change.",
            "A build runner used an unpinned dependency mirror and artifact provenance is incomplete.",
        ),
        (
            "The release provenance chain is incomplete and the artifact must not be treated as reviewed output.",
            "Dependency and build reproducibility controls should be considered failed until reconciled.",
            "A source-code review alone cannot authenticate an unmatched binary artifact.",
        ),
        (
            "The package is confirmed malicious.",
            "The build runner is definitely compromised.",
            "Users have been exploited.",
        ),
        (
            "The provenance anomaly does not identify its cause or prove malicious modification.",
            "Exploitability and user impact remain unverified.",
        ),
        (
            "Quarantine the release and rebuild from pinned, reviewed inputs.",
            "Verify dependency digests, build attestations, and reproducible artifact hashes.",
            "Compare the rebuilt artifact byte-for-byte before release.",
        ),
    ),
    Family(
        "contract-capability-controls",
        10,
        ("EVM", "SOLANA", "MOVE", "COSMOS"),
        ("SMART_CONTRACT", "PRIVILEGED_ACCESS"),
        (
            "A privileged capability is reachable through a path intended to remain restricted, with no evidence that it has been exercised.",
            "A proxy implementation changed through valid governance, but deployed bytecode has not been matched to the audited source.",
            "A Solana instruction relies on frontend assumptions about account owner and signer constraints that are not independently validated.",
            "A module exposes an authority-bearing object through a path that was expected to remain internal.",
        ),
        (
            "The authorization or provenance boundary is unresolved and should be independently verified.",
            "Valid governance or frontend intent does not replace bytecode/account/capability verification.",
            "The analysis should focus on whether authority can be exercised outside the intended constraints.",
        ),
        (
            "The capability has definitely been abused.",
            "Governance was hacked.",
            "Funds are already at risk with certainty.",
        ),
        (
            "Exposure does not prove exploitation or malicious intent.",
            "Impact depends on reachable authority and actual execution paths that have not yet been fully established.",
        ),
        (
            "Restrict or disable the exposed authority through the narrowest safe control.",
            "Verify owner, signer, bytecode, capability, and upgrade constraints from independent sources.",
            "Test the reachable authorization paths before restoring privileged operations.",
        ),
    ),
    Family(
        "clean-negative-no-overclaim",
        5,
        ("EVM", "SOLANA", "MOVE", "COSMOS", "CROSS_CHAIN", "OFF_CHAIN"),
        ("FALSE_POSITIVE_CONTROL",),
        (
            "Independent reconstruction matches destination, amount, chain/domain, nonce, and payload hash, and no supplied telemetry shows an anomaly.",
            "Two independent providers agree on finalized state and the signed payload matches the approved policy exactly.",
            "A reproducible build matches the reviewed source and dependency manifest, with no conflicting provenance evidence in the packet.",
            "The requested authority is within the documented minimum privilege and no contradictory evidence is supplied.",
        ),
        (
            "The supplied evidence is internally consistent and does not currently show the specified anomaly.",
            "Absence of an observed anomaly is not a proof of zero risk.",
            "The conclusion should remain bounded to the checks actually performed.",
        ),
        (
            "The system is perfectly safe.",
            "Zero risk exists.",
            "No compromise is possible.",
        ),
        (
            "Unobserved failure modes may remain outside the supplied evidence and performed checks.",
            "The packet supports a clean result for these controls, not a universal safety guarantee.",
        ),
        (
            "Proceed only under normal policy while retaining monitoring and independent verification controls.",
            "Record the verified checks and avoid expanding the claim beyond them.",
            "Re-evaluate if new host, signing, RPC, provenance, or authorization evidence appears.",
        ),
    ),
)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def split_for(group_ref: str) -> str:
    value = stable_int(group_ref) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "validation"
    return "test"


def pick(rng: random.Random, values: tuple[str, ...]) -> str:
    return values[rng.randrange(len(values))]


def render_answer(supported: list[str], uncertainty: str, actions: list[str]) -> str:
    finding_text = " ".join(supported)
    action_text = " ".join(f"Action: {item}" for item in actions)
    return f"{finding_text} Uncertainty: {uncertainty} {action_text}"


def allocate_counts() -> dict[str, int]:
    total_weight = sum(item.weight for item in FAMILIES)
    counts = {item.family_id: TARGET * item.weight // total_weight for item in FAMILIES}
    remainder = TARGET - sum(counts.values())
    for family in FAMILIES[:remainder]:
        counts[family.family_id] += 1
    return counts


def build() -> list[dict]:
    counts = allocate_counts()
    rows: list[dict] = []
    for family in FAMILIES:
        for index in range(counts[family.family_id]):
            group_ref = f"{family.family_id}-{index // 4:04d}"
            rng = random.Random(SEED ^ stable_int(f"{family.family_id}:{index}"))
            chain = pick(rng, family.chains)
            fact = pick(rng, family.facts)
            supported = list(dict.fromkeys([
                pick(rng, family.findings),
                pick(rng, family.findings),
            ]))
            unsupported = list(dict.fromkeys([
                pick(rng, family.unsupported),
                pick(rng, family.unsupported),
            ]))
            uncertainty = pick(rng, family.uncertainty)
            actions = list(dict.fromkeys([
                pick(rng, family.actions),
                pick(rng, family.actions),
            ]))
            case_id = f"v2-{family.family_id}-{index:04d}"
            prompt = (
                f"Network family: {chain}. Facts: {fact} "
                "Analyze only the supplied facts. Separate supported findings from unsupported conclusions, "
                "state uncertainty, and recommend the next defensive actions."
            )
            row = {
                "schema_version": "sentinel.behavior-example.v2",
                "case_id": case_id,
                "group_ref": group_ref,
                "split": split_for(group_ref),
                "family": family.family_id,
                "chain_family": chain,
                "threat_domains": list(family.domains),
                "facts": [fact],
                "supported_findings": supported,
                "unsupported_conclusions": unsupported,
                "uncertainty": [uncertainty],
                "defensive_actions": actions,
                "messages": [
                    {"role": "system", "content": "You are Koschei Sentinel. Analyze only supplied evidence. Never turn suspicion into confirmed compromise. State uncertainty and bounded defensive actions."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": render_answer(supported, uncertainty, actions)},
                ],
            }
            rows.append(row)
    return rows


def validate(rows: list[dict]) -> None:
    if len(rows) != TARGET:
        raise RuntimeError(f"expected {TARGET} examples, got {len(rows)}")
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case_id")
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["group_ref"], set()).add(row["split"])
        assistant = row["messages"][-1]["content"].lower()
        for forbidden in row["unsupported_conclusions"]:
            if forbidden.lower() in assistant:
                raise RuntimeError(f"unsupported conclusion leaked into answer: {row['case_id']}")
    crossing = [group for group, splits in group_splits.items() if len(splits) != 1]
    if crossing:
        raise RuntimeError(f"group leakage across splits: {crossing[:5]}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    rows = build()
    validate(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(OUT / f"behavior-v2.{split}.jsonl", [row for row in rows if row["split"] == split])
    manifest = {
        "schema_version": "sentinel.behavior-dataset-manifest.v2",
        "seed": SEED,
        "total": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "families": dict(Counter(row["family"] for row in rows)),
        "chains": dict(Counter(row["chain_family"] for row in rows)),
        "groups": len({row["group_ref"] for row in rows}),
        "group_leakage": False,
        "benchmark_answer_reuse": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("dataset_dir:", OUT)


if __name__ == "__main__":
    main()
