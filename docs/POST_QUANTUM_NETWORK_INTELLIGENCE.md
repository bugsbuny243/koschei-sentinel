# Post-Quantum Network Intelligence v1

## Purpose

Koschei Sentinel treats post-quantum migration as a security-intelligence problem, not as a news feed.
The lane records protocol decisions, cryptographic transition surfaces, migration dependencies, and
primary evidence in a form that can later support training and security-specific evaluation.

This lane does **not** authorize raw web discoveries for training. Discovery, evidence verification,
dataset admission, Gold review, and model promotion remain separate stages.

## Model identity boundary

The long-term Koschei Sentinel product target is a Sentinel-native sparse/MoE architecture at
approximately 397B total parameters and approximately 35B active parameters. The pinned
`Qwen/Qwen3.5-397B-A17B` target remains an external bootstrap/trainer-compatibility vehicle and is
not the identity of Koschei Sentinel.

The machine-readable target is:

`configs/model/sentinel-moe-397b-a35b.target.json`

No paid large-model run is authorized by this document.

## Intelligence representation

Each admitted PQ network record follows the Sentinel security graph:

`Entity -> Event -> Intent -> Capability -> Vulnerability -> Action -> Consequence -> Evidence`

In addition, the record carries a migration surface:

- accounts
- validators
- consensus
- wallets/HSMs
- bridges/cross-chain verification

The event schema is `schemas/pq-network-intelligence-v1.schema.json`.

## Fail-closed source flow

1. Discover a potentially relevant protocol publication, EIP/BIP/SIMD/AIP/ACP, client change,
   testnet event, wallet/HSM capability, or bridge compatibility change.
2. Resolve the canonical locator. Protocol-status claims require a primary protocol source.
3. Materialize an immutable source snapshot.
4. Compute SHA256 for the exact snapshot bytes.
5. Bind the snapshot to the repository provenance/receipt chain.
6. Extract structured claims with explicit uncertainty.
7. Human-review evidence and labels before Gold admission.
8. Keep HOLDOUT material isolated from training.
9. Run security-specific regression gates before any candidate promotion.

`configs/corpus/pq-network-watch.v1.json` intentionally sets all seed discoveries to
`training_authorization=false`. A title or secondary report may create a discovery lead, but it
cannot authorize a protocol fact, a target date, an EIP ranking, or a training row.

## Initial high-priority discoveries

The intake queue preserves two 2026-09-07 Ethereum Protocol Cluster discoveries as priority leads:

- `EF Protocol: Current and Emerging Priorities`
- `Hegotá EIP Opinion Post and Tier List`

Their canonical official locators and immutable source hashes must be resolved before Sentinel
extracts protocol claims from them. This prevents a search-engine snippet, repost, or model output
from becoming trusted training evidence.

An already-resolved official Ethereum Foundation discovery anchor,
`Protocol Priorities Update for 2026` (2026-02-18), is also registered, but remains unauthorized
until the repository snapshot/receipt process is completed.

## Network watch set

The first watch set is Ethereum, Bitcoin, Solana, Sui, NEAR, Cardano, Aptos, Avalanche, and Cosmos.
Membership in this list means only "monitor this network". It is not evidence that the network has
started, completed, or committed to a post-quantum migration.

## What Sentinel should learn

The lane should eventually support evaluation of whether a model can:

- distinguish research, proposal, test, deployment, rejection, and unknown states;
- identify which cryptographic surface is actually affected;
- reconstruct migration dependencies across accounts, validators, consensus, wallets, and bridges;
- resist converting missing evidence into a negative claim;
- separate a target date from a committed implementation deadline;
- ground every material conclusion in immutable evidence; and
- express confidence and uncertainty when attribution or protocol intent is incomplete.

## Promotion boundary

A model checkpoint does not gain authority because it predicts PQ migration correctly on a few
examples. Promotion still requires provenance-clean Gold/HOLDOUT evaluation, security regression
checks, checkpoint identity/hashes, and the existing Sentinel promotion gates.
