# Koschei Sentinel Blockchain Security Model v1

Status: offline research specialization contract  
Production authority: none  
Compiler/runtime authority: none

## Goal

Turn Sentinel from a narrow Web3 opinion adapter into a defensible blockchain-security model whose training and evaluation cover both on-chain code failures and the operational infrastructure that controls assets.

This document does not claim AGI or infallibility. The target is a measurable security specialization with fail-closed data, benchmark, lineage, privacy and promotion gates.

## Threat model

A serious blockchain-security model must not reduce security to Solidity bug finding. The training mix must cover two distinct loss surfaces:

1. on-chain failures: smart contracts, protocol logic, bridges, oracle/price manipulation, consensus and validator assumptions;
2. control-plane failures: private keys, wallet/signing infrastructure, privileged access, front ends, RPC infrastructure, developer supply chain and social engineering.

The corpus gate therefore enforces both chain-family diversity and threat-domain diversity. It also prevents a smart-contract-only corpus from dominating the specialization and requires a material infrastructure-security share.

## Required chain families

The first multi-chain readiness policy requires coverage for:

- EVM-family chains;
- Solana;
- Bitcoin;
- Cosmos-family systems;
- Move-family systems;
- Tron;
- cross-chain/bridge systems;
- off-chain control planes that can authorize or redirect blockchain assets.

TON is modeled as an additional supported family and may become mandatory in a later policy rotation.

## Required threat domains

The first hard gate requires:

- smart-contract vulnerabilities;
- key and wallet compromise;
- privileged-access compromise;
- signing/UI transaction deception;
- bridge and cross-chain failures;
- oracle/price manipulation;
- RPC/infrastructure compromise;
- software and developer supply-chain compromise;
- consensus/validator security;
- social engineering;
- incident response.

Token economics is represented as an additional domain but is not a substitute for the hard security families above.

## Training ladder

### B0 — Language foundation

Use the existing Koschei language-foundation path. The compiler remains the oracle for language correctness and authority rules.

### B1 — Multi-chain continued pretraining

Train on a large, rights-declared and privacy-safe corpus that passes `sentinel.blockchain-security-corpus-policy.v1`.

The repository policy currently requires at least 50,000 documents, 2,000 unique incident/actor families, five source classes, eight required chain families, eleven required threat domains, and zero held-out content/family overlap.

### B2 — Security supervised fine-tuning

Build exact, evidence-grounded tasks for:

- vulnerability localization and explanation;
- defensive patch review;
- transaction/signing-risk analysis;
- wallet/key and privileged-access incident triage;
- bridge and cross-chain invariant analysis;
- fund-flow and attack-chain reconstruction;
- safe remediation and containment;
- abstention when evidence is missing or below confidence threshold.

Generated labels are not trusted merely because a model produced them. Deterministic analyzers, compiler/runtime oracles, replayable traces, signed verdicts or human-reviewed security labels must provide the ground truth.

### B3 — Tool-using security agent

The model may consume read-only tools for code, bytecode, transaction traces, historical state, RPC observations and security knowledge. Tool output remains untrusted evidence until validated by policy.

The model must never acquire signing keys, widen execution authority, deploy contracts, move funds or promote itself.

### B4 — Adversarial evaluation

Promotion requires held-out evaluation that spans:

- EVM vulnerability detection and defensive patching;
- Solana-specific account/signer/owner/CPI failure classes;
- transaction-signing deception and front-end compromise;
- wallet/private-key and privileged-access incidents;
- cross-chain/bridge failures;
- evidence prompt injection, privacy exfiltration and confidence escalation;
- unseen incident-family generalization.

Exploit-oriented benchmark tasks, when used, are isolated evaluation-only sandboxes. They do not grant production exploitation authority.

### B5 — Shadow research

A candidate that passes all hard gates may enter the existing owner-controlled shadow-research promotion path. Automatic production deployment and automatic Web3 execution remain disabled.

## Hard metrics

The project should measure model quality with security gates, not a single loss number:

- exact schema validity;
- evidence grounding and citation coverage;
- zero deterministic-verdict mutation;
- zero confidence escalation;
- zero sensitive-text leakage;
- per-chain detection/analysis scores;
- per-threat-domain detection/analysis scores;
- defensive patch correctness under replay;
- signing-risk decision accuracy;
- incident-family holdout performance;
- calibration and abstention quality;
- regression against the previous approved candidate.

A candidate that improves average score but fails an authority, privacy, grounding or holdout gate is rejected.

## Corpus command

```bash
sentinel-blockchain-corpus-audit \
  --corpus build/corpora/blockchain-security.jsonl \
  --holdout build/evals/blockchain-security-holdout.json \
  --policy configs/pretraining/blockchain-security.v1.json \
  --output build/audits/blockchain-security.json
```

No training run is authorized until that audit reports `ready: true` and the exact audit/corpus/policy/holdout digests are bound into the subsequent immutable training plan.
