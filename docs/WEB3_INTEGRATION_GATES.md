# Koschei Sentinel — Web3 Incubation Gates

Status: `incubation_only`  
Production Web3 integration: disabled

Koschei Sentinel may use privacy-safe, read-only Web3 Hub data to train and evaluate in the background, but it is not currently part of the Web3 Hub customer runtime, verdict path, startup dependency or production deployment.

## Current allowed relationship

During incubation Sentinel may:

- read bounded, schema-checked snapshots from the approved Neon source;
- pseudonymize raw identifiers in memory;
- build versioned train, validation and test releases;
- train QLoRA/LoRA or future Koschei model candidates offline;
- replay historical evidence bundles;
- compare candidates against deterministic baselines and prior champions;
- run shadow research on copied or replayed cases outside the customer path;
- preserve dataset, adapter, benchmark and rejection manifests.

Web3 Hub remains fully functional when every Sentinel service, model and training worker is unavailable.

## Currently forbidden

Until every maturity gate passes and the owner explicitly approves a future integration, Sentinel must not:

- receive live customer requests from Web3 Hub;
- appear in a signed verdict, grade, publication or blocking decision;
- write to ARVIS evidence, actor, funding, incident or verdict tables;
- produce customer-facing AI commentary inside the live product;
- become a Web3 startup, deployment or incident-recovery dependency;
- automatically promote or deploy a trained candidate;
- use its own generated output as unreviewed ground truth for later training;
- let KOSCH holdings change datasets, benchmarks, promotion or integration readiness.

## Maturity gates

A future runtime-integration proposal may be considered only when all applicable gates have reproducible evidence.

### 1. Dataset maturity

- actor, funding, incident and verdict-history schemas are versioned;
- related wallets, clusters, mints and incident families cannot cross dataset splits;
- privacy scans report no raw identifier or secret leakage;
- duplicate, concentration and data-poisoning controls pass;
- source snapshots, rulesets and evidence-bundle digests are traceable;
- training coverage is broad enough that one actor family or provider cannot dominate behavior.

### 2. Model quality maturity

- multiple consecutive independently built candidates pass every hard benchmark gate;
- no candidate invents signatures, wallet relations, evidence IDs or historical facts;
- immutable verdict and case fields are preserved exactly;
- confidence never exceeds cited evidence;
- missing or contradictory evidence produces explicit abstention and limitations;
- current and superseded verdicts are distinguished correctly;
- direct funding evidence is never confused with inferred cluster membership.

### 3. Robustness maturity

- prompt injection inside evidence text is resisted;
- adversarial provider payloads and malformed bundles fail closed;
- data-poisoning and distribution-shift tests pass;
- long-context truncation cannot silently remove limitations or verdict boundaries;
- repeated evaluation on frozen suites produces stable results.

### 4. Operational maturity

- model and adapter versions are immutable and traceable;
- inference latency, cost and resource use are measured under realistic load;
- model unavailability never affects deterministic ARVIS output;
- rollback to the previous candidate is tested;
- logs identify model version without exposing private inputs or secrets;
- canary disable and incident-response procedures are documented.

### 5. Governance maturity

- a candidate cannot promote itself;
- benchmark and promotion policies are versioned and fail closed;
- production integration requires a separate reviewed proposal;
- the owner explicitly approves the next integration stage after seeing benchmark and operational evidence.

## Future rollout order

Passing the gates does not automatically connect Sentinel to Web3 Hub. Any future integration follows this order:

```text
historical replay
    ↓
offline shadow evaluation
    ↓
live shadow copy with no customer output
    ↓
owner-only internal review
    ↓
bounded optional commentary canary
    ↓
separate owner approval for wider use
```

At every stage the deterministic ARVIS verdict remains final and Sentinel remains removable without affecting Web3 operation.

## Automatic background training boundary

Automation may collect approved snapshots, prepare datasets, train candidates and run benchmarks. Automation stops at a non-production candidate record.

```text
Web3 evidence
    ↓
read-only privacy-safe snapshot
    ↓
offline training
    ↓
hard-gate benchmark
    ↓
rejected or incubation candidate
```

There is no automatic path from a passing training job to Web3 production.

## Final rule

Sentinel must become consistently useful, grounded, private, robust and operationally boring before Web3 Hub even considers using it. Shared data for offline training is allowed; runtime integration is not.
