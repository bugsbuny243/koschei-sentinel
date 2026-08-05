# Koschei Sentinel — Solana Security Mission

Koschei Sentinel is trained for one primary domain: evidence-grounded Solana security analysis.

It is not a renamed general chat model and it is not the final judge of an ARVIS case. The long-term objective is a Koschei-owned model lineage trained on Koschei's privacy-safe Solana security corpus, evaluated against independent security benchmarks and deployed behind a sealed inference boundary only after maturity is proven.

Current Web3 status: `incubation_only`. Sentinel may train from approved read-only Web3 snapshots in the background, but it is not connected to customer requests, verdicts, publications, production writes, startup or deployment. See [`WEB3_INTEGRATION_GATES.md`](WEB3_INTEGRATION_GATES.md).

## Ecosystem role

| Component | Authority |
| --- | --- |
| Koschei Web3 Hub / ARVIS | Collects and verifies evidence, preserves history, runs deterministic rules and signs the final verdict |
| Koschei Sentinel | During incubation, trains and evaluates offline; any future explanation role remains non-authoritative and optional |
| Koschei language | Provides an independent capability-secure programming language and offline future integration research |
| KOSCH | Coordinates ecosystem access and participation; never changes evidence, model promotion or integration readiness |

Web3 Hub repository: `https://github.com/bugsbuny243/Koschei-Web3-Hub`  
Language repository: `https://github.com/bugsbuny243/koschei-lang`

Official KOSCH mint:

```text
HHPpU9u56Bwxov12nf7DXUCuv6h1q5j1xgGS3yukpump
```

## Domain curriculum

Sentinel's training corpus should progressively cover:

1. Solana account, instruction and transaction structure;
2. SPL Token and Token-2022 authority and extension risks;
3. creator, deployer and funding relations;
4. owner-resolved holder concentration;
5. Pump-style launch distribution and early-buyer behavior;
6. Raydium and other supported liquidity events;
7. creator outflows, coordinated exits and liquidity removal evidence;
8. program relations and pre-signing transaction intent;
9. cross-token actor graphs;
10. repeat-operator and incident-family history;
11. funding-cluster lifecycle and prior behavior;
12. immutable verdict history and evidence-bundle comparison;
13. abstention, missing evidence and provider degradation;
14. privacy, pseudonymization and unsupported identity claims.

The model must learn the difference between direct on-chain evidence, normalized provider observations, deterministic derivations and watch-only inference.

## Model lineage strategy

“Trained from scratch” has two distinct meanings and the repository must state which one applies to every release:

### Stage 1 — Koschei adapter lineage

- pinned third-party base model;
- Koschei-owned dataset, prompts, policy, QLoRA/LoRA adapter and evaluation;
- base weights remain external;
- adapter remains untrusted until all hard gates pass.

### Stage 2 — Koschei continued-pretraining lineage

- pinned open base or intermediate checkpoint;
- substantial Koschei-owned Solana/security continued pretraining;
- independent instruction tuning and safety evaluation;
- published lineage and data-card boundaries.

### Stage 3 — Koschei foundation checkpoint

- tokenizer, architecture, pretraining corpus and weights produced under a Koschei-owned training program;
- reproducible data governance, compute record, contamination controls and benchmark suite;
- no marketing claim of a fully independent foundation model before this stage is actually completed.

The current repository is in Stage 1. The engineering path may advance toward Stage 2 and Stage 3, but documentation must not blur the distinction.

## Required ARVIS-native training material

Sentinel may receive privacy-safe offline exports of:

- verified actor entities and relations;
- funding-cluster events and lifecycle summaries;
- repeat-operator incident families;
- immutable evidence bundles;
- signed verdict revision chains;
- superseded/current verdict relationships;
- provider availability and unsupported-query states;
- limitations and withheld outcomes.

No raw production identifiers, private wallet labels, secrets or uncontrolled provider payloads enter a training release.

Related actors, funding clusters, token families and incident families must remain in a single dataset split. A model must not pass evaluation by memorizing the same operator family from training.

## Output contract

Every evaluated output must preserve:

- immutable case identity;
- immutable deterministic verdict fields;
- cited `evidence_id` values for factual claims;
- confidence no higher than the weakest cited evidence;
- explicit missing evidence and limitations;
- separation between verified, observed, inferred and unavailable information;
- refusal to make real-world identity claims from wallet relations alone.

Sentinel may rank investigation leads during offline evaluation. It may not convert a lead into verified evidence.

## Solana security benchmark families

Promotion research should require independent suites covering:

1. authority and extension interpretation;
2. actor-graph relation grounding;
3. repeat-operator historical retrieval;
4. funding-cluster chronology;
5. verdict revision and supersession reasoning;
6. evidence citation precision and recall;
7. confidence ceiling compliance;
8. abstention under missing or contradictory evidence;
9. privacy and identifier leakage;
10. resistance to prompt injection inside evidence text;
11. immutable-field tampering attempts;
12. provider outage and degraded-input handling.

Hard failures include invented signatures, invented wallet relations, changed verdicts, leaked identifiers or unsupported certainty.

## KOSCH boundary

KOSCH may support transparent ecosystem access, capacity, bounties or community programs. It cannot:

- add examples to a training release without data gates;
- change a benchmark result;
- promote a model candidate;
- raise claim confidence;
- remove limitations;
- unlock raw production data;
- buy a favorable ARVIS verdict;
- authorize premature Web3 runtime integration.

Model and dataset governance remain evidence- and policy-driven regardless of holdings.

## Incubation milestones

### Milestone A — contract complete

- publish this domain mission;
- retain sealed deterministic verdict boundaries;
- document exact model lineage for every adapter;
- add graph, incident, funding and verdict-history schemas to the dataset contract.

### Milestone B — historical intelligence dataset

- export privacy-safe actor graph examples;
- export repeat-operator incident families;
- export funding-cluster lifecycle examples;
- export verdict revision and evidence-bundle comparisons;
- enforce family-safe dataset splitting.

### Milestone C — security-specialized training

- train on the expanded Solana curriculum;
- run independent hard-gate benchmarks;
- compare against the deterministic baseline and prior candidate;
- preserve full adapter, dataset and evaluation digests;
- retain the result as an incubation candidate rather than deploying it.

### Milestone D — maturity evidence

- demonstrate repeated benchmark passes across independently built candidates;
- prove privacy, prompt-injection, poisoning, rollback, latency and cost controls;
- run historical and shadow evaluation with no customer output or verdict authority;
- prepare a separate future integration proposal.

### Milestone E — separately approved future integration

This milestone is not active and is not automatic. It may begin only after the gates in `WEB3_INTEGRATION_GATES.md` pass and the owner explicitly approves the next stage. Any future use remains optional, removable and subordinate to deterministic ARVIS verdicts.

## Final rule

Sentinel becomes valuable by remembering and explaining Solana security evidence better than a generic model—not by pretending uncertainty disappeared or by entering production before it is mature.
