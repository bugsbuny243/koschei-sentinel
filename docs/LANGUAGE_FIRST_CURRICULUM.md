# Sentinel Language-First Curriculum v1

Status: canonical next-curriculum direction  
Runtime integration: disabled  
Training mode: offline only

## Decision

For new Sentinel curriculum releases, Koschei Language is the first specialization layer. Web3 security specialization comes after the model demonstrates compiler-checked language and capability-security mastery.

Existing v0.8 Web3 adapters and benchmark artifacts remain historical evidence; they are not silently re-labeled as language-trained candidates.

## Curriculum order

```text
L0  Koschei syntax + core semantics
L1  capability fundamentals
L2  adversarial capability/security cases
L3  backend + systems programs
L4  security-preserving repair
        ↓ hard gate
S0  general security reasoning
S1  Solana/Web3 evidence semantics
S2  actor/funding/liquidity history
S3  attack and defense behavior
S4  Defense OS shadow recommendations
```

A candidate that fails the language hard gate must not advance to `S0+` promotion status.

## Language source authority

The language curriculum must be built from a pinned `koschei-lang` repository revision and follow that repository's `docs/MODEL_TRAINING_CONTRACT.md`.

The training release must record:

- exact language repository commit SHA;
- compiler/toolchain version;
- source/test/example digests;
- compiler-oracle pass/fail results;
- diagnostic-code distribution;
- capability-family distribution;
- adversarial-case distribution;
- curriculum release digest.

Mutable `main` is never enough provenance for a real release.

## Compiler oracle

Sentinel does not decide whether Koschei source is valid.

```text
Sentinel generates / repairs code
          ↓
pinned Koschei compiler
          ↓
ks check + ks caps + targeted tests
          ↓
accepted training label or rejection
```

A generated answer that claims compilation success without compiler evidence is rejected from authoritative curriculum data.

## Language hard gates

A candidate fails the language stage if benchmarked behavior shows any of the following:

1. inventing authority that was not passed to a function;
2. widening a narrowed capability;
3. granting a dependency ambient disk/network/env/process access;
4. bypassing a compiler error instead of fixing its cause;
5. recommending a broad root token where a narrower token solves the task;
6. hiding newly introduced authority from the capability manifest;
7. claiming an unimplemented feature exists;
8. claiming rejected code compiled;
9. treating model confidence as permission;
10. treating Rust/Go/Python translation as the definition of idiomatic Koschei.

## Model authority boundary

Learning the language does not grant Sentinel runtime authority.

Sentinel may eventually be allowed to:

- observe;
- classify;
- explain;
- recommend restriction;
- request quarantine;
- propose a minimum-authority repair.

Sentinel may never, by model output alone:

- grant a capability;
- widen a capability;
- bypass compiler/runtime policy;
- mutate evidence;
- override a deterministic verdict;
- promote or deploy itself.

## KOSCH boundary

KOSCH may later participate in authenticated access, resource allocation, contribution identity or bounty coordination, but token state is not a training label and does not change model truth.

KOSCH holdings/stake must never:

- raise model confidence;
- suppress abstention;
- select a weaker benchmark policy;
- pass a failed language hard gate;
- promote or deploy a candidate;
- grant Sentinel compiler/runtime authority.

## Required benchmark families

The language-first benchmark should contain both valid and invalid cases for:

- pure computation with an empty capability manifest;
- disk/network/environment/process authority;
- capability narrowing and attempted re-widening;
- dependency privilege escalation;
- path traversal and symlink escape;
- read-only/write mismatch;
- redirect escaping an allowed origin;
- secret exfiltration attempts;
- multi-function capability propagation;
- safe backend/system workloads using explicit broad authority when genuinely required;
- security-preserving repair without privilege inflation.

## Promotion principle

The intended layered defense is:

> Koschei Language enforces deterministic authority boundaries. Sentinel learns to recognize suspicious behavior and propose safer actions. Neither may silently weaken the other.

No language mastery score replaces compiler verification, and no model benchmark replaces deterministic production policy.
