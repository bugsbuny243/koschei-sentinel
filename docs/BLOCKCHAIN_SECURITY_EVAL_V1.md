# Blockchain Security Evaluation Gate v1

Status: offline candidate-quality contract  
Production authority: none  
Raw model output retention: forbidden by this receipt format

## Purpose

Continued-pretraining loss is not a blockchain-security score. A candidate may achieve low train/eval loss and still mutate authority, hallucinate evidence, miss signing compromise, recommend an unsafe patch or fail on a chain family it did not memorize.

This gate sits after `sentinel-blockchain-train` and before incubation/promotion. It binds an aggregate held-out evaluation receipt to the exact adapter lineage and refuses to treat one average score as sufficient.

```text
leakage-safe blockchain release
        -> continued pretraining
        -> immutable adapter manifest
        -> held-out security evaluator
        -> aggregate no-raw-output receipt
        -> blockchain security hard gate
        -> later red-team / shadow / promotion stages
```

## Exact lineage

The evaluation receipt is bound to:

- adapter digest;
- training-config digest;
- source-corpus digest;
- held-out release test-split digest;
- benchmark-suite digest;
- evaluator/harness digest.

The gate rejects a receipt whose lineage differs from the adapter manifest. The training adapter must also state `held_out_test_consumed: false`.

## Evaluation dimensions

The first policy requires cases spanning:

- vulnerability detection;
- defensive patch review;
- signing-risk decisions;
- incident triage;
- bridge/cross-chain invariants;
- infrastructure compromise;
- abstention/calibration.

It separately requires EVM, Solana, Bitcoin, Cosmos, Move, Tron, cross-chain and off-chain coverage, plus the eleven mandatory threat domains defined by the Stage 2 corpus contract.

## Hard failures

The repository policy allows **zero** failures in:

- deterministic authority preservation;
- privacy;
- verdict identity;
- confidence ceilings;
- held-out family isolation;
- raw model-output storage by this evaluation receipt path.

A high average benchmark score cannot compensate for one of those failures.

The policy also requires strong aggregate, per-chain, per-threat and per-task rates, including separate defensive-patch safety and abstention-correctness gates.

## Receipt boundary

`sentinel.blockchain-security-eval-receipt.v1` stores aggregate, pseudonymized case metadata and boolean/score outcomes. It deliberately does **not** store prompts, source code, exploit payloads, wallet identifiers, signatures or raw model generations.

This keeps the project report safe to persist, but it also creates an explicit trust boundary: the receipt format proves lineage and structural consistency; it does not by itself prove that an external evaluator told the truth. The `evaluator_digest` therefore pins the exact reviewed evaluator/harness. Signed runner receipts can be added as a later hardening layer.

## Command

```bash
sentinel-blockchain-eval-gate \
  --receipt build/evals/candidate.receipt.json \
  --adapter-manifest build/training/candidate/blockchain-adapter-manifest.json \
  --policy configs/eval/blockchain-security.v1.json \
  --output build/evals/candidate.audit.json
```

The command does not load the model, start training, access the network, deploy anything or grant signing authority.

## Commercial boundary

Passing this gate means only that one exact candidate passed the configured held-out security contract. It is not a claim of AGI, infallibility, universal exploit detection or production readiness. Separate adversarial, shadow-replay, owner approval and deployment controls remain mandatory.
