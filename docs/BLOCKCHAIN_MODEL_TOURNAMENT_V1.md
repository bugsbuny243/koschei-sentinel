# Blockchain Model Tournament v1

Status: offline candidate-selection contract  
Production authority: none  
Automatic promotion: disabled

## Goal

Choose the strongest practical blockchain-security candidate only after every candidate has been trained on comparable lineage and independently passed the multi-chain held-out security gate.

The tournament is deliberately **security first**. A smaller/faster model may win only when its security floor is at least as strong as its competitors. Runtime efficiency is a tie-breaker after hard security quality, never a substitute for it.

## Entry requirements

Every candidate entry binds four immutable artifacts:

1. `sentinel.blockchain-adapter-manifest.v1`;
2. `sentinel.blockchain-security-eval-receipt.v1`;
3. the stored `sentinel.blockchain-security-eval-audit.v1` recomputed from the receipt;
4. `sentinel.blockchain-runtime-profile.v1` from a common runtime suite/environment.

A candidate is rejected if its adapter consumed the held-out test split, its stored audit is stale, its receipt does not match adapter lineage, or its runtime profile does not bind the same adapter digest.

## Apples-to-apples boundary

A ready tournament requires all candidates to share exactly one:

- source corpus digest;
- held-out test split digest;
- benchmark suite digest;
- evaluator digest;
- runtime suite digest;
- runtime environment digest.

This prevents comparing one model on an easier benchmark, another on a different corpus, or a third on faster hardware and calling that a fair model race.

## Security-first ranking

Only candidates whose blockchain security evaluation audit is `ready: true` enter the ranking.

The ranking is lexicographic:

1. highest **security floor** — the worst value among overall, worst-chain, worst-threat, worst-task, grounding, task-correctness, patch-safety and abstention-correctness rates;
2. highest sum of those security metrics;
3. highest overall held-out pass rate;
4. highest measured generation throughput;
5. lowest p95 case latency;
6. lowest peak GPU memory;
7. candidate ID only as a deterministic final tie-breaker.

This ordering means a very fast candidate with a weak security blind spot cannot beat a slower candidate with a stronger worst-case security surface.

## Runtime receipt

The runtime profile stores only aggregate measurements:

- measured case count;
- prompt/generated token totals;
- total wall time;
- p95 case latency;
- peak GPU memory;
- exact runtime-suite and environment digests.

Raw model outputs are forbidden from the runtime profile.

## Repository policy

`configs/eval/blockchain-model-tournament.v1.json` currently requires at least three candidates and at least three candidates that already pass the blockchain-security hard gate. Duplicate adapter digests are rejected.

The tournament winner is still only a research candidate. The result explicitly keeps automatic production promotion disabled. Shadow research, adversarial evaluation and owner-controlled promotion remain separate gates.

## Command

```bash
sentinel-blockchain-tournament \
  --tournament build/tournaments/candidates.json \
  --policy configs/eval/blockchain-model-tournament.v1.json \
  --eval-policy configs/eval/blockchain-security.v1.json \
  --root . \
  --output build/tournaments/result.json
```

The command performs no training, no network fetch, no signing and no deployment.
