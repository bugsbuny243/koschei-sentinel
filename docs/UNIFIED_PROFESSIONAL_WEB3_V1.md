# Koschei Sentinel — Unified Professional / Web3 Fabric v1

Status: integration contract

Koschei Sentinel remains an independent cybersecurity model, evaluation and inference project. It is commercially included in the unified **Koschei Professional** ecosystem together with Koschei Web3 Hub, ARVIS and Koschei Lang.

## Product boundary

- Koschei Web3 Hub is the customer security center.
- ARVIS owns deterministic Web3 evidence and the signed customer verdict.
- Sentinel receives bounded evidence cases through Koschei Fabric and returns evidence-grounded commentary.
- Sentinel output cannot replace, lower, rewrite or silently create ARVIS verdict authority.
- Sentinel model training, promotion, readiness and production-write gates remain independent and fail-closed.

## Authenticated Fabric API

Web3 uses the existing `POST /v1/opinions` contract with `sentinel.case.v1` input. Production requests require:

```env
APP_ENV=production
SENTINEL_API_TOKEN=<runtime-secret>
```

The matching Web3 service sends the token as an HTTP Bearer credential. Health/readiness endpoints remain available to deployment health probes.

The opinion response remains bound to the submitted `case_id` and ARVIS `verdict_signature`. Web3 rejects a response whose identity or signature does not match the submitted signed case.

## One official ecosystem token

The unified Koschei ecosystem has one official token: **KOSC** on Solana mainnet, mint `7X9V77axASFAV8hKqqn2EfyAz4Qz3tceN8iikfukLqy1`.

There is no separate Sentinel token. KOSC is a commercial ecosystem/settlement asset only. Token ownership or payment must never alter Sentinel evidence policy, training data acceptance, benchmark results, candidate promotion, inference authority or production-write safety gates.

## Professional inclusion

Professional is the commercial umbrella. Inclusion means eligible customers may receive Sentinel-backed capabilities when their runtime/release gates are enabled; it does not claim that every experimental Sentinel capability is production-enabled. Runtime availability must be reported separately from commercial entitlement.
