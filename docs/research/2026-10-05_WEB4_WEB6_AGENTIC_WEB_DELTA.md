# Web4–Web6 Research Delta — 2026-10-05

## Status discipline

`Web4`, `Web5`, and `Web6` are used here only as architecture horizons. They are not treated as standardized Web version numbers.

The concrete signals below come from active standards/community work around an agentic, decentralized, cryptographically verifiable Web. W3C Community Group work is exploratory/community specification work and must not be represented as a W3C Recommendation.

## Fresh findings

### Agent protocol remains centered on interoperable agent collaboration

The W3C AI Agent Protocol Community Group continues work on open protocols for agent discovery, identity, capability/intent exchange, negotiation, collaboration, security/privacy, and interoperability with existing Web protocols.

The group has an upcoming meeting on 2026-10-07. This makes the area active rather than a frozen design target.

### Agent identity is becoming a first-class security boundary

The W3C Agent Identity Registry Protocol Community Group is explicitly working on cryptographically verifiable agent identity bound to controlling organizations, authorization scope, DID/Verifiable Credential based identity, revocation/lifecycle, trust negotiation, integration profiles, and post-quantum requirements.

The group also anticipates coordination with IETF WIMSE.

## Sentinel implications

Sentinel should keep these as stable product invariants:

1. Never authorize an action from a model assertion alone.
2. Bind agent identity to controller identity and explicit authorization scope.
3. Make delegation attenuated, reconstructable, revocable, and auditable.
4. Treat credential lifecycle/revocation as runtime security state, not static configuration.
5. Keep protocol adapters replaceable as community drafts evolve.
6. Preserve the distinction between requested action, authorization decision, executor acknowledgement, and independently observed effect.
7. Record enough provenance to reproduce why a defensive action was permitted or denied.
8. Fail closed when identity, delegation, evidence, or effect observation is indeterminate.

## Product direction

These findings reinforce Sentinel's architecture as an autonomous defensive cybersecurity system:

`Observe → Correlate → Reason → Verify → Contain → Recover → Prove`

The reasoning engine proposes findings and defensive actions. Deterministic policy/verifier components retain authority. Executors receive only predefined defensive capabilities inside the authorized protection boundary. Independent observation verifies effects before an incident is considered contained.

## Sources reviewed

- W3C AI Agent Protocol Community Group summary and scope, reviewed 2026-10-05.
- W3C AI Agent Protocol Community Group calendar, reviewed 2026-10-05.
- W3C Agent Identity Registry Protocol Community Group summary and scope, reviewed 2026-10-05.

## Next engineering consequence

The next production evidence package should prove the architecture rather than merely describe it: versioned security corpus, live inference evidence, tenant-isolation evidence, containment/effect evidence, recovery evidence, CI evidence, and a reproducible conformance report bound into the production evidence manifest.