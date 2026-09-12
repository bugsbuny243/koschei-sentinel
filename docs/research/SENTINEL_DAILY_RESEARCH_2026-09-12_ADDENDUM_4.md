# Sentinel Daily Research — 2026-09-12 Addendum 4

Scope: defensive cybersecurity research only. Driver remains out of scope.

## 1. Formal verification becomes cross-cutting Ethereum security tooling

Evidence maturity: A — canonical Ethereum Foundation / ethereum.org material.

Ethereum's September 2026 protocol priorities explicitly elevate formal verification as tooling shared across fast-finality, privacy, state, zkEVM, and post-quantum work. The L1 zkEVM path also changes the verification model from validator re-execution toward succinct execution proofs.

Sentinel primitives:
- proof statement vs intended security property
- verifier correctness and version provenance
- circuit/specification equivalence
- trusted parameters and cryptographic assumptions
- proof freshness and block/state binding
- proof availability vs prover centralization

Evaluation rule: `proof_verified == true` MUST NOT imply the surrounding application, policy, signer set, or economic action is safe.

## 2. EIP-7702 is delegation, not least-privilege permissioning

Evidence maturity: A — canonical EIP.

EIP-7702 gives EOAs delegated code behavior but its security notes explicitly warn that delegated code has unrestricted account access at this layer. Fine-grained permission systems must live in standardized modules/extensions above the delegation.

Sentinel primitives:
- delegated-code provenance
- delegation target auditability
- master-key fallback authority
- delegation replacement/reset path
- module permission enforcement
- delegation-chain / loop detection

Evaluation rule: detect the false claim `7702 delegation => bounded session key` unless an independently verified permission layer proves the bound.

## 3. ERC-7715 permission requests are useful but still draft

Evidence maturity: C — ERC Draft.

ERC-7715 defines wallet-requested execution permissions with chain scope, target session account, permission type, optional rules such as expiry, revocation, and delegation-manager context.

Sentinel primitives:
- requested permission vs actually granted permission
- chain-id binding
- target account binding
- allowance/value ceiling
- expiry / revocation
- permission type collision / semantic ambiguity
- delegation-manager provenance
- dependency/factory provenance

Evaluation rule: require comparison of requested vs returned permissions because the wallet response is not guaranteed to be identical to the request.

## 4. Continuous remote attestation is stronger than one-shot TEE claims

Evidence maturity: B/C — Confidential Computing Consortium public draft-review material.

The CCC's 2026 work includes a Continuous Remote Attestation Framework draft under public review. This reinforces the Sentinel rule that one historical attestation should not become a permanent trust certificate.

Sentinel primitives:
- attestation freshness
- measurement identity
- expected-code binding
- verifier policy
- signer / endorsement chain
- TCB version and revocation
- runtime drift and re-attestation

Evaluation rule: stale or policy-unbound attestation must force bounded confidence or abstention.

## 5. x402 is becoming governed infrastructure for agent payments

Evidence maturity: A/B — Linux Foundation governance plus deployed ecosystem integrations.

In July 2026 the Linux Foundation announced operational launch of the x402 Foundation with open governance and a broad member set. Earlier 2026 work expanded x402 toward ERC-20 assets and gas-sponsorship flows; AWS integration is a deployed signal that agent payments are moving into ordinary web/API infrastructure.

Sentinel primitives:
- tool authorization != payment authorization
- token / chain / recipient binding
- spending limit and expiry
- Permit2 / approval scope
- gas sponsor / paymaster trust
- payment receipt -> requested action binding
- facilitator provenance
- replay / duplicate payment handling

Evaluation rule: an agent's ability to call an API must never be interpreted as authority to spend arbitrary funds.

## 6. AI-agent vulnerability research needs evidence triage, not model prestige

Evidence maturity: A — Ethereum Foundation Protocol Security reporting.

Ethereum Foundation Protocol Security reported running coordinated AI agents against real protocol code and described triage quality as the central operational problem. The useful lesson for Sentinel is not 'AI found bugs' but that findings must survive reproduction, source-level evidence, exploitability review, affected-version analysis, and remediation verification.

Sentinel primitives:
- candidate finding -> reproduced finding boundary
- false-positive suppression
- affected-version binding
- crash/bug vs security-impact distinction
- evidence packet completeness
- remediation regression proof

Evaluation rule: model-generated claims remain hypotheses until deterministic or reviewed evidence promotes them.

## 7. Post-quantum migration remains a multi-surface problem

Evidence maturity: A — ethereum.org roadmap.

Ethereum continues to identify BLS consensus signatures, KZG commitments, account ECDSA, and some ZK systems as separate migration surfaces. Native account abstraction improves account-signature agility, but consensus/data cryptography still require protocol migrations.

Sentinel primitives:
- crypto inventory by layer
- classical fallback detection
- downgrade path detection
- long-lived signature/key exposure
- bridge validator / recovery signer migration
- PQ algorithm maturity and parameter provenance

Evaluation rule: `wallet is PQ-safe` MUST NOT imply `protocol/bridge/system is PQ-safe`.

## New curriculum families

- `formal_verification_and_proof_statement_binding`
- `delegated_code_and_wallet_permission_security`
- `continuous_remote_attestation_and_runtime_drift`
- `agent_payment_authorization_and_receipt_binding`
- `ai_security_finding_triage_and_reproduction`
- `multi_surface_post_quantum_migration`

## Training admission requirements

Every admitted case should include:
1. canonical source and date/version,
2. maturity class: standard / deployed / draft / research,
3. actor and authority graph,
4. concrete security primitive,
5. expected evidence,
6. unsupported inference to reject,
7. abstention condition,
8. remediation or next verification step,
9. leakage-safe family/group identifier.

## Source set

- https://blog.ethereum.org/2026/09/07/protocol-priorities
- https://ethereum.org/roadmap/security/
- https://ethereum.org/roadmap/zkevm/
- https://eips.ethereum.org/EIPS/eip-7702
- https://eips.ethereum.org/EIPS/eip-7715
- https://confidentialcomputing.io/2026/
- https://www.linuxfoundation.org/press/linux-foundation-announces-operational-launch-of-x402-foundation-to-standardize-internet-native-payments-for-ai-agents-and-applications
- https://www.coinbase.com/developer-platform/discover/launches/x402-ERC20
- https://blog.ethereum.org/2026/07/09/triage-is-the-product
- https://ethereum.org/roadmap/security/quantum-resistance/
