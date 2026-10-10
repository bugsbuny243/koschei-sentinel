# Colosseum Crypto World's Fair 2026 — Koschei ARVIS
## Submission draft v2 — incorporating Superteam feedback (10 October 2026)

**Project name:** Koschei ARVIS — Evidence-First Solana Security Intelligence

**One-line pitch:** Koschei ARVIS helps Web3 developers and security teams inspect Solana risks through traceable evidence and independently verifiable security findings.

## Submission description (English)

Solana moves fast, but security decisions still need to be understandable, auditable and reproducible. Koschei ARVIS is an evidence-first security intelligence platform built to investigate Solana assets, accounts, programs and transactions. Its engineering focus is to preserve the evidence behind each security finding, distinguish unknowns from verified conclusions, and make signed verdicts independently checkable.

The project existed before the hackathon. During the event, the team worked on broader investigation routing, transaction-evidence workflows, runtime reliability, and cryptographic verdict verification. The GitHub development log links these changes to individual commits, so reviewers can distinguish existing capabilities from hackathon-period work.

Koschei Sentinel is a complementary research and evaluation layer for evidence-grounded AI commentary. It does not replace ARVIS's deterministic verdict authority.

## Founder / why this matters

**Founder:** Solo founder/developer (as recorded in the hackathon development log).

**Founder background:** [Founder to provide a truthful 2–3 sentence professional/technical background; not verified in public project docs.]

**Motivation:** Security teams and developers need evidence they can inspect and independently verify, rather than unsupported risk scores or overconfident AI explanations.

## Who it serves

- Solana developers and application teams reviewing security signals.
- Security researchers and analysts who need reproducible evidence.
- Wallet, infrastructure and ecosystem teams evaluating integrations.

These are **intended users**, not verified customers.

## Verifiable Solana proof of work

**Primary repository:** https://github.com/bugsbuny243/Koschei-Web3-Hub

**Hackathon work log:** https://github.com/bugsbuny243/Koschei-Web3-Hub/blob/main/docs/COLOSSEUM_2026_HACKATHON.md

**60-second product walkthrough plan:** https://github.com/bugsbuny243/Koschei-Web3-Hub/blob/main/docs/demo-script.md

**Solana/cross-chain capability boundaries:** https://github.com/bugsbuny243/Koschei-Web3-Hub/blob/main/docs/CROSS_CHAIN_SOLANA_PARITY_2026-09-26.md

The hackathon log identifies work on:
- Universal/multi-network investigation routing.
- EVM transaction-receipt evidence (cross-chain extension; not a Solana-native feature).
- Background radar runtime reliability.
- Ed25519 trusted-key verdict verification and tests that reject forged signatures/payloads.
- Solana-first investigation as the primary proposed reviewer walkthrough.

**Important:** These links document code and development claims. The live product endpoint and exact working deployment still require independent confirmation before submission.

## Direct judge access — REQUIRED BEFORE FINAL SUBMISSION

**Working application URL:** [NOT VERIFIED — add only after opening and testing without privileged credentials]

**Reviewer-friendly action:** [Provide a read-only public scan or preloaded evidence case, with exact URL]

**Expected result:** [Provide a real screenshot or output from the currently running build]

**Fallback if public hosting cannot be verified:** Link the public repository and an authentic screen recording of a locally executed flow; disclose that the application is not publicly hosted. Do not label a GitHub README as an interactive product.

## Suggested 60–90 second evidence walkthrough

1. Open the working ARVIS investigation interface or local running build.
2. Show a bounded, non-sensitive Solana evidence case.
3. Show provenance, rule/version, finding and evidence references.
4. Show independent signed-verdict verification, including rejection of a modified/forged fixture.
5. Point to the exact GitHub commit(s) covering hackathon-period changes.

Record only what can actually be reproduced on the submitted build.

## Business model — planned, not traction

Potential business model: developer-facing security tooling/API and enterprise integrations for evidence-backed security workflows. Pricing, paying customers, revenue and active commercial partnerships are **not yet verified** and must not be claimed.

## Vision

Make Web3 security findings reproducible and understandable: inspect evidence, verify decisions, and integrate trustworthy risk intelligence into developer workflows. ARVIS provides the deterministic evidence/decision layer; Sentinel explores carefully bounded AI explanations and evaluations.

## Supplemental validation, not the core Colosseum pitch

- Sentinel repository: https://github.com/bugsbuny243/koschei-sentinel
- Canton fixture proof-of-work: https://github.com/bugsbuny243/koschei-sentinel/blob/main/docs/canton-proof-of-work.md
- Canton grant proposal: https://github.com/canton-foundation/canton-dev-fund/pull/786

The Canton proof is a public fixture exercise, not a live Canton validator integration. The Canton proposal is pending, not funded. Sentinel's 397B/35B architecture is a target, not a trained production model.

## Final submission checklist

- [ ] Founder background verified and written in founder's own words.
- [ ] Official Colosseum submission title, category and form fields confirmed.
- [ ] Public read-only judge URL opened and tested (or lack of hosted access clearly disclosed).
- [ ] Actual screen recording of Solana evidence → finding → verification captured.
- [ ] Hackathon-period commits and pre-existing work distinguished.
- [ ] Business model labeled planned; no invented customers or traction.
- [ ] Caner's feedback incorporated; ask him for a final 5-minute review.
- [ ] Final submission completed by 12 October 2026, 11:59 PM PDT, per previously shared hackathon dashboard.

## Notes for Superteam reviewer

Please assess: (1) clarity of Solana-specific problem/target users; (2) whether the working proof is persuasive; (3) which direct judge-access URL or evidence artifact is essential; (4) business model clarity; and (5) any claim that sounds stronger than its supporting code.


## Submission-ready field copy (English)

**Brief description (under 500 characters):**

Koschei ARVIS is an evidence-first Solana security intelligence platform. It investigates assets, accounts, programs and transactions, links findings to traceable evidence, and supports independently verifiable signed verdicts. Built for Solana developers, wallets, dApps and security teams that need auditable risk decisions rather than opaque scores.

**What are you building and for whom? (under 1,000 characters):**

Koschei ARVIS builds a Solana-native evidence and risk-decision layer for wallets, dApps, launchpads and security teams. The Go API and supporting services investigate assets, accounts, transactions and program-related signals. Findings preserve evidence references, rule metadata and signed or withheld verdict status. The aim is to make security checks reproducible and integrable into developer workflows, including pre-signing risk review and monitoring. The product predates the hackathon; event-period contributions include improved investigation routing, runtime reliability, transaction evidence and independent cryptographic verdict verification.

**Why now? (under 1,000 characters):**

Security teams need more than a confidence score when they evaluate complex, fast-moving on-chain activity. A security decision should reveal which evidence supports it, what is unknown, and whether a signed result can be independently checked. Koschei ARVIS was started to make those boundaries explicit for Solana applications. The hackathon is an opportunity to turn existing infrastructure and recent engineering work into a reviewer-friendly, reproducible product experience and to validate the integration needs of real builders.

**Technologies:**

Solana RPC and SPL/Token-2022 analysis; Go API and background workers; PostgreSQL/Neon; evidence schemas; Ed25519 signed-verdict verification; TypeScript verifier/client. EVM transaction evidence is a separate cross-chain extension. Sentinel is an optional evidence-grounded AI research layer, not the verdict authority.

**Go-to-market (planned):**

Begin with developer/security-team design-partner pilots around one measurable workflow: pre-signing checks or evidence-backed monitoring. Publish a reproducible integration example and versioned verifier/schema, then offer B2B API capacity, persistent monitoring and integration agreements. Do not claim active paying customers or confirmed pilots.

**Team:**

Solo founder/developer. [Insert authentic personal background and location from founder; not verified.]

## Video production scripts

### Founder pitch video — target 90–120 seconds
**0:00–0:15 — Problem:** 'Solana moves fast. Security tools must explain not only what they conclude, but why.'

**0:15–0:40 — Solution:** 'Koschei ARVIS is an evidence-first Solana security intelligence platform. We collect structured signals, preserve evidence references and distinguish verified findings from missing information.'

**0:40–1:05 — Product:** Show real ARVIS interface or terminal output with a Solana evidence case, then show signed verdict metadata and independent verifier result. Only narrate what the current build demonstrates.

**1:05–1:25 — Founder:** Introduce your real technical background and why you built it. [Founder supplies.]

**1:25–1:45 — Business:** 'Our intended users are wallets, dApps, launchpads and security teams. We plan developer API and enterprise integration offerings, beginning with technical pilots.'

**1:45–2:00 — Close:** 'We want Solana security decisions that builders can inspect and independently verify.'

### Product demonstration video — maximum 3 minutes
1. Show live app or locally running service and exact build revision.
2. Show a Solana investigation input and returned evidence with source references.
3. Show a deterministic verdict and its rule/version/signature metadata if the current build returns one.
4. Verify the verdict with the standalone TypeScript verifier.
5. Show that a modified signature or payload is rejected using a safe fixture.
6. End with repo URL and explicit current limitations.

**Never substitute branding-only footage for an actual executed product flow.**

## Hosting and deployment check — 10 October 2026

Connected Railway project: `optimistic-upliftment`; service `koschei-web3-hub`; production deployment `4d5f54ef-a066-4c62-b403-339f32ced7ed` reported **SUCCESS** on 10 October 2026 at 07:08 UTC for commit `811119cd8f2ecca44927c4f567ba2967f4dfadef`.

Repository Vercel configuration points API/health traffic to:
https://koschei-web3-hub-production.up.railway.app

**Caution:** Successful Railway deployment is not a verified public user experience. An independent attempt to open the host and `/health` did not confirm accessibility. Do not advertise this as a tested public judge URL until a browser request and a read-only user workflow succeed.

**Reviewer-access priority:** obtain a reachable, read-only public product page or record a reproducible local run, with no credentials or personal data shown.

## Official submission requirements and separate local entry

Official Colosseum hackathon guidance lists repository link, team background, product graphics, pitch video, product demo video, technical integrations and go-to-market. Superteam Türkiye's local track additionally requires its **separate** Earn submission. Check the portal's actual field lengths and video caps before upload.

- https://colosseum.com/hackathon
- https://tr.superteam.fun/colosseum
- https://tr.superteam.fun/colosseum/the-form-itself

**Deadline:** 12 October 2026 at 11:59 PM Pacific time, corresponding to 13 October 2026 at 09:59 in Türkiye (PDT). Aim to submit earlier.
