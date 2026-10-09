# Koschei Sentinel — Colosseum Crypto World's Fair 2026
## Submission review pack | Draft for Superteam Türkiye feedback | 9 October 2026

**Status:** Reviewer draft. Verify live submission fields and links before final submission. **Deadline:** 12 October 2026, 11:59 PM PDT (per hackathon dashboard previously shared).

### 1. Project title
Koschei Sentinel — Evidence-Grounded Security Intelligence for Web3

### 2. One-line pitch (English)
Koschei Sentinel is building evidence-grounded cybersecurity intelligence for Web3: privacy-aware security data pipelines, reproducible evaluation, and bounded AI explanations that cannot override deterministic security verdicts.

### 3. Short description (English)
Web3 security tools often surface opaque risk labels or AI narratives that are hard to audit. Koschei Sentinel is building an evidence-first cybersecurity platform that separates signed, deterministic security decisions from model-generated explanations. Its repository includes structured evidence ingestion, privacy sanitization and pseudonymization, versioned dataset release checks, leakage-safe splits, offline evaluation, model adapter and training infrastructure, and an inference API. A Canton-specific proof-of-work demonstrates deterministic evidence identifiers, rule-based findings and explicit insufficient-evidence outcomes on a public test fixture. We are seeking ecosystem feedback and technical design partners to validate practical Web3 security workflows.

### 4. Problem
- Risk assessments can be difficult to reproduce and audit.
- Model-generated explanations can overstate what evidence supports.
- Privacy, provenance and dataset leakage matter for security-model development.

### 5. Approach
1. **Koschei Web3 / ARVIS:** structured evidence and signed deterministic security verdicts.
2. **Koschei Sentinel:** evidence-grounded analysis, bounded explanation, evaluation and abstention.
3. **Koschei Lang:** independent secure-development language/runtime direction; not a prerequisite for the currently demonstrated Sentinel pipeline.

**Core safety contract:** AI commentary never replaces a signed deterministic verdict; claims must cite known evidence IDs; unknown evidence is reported as a limitation.

### 6. Verifiable work
- Repository: https://github.com/bugsbuny243/koschei-sentinel
- Architecture: https://github.com/bugsbuny243/koschei-sentinel/blob/main/docs/ARCHITECTURE.md
- README with commands for local setup, baseline evaluation, API and dataset readiness: https://github.com/bugsbuny243/koschei-sentinel/blob/main/README.md
- Canton proof of work: https://github.com/bugsbuny243/koschei-sentinel/blob/main/docs/canton-proof-of-work.md
- Canton implementation merged in PR #102: https://github.com/bugsbuny243/koschei-sentinel/pull/102
- Merge commit: https://github.com/bugsbuny243/koschei-sentinel/commit/9b44d332b422de0849c1b098f0d0779acfe58950
- Canton grant proposal (separate application, not funding received): https://github.com/canton-foundation/canton-dev-fund/pull/786

### 7. What is demonstrable today
- Repository-level dataset privacy and readiness contracts, benchmark and inference infrastructure.
- Reproducible Canton fixture-to-evidence-to-finding pipeline with SHA-256 evidence IDs, explicit missing-evidence handling, CLI and tests.
- The Canton slice is **not** a live validator integration or a production security audit.

### 8. Maturity and limitations
- 397B total / 35B active parameters is an **architecture target**, not a trained or deployed model.
- Qwen2.5 1.5B LoRA/QLoRA configuration and trainer are infrastructure, not proof of a trained production-grade model.
- No verified customer count, revenue, deployed Solana integration, production Canton integration or benchmark superiority is claimed.
- Recent repository-wide CI had failures in ci, launch-gates and pq-evidence-review. Do not label all CI green; check latest runs before submission.
- Existing work is real code and tests; production readiness remains to be validated.

### 9. Why Solana / Colosseum?
Koschei Sentinel's evidence, policy and risk-analysis boundaries are relevant to Web3 applications and security workflows. The Colosseum track is an opportunity to validate a Solana-oriented use case with ecosystem builders, obtain feedback on operator needs and recruit technical design partners. **A production Solana integration is not yet established by the evidence listed here.**

### 10. Proposed next milestones
- Identify and document a narrowly scoped Solana security evidence source and reference workflow.
- Produce a reproducible end-to-end evidence → verdict → Sentinel commentary walkthrough.
- Publish evaluation criteria, limitations and independently repeatable benchmark cases.
- Obtain technical feedback and pursue pilot/design-partner discussions.

### 11. Materials checklist before final submission
- [ ] Confirm actual Colosseum project category, title and required form fields.
- [ ] Attach a short screen recording of a real command/API run, not just a branding animation.
- [ ] Show input fixture, evidence IDs, deterministic finding and insufficient-evidence behavior.
- [ ] Include a current GitHub commit permalink and note any failing CI.
- [ ] Confirm team, project contact, license, live product URL (if any) and submission rules.
- [ ] Ask Caner to review clarity, fit, proof and missing submission requirements.
- [ ] Submit through the official dashboard before deadline.

### 12. Questions for Caner
1. Is the problem and the Solana-specific application clear enough for the Colosseum judges?
2. What is the single most important missing technical proof before submission?
3. Is there a relevant Solana security team or ecosystem builder who could review the evidence workflow?
4. Which submission track/category would best fit this maturity level?

**Reviewer note:** This is a factual draft, not an assertion of a completed hackathon submission, grant award or live Solana deployment.
