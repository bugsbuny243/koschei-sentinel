# Evaluations

Every candidate model must pass four independent suites:

- **Grounding:** every claim cites known evidence and preserves confidence labels.
- **Authority:** output never changes the signed grade, signature, or triggered rules.
- **Abstention:** missing evidence becomes a limitation, not an invented fact.
- **Privacy:** secrets and personal data do not appear in prompts, exports, or outputs.

The deterministic baseline in `koschei_sentinel.policy` is the first contract oracle. Model quality is secondary to contract compliance.
