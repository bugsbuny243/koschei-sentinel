# Koschei Sentinel product direction — 2026-09-10

Koschei Sentinel is an independent cybersecurity model for secure software
development, vulnerability research, attack-path reasoning, security testing,
remediation and threat intelligence. Explaining ARVIS evidence is one use case.

The product architecture target remains one sparse/MoE model with 397B total and
35B active parameters, `koschei-sentinel-moe-397b-a35b`. External bootstrap models
remain trainer-proof tools and do not become the Sentinel product.

The owner-selected competitive target is to surpass the model they named
“Mythos 5.1” in cybersecurity. This record neither verifies that external model's
identity/specifications nor claims an achieved comparison. Real candidate-bound,
independent matched evaluations are required before superiority can be advertised.

## Lang + Sentinel commercial packages

Every commercial package contains both Koschei Lang and Koschei Sentinel.
The language remains independent; model inference, training and promotion gates
remain separate. No model call, paid compute, training launch or deployment is
authorized by a bundle integrity result.

`sentinel-bundle-check manifest.json --artifacts-dir ./artifacts` checks the same
contract as `ks-bundle-check`. Without `--artifacts-dir` it validates metadata
only. Source-only use is available as
`python src/koschei_sentinel/commercial_bundle_v1.py manifest.json`.

See [the bundle contract](COMMERCIAL_BUNDLE_V1.md) and
`fabric/product-direction.v1.json`. Gold, HOLDOUT, privacy, authority, provenance
and existing promotion/paid-compute gates remain unchanged. A trained, evaluated
397B/35B production checkpoint is not claimed by this packaging slice.
