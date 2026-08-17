# Cyber Corpus v3 License Review Batch 0002

This batch promotes only sources with an exact immutable revision and verified repository-level licensing compatible with the Cyber Corpus v3 training policy.

## Approved in this batch

### Kubernetes website/documentation

- source id: `kubernetes.security.docs`
- repository: `kubernetes/website`
- pinned revision: `2a031a5f0a9382d26c25cb1e58016001a21ce7b6`
- license: CC-BY-4.0
- exact license evidence: repository `LICENSE` at the pinned revision
- decision: `ALLOW_WITH_ATTRIBUTION`
- acquisition scope: security-relevant English Kubernetes documentation only; localization copies, generated navigation, vendor assets, unrelated website code and separately licensed third-party material are excluded by extraction/review policy

### YARA engine and documentation

- source id: `yara.official.rules.docs`
- repository: `VirusTotal/yara`
- pinned revision: `604822da04103d13812dbcb08f4d7d42b61f94a8`
- license: BSD-3-Clause
- exact license evidence: repository `COPYING` at the pinned revision
- decision: `ALLOW_WITH_ATTRIBUTION`
- acquisition scope: first-party YARA language, engine, parser, documentation and security-relevant examples; this approval does not authorize unrelated third-party YARA rule repositories

## Held for narrower review

NIST, CISA, NVD and OSV remain artifact- or dataset-specific review items. A permissive website policy, code-repository license, or US-government publication policy is not treated as blanket permission for imported third-party artifacts or vulnerability records.

No corpus is sealed and no training run is authorized by this review batch.
